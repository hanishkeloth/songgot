"""Songgot trainer on Apple silicon with MLX. Same tokenizer, shards, config and text format as
the Modal path; saves an HF-compatible Llama checkpoint (config.json + model.safetensors).

    .venv/bin/python harness/train_mlx.py bench --seconds 45 --layers 8
    .venv/bin/python harness/train_mlx.py pretrain --tokens 6e8 --layers 8 --batch 32
    .venv/bin/python harness/train_mlx.py sft --data data/sft_train.jsonl --init vol/ckpt/pre_mlx/final
"""
from __future__ import annotations

import argparse, json, math, os, pathlib, random, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.models import llama as L

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOL = pathlib.Path(os.environ.get("SONGGOT_VOL", ROOT / "vol"))
sys.path.insert(0, str(ROOT / "harness"))
import songgot_modal as sm  # noqa: E402

SEQ = 1024; VOCAB = 32000


def args_for(layers, d, inter, heads, kv):
    return L.ModelArgs(model_type="llama", hidden_size=d, num_hidden_layers=layers, intermediate_size=inter,
                       num_attention_heads=heads, num_key_value_heads=kv, vocab_size=VOCAB, rms_norm_eps=1e-5,
                       rope_theta=10000.0, tie_word_embeddings=True, max_position_embeddings=2048)


def hf_config(a: L.ModelArgs) -> dict:
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": a.hidden_size, "intermediate_size": a.intermediate_size,
            "num_hidden_layers": a.num_hidden_layers, "num_attention_heads": a.num_attention_heads, "num_key_value_heads": a.num_key_value_heads,
            "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 2048, "tie_word_embeddings": True,
            "hidden_act": "silu", "bos_token_id": 1, "eos_token_id": 2, "attention_bias": False, "mlp_bias": False, "torch_dtype": "bfloat16",
            "transformers_version": "4.51.3"}


def save_hf(model, a, out: pathlib.Path):
    out.mkdir(parents=True, exist_ok=True)
    weights = {k: v for k, v in tree_flatten(model.parameters())}
    weights = {(k if k.startswith("model.") or k.startswith("lm_head") else "model." + k): v.astype(mx.bfloat16) for k, v in weights.items()}
    mx.save_safetensors(str(out / "model.safetensors"), weights, {"format": "pt"})
    json.dump(hf_config(a), open(out / "config.json", "w"), indent=1)
    import shutil; shutil.copy(VOL / "tok" / "spm.model", out / "tokenizer.model")
    json.dump({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<|pad|>", "unk_token": "<unk>", "additional_special_tokens": sm.SPECIAL,
               "model_max_length": 2048, "tokenizer_class": "LlamaTokenizer", "legacy": False}, open(out / "tokenizer_config.json", "w"), indent=1)


def load_hf(init: pathlib.Path, a: L.ModelArgs):
    model = L.Model(a)
    w = mx.load(str(init / "model.safetensors"))
    w = {k[len("model."):] if k.startswith("model.") else k: v for k, v in w.items()}
    model.load_weights(list(w.items()), strict=False)
    return model


def shards():
    tokd = VOL / "tok"; out = {}
    for lang in ("ko", "en"):
        fs = sorted(tokd.glob(f"{lang}_*.bin"))
        out[lang] = [np.memmap(f, dtype=np.uint16, mode="r") for f in fs if f.stat().st_size > 2 * SEQ * 4]
    return out


class Sampler:
    def __init__(self, mm, p_ko, B, seed=0):
        self.mm, self.p_ko, self.B = mm, p_ko, B; self.rng = np.random.default_rng(seed)
        self.w = {l: np.array([len(m) for m in ms], dtype=np.float64) for l, ms in mm.items()}
        for l in self.w:
            if self.w[l].sum():
                self.w[l] = self.w[l] / self.w[l].sum()
    def batch(self):
        x = np.empty((self.B, SEQ + 1), dtype=np.int32)
        for i in range(self.B):
            lang = "ko" if (self.rng.random() < self.p_ko and self.mm["ko"]) else ("en" if self.mm["en"] else "ko")
            m = self.mm[lang][self.rng.choice(len(self.mm[lang]), p=self.w[lang])]
            off = int(self.rng.integers(0, len(m) - SEQ - 1)); x[i] = m[off:off + SEQ + 1]
        t = mx.array(x); return t[:, :-1], t[:, 1:]


def loss_fn(model, x, y, mask=None):
    logits = model(x)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
    if mask is None:
        return ce.mean()
    return (ce * mask).sum() / mx.maximum(mask.sum(), 1)


def lr_sched(steps, warm, lr_max):
    lr_min = lr_max * 0.1
    def f(s):
        if s < warm:
            return lr_max * (s + 1) / warm
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    return f


def cmd_bench(a):
    mm = shards(); args = args_for(a.layers, a.d, a.inter, a.heads, a.kv); model = L.Model(args)
    if a.bf16:
        model.set_dtype(mx.bfloat16)
    n = sum(v.size for _, v in tree_flatten(model.parameters())); print(f"[mlx] params {n/1e6:.1f}M layers {a.layers}", flush=True)
    opt = optim.AdamW(learning_rate=1e-3); vg = nn.value_and_grad(model, loss_fn); s = Sampler(mm, 0.5, a.batch)
    t0 = time.time(); seen = 0; step = 0
    while time.time() - t0 < a.seconds:
        x, y = s.batch(); loss, grads = vg(model, x, y); opt.update(model, grads); mx.eval(model.parameters(), opt.state, loss)
        seen += x.size; step += 1
        if step == 3:
            t0 = time.time(); seen = 0
    el = time.time() - t0
    print(f"[bench] layers {a.layers} d {a.d} batch {a.batch}: {seen/el/1e3:.1f}k tok/s -> 1B tokens in {1e9/(seen/el)/3600:.1f} h; loss {loss.item():.3f}")


def cmd_pretrain(a):
    mm = shards(); args = args_for(a.layers, a.d, a.inter, a.heads, a.kv); model = L.Model(args)
    if a.bf16:
        model.set_dtype(mx.bfloat16)
    n = sum(v.size for _, v in tree_flatten(model.parameters())); print(f"[mlx] params {n/1e6:.1f}M", flush=True)
    steps = int(a.tokens // (a.batch * SEQ)); sched = lr_sched(steps, min(500, steps // 20), a.lr)
    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.95], weight_decay=0.1); vg = nn.value_and_grad(model, loss_fn); s = Sampler(mm, a.p_ko, a.batch)
    ck = VOL / "ckpt" / "pre_mlx"; ck.mkdir(parents=True, exist_ok=True); log = open(VOL / "pretrain_mlx.log", "a")
    t0 = time.time(); seen = 0; acc = 0.0; k = 0
    for step in range(steps):
        opt.learning_rate = sched(step)
        x, y = s.batch(); loss, grads = vg(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads); mx.eval(model.parameters(), opt.state, loss)
        acc += loss.item(); k += 1; seen += x.size
        if step % 20 == 0 or step == steps - 1:
            el = time.time() - t0
            msg = f"[pre] step {step}/{steps} loss {acc/k:.4f} lr {sched(step):.2e} {seen/el/1e3:.1f}k tok/s elapsed {el/60:.1f}m eta {(steps-step-1)*el/max(1,step+1)/60:.0f}m"
            print(msg, flush=True); log.write(time.strftime("%F %T ") + msg + "\n"); log.flush(); acc = 0.0; k = 0
        if a.ckpt_every and step > 0 and step % a.ckpt_every == 0:
            save_hf(model, args, ck / f"hf_step{step}")
    save_hf(model, args, ck / "final"); print("[pre] DONE", flush=True)


def cmd_sft(a):
    init = pathlib.Path(a.init); cfg = json.load(open(init / "config.json"))
    args = args_for(cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"])
    model = load_hf(init, args)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(VOL / "tok" / "spm.model")); pad = sp.encode("<|pad|>")[0]
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]; random.Random(0).shuffle(rows)
    if a.limit:
        rows = rows[: a.limit]
    def encode(ex):
        p, c = sm.render(ex)
        pi = [sp.bos_id()] + sp.encode(p); ci = sp.encode(c) + [sp.eos_id()]
        return (pi + ci)[: a.max_len], ([0] * len(pi) + [1] * len(ci))[: a.max_len]
    enc = [encode(r) for r in rows]; enc.sort(key=lambda e: len(e[0]))
    nb = len(enc) // a.batch; steps = a.epochs * nb; sched = lr_sched(steps, 100, a.lr)
    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.95], weight_decay=0.05); vg = nn.value_and_grad(model, loss_fn)
    log = open(VOL / "sft_mlx.log", "a"); t0 = time.time(); s = 0
    print(f"[sft] {len(enc)} examples, {steps} steps", flush=True)
    for ep in range(a.epochs):
        order = list(range(nb)); random.Random(ep).shuffle(order)
        for bi in order:
            chunk = enc[bi * a.batch:(bi + 1) * a.batch]; Lm = max(len(e[0]) for e in chunk)
            ids = np.full((len(chunk), Lm), pad, dtype=np.int32); msk = np.zeros((len(chunk), Lm), dtype=np.float32)
            for j, (x, m) in enumerate(chunk):
                ids[j, :len(x)] = x; msk[j, :len(m)] = m
            X = mx.array(ids); x, y, mask = X[:, :-1], X[:, 1:], mx.array(msk)[:, 1:]
            opt.learning_rate = sched(s); loss, grads = vg(model, x, y, mask); grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(model, grads); mx.eval(model.parameters(), opt.state, loss)
            if s % 20 == 0:
                msg = f"[sft] ep {ep} step {s}/{steps} loss {loss.item():.4f} elapsed {(time.time()-t0)/60:.1f}m"
                print(msg, flush=True); log.write(time.strftime("%F %T ") + msg + "\n"); log.flush()
            s += 1
    out = VOL / "ckpt" / "sft_mlx" / "final"; save_hf(model, args, out); print("[sft] DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["bench", "pretrain", "sft"])
    ap.add_argument("--layers", type=int, default=8); ap.add_argument("--d", type=int, default=512); ap.add_argument("--inter", type=int, default=1408)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--kv", type=int, default=2); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seconds", type=int, default=45); ap.add_argument("--bf16", action="store_true"); ap.add_argument("--tokens", type=float, default=6e8); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--p-ko", type=float, default=0.5); ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--data"); ap.add_argument("--init"); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); {"bench": cmd_bench, "pretrain": cmd_pretrain, "sft": cmd_sft}[a.cmd](a)
