"""Songgot local trainer for Apple silicon (MPS). Same model, tokenizer and data format as the
Modal path, so a Mac-trained checkpoint is a drop-in for eval and export.

    .venv/bin/python harness/train_local.py bench                       # measure tok/s for 60 s
    .venv/bin/python harness/train_local.py pretrain --tokens 4e8 --layers 8
    .venv/bin/python harness/train_local.py sft --data data/sft_train.jsonl --init vol/ckpt/pre/final
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOL = pathlib.Path(os.environ.get("SONGGOT_VOL", ROOT / "vol"))
sys.path.insert(0, str(ROOT / "harness"))
import songgot_modal as sm  # noqa: E402  (model config, tokenizer wrapper, render)

SEQ = 1024


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def config(layers: int, d: int, inter: int, heads: int, kv: int):
    c = sm.make_config()
    c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads = layers, d, inter, heads, kv
    return c


def shards():
    tokd = VOL / "tok"
    out = {}
    for lang in ("ko", "en"):
        fs = sorted(tokd.glob(f"{lang}_*.bin"))
        out[lang] = [np.memmap(f, dtype=np.uint16, mode="r") for f in fs if f.stat().st_size > 2 * SEQ * 4]
    return out


class Sampler:
    def __init__(self, mm, p_ko, B, seed=0):
        self.mm, self.p_ko, self.B = mm, p_ko, B
        self.rng = np.random.default_rng(seed)
        self.w = {l: np.array([len(m) for m in ms], dtype=np.float64) for l, ms in mm.items()}
        for l in self.w:
            self.w[l] = self.w[l] / self.w[l].sum() if self.w[l].sum() else self.w[l]

    def batch(self, dev):
        x = np.empty((self.B, SEQ + 1), dtype=np.int64)
        for i in range(self.B):
            lang = "ko" if (self.rng.random() < self.p_ko and self.mm["ko"]) else ("en" if self.mm["en"] else "ko")
            m = self.mm[lang][self.rng.choice(len(self.mm[lang]), p=self.w[lang])]
            off = int(self.rng.integers(0, len(m) - SEQ - 1))
            x[i] = m[off:off + SEQ + 1]
        t = torch.from_numpy(x).to(dev)
        return t[:, :-1], t[:, 1:]


def build(cfg, dev):
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM(cfg).to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"[local] params {n/1e6:.1f}M layers {cfg.num_hidden_layers} d {cfg.hidden_size}", flush=True)
    return model


def train_loop(model, opt, sampler, dev, steps, lr_max, warm, log, ck_dir, ck_every, tag):
    lr_min = lr_max * 0.1
    def lr_at(s):
        if s < warm:
            return lr_max * (s + 1) / warm
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    t0 = time.time(); seen = 0; acc = 0.0; n = 0
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = sampler.batch(dev)
        with torch.autocast("mps", dtype=torch.bfloat16) if dev.type == "mps" else torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        acc += out.loss.item(); n += 1; seen += x.numel()
        if step % 20 == 0 or step == steps - 1:
            el = time.time() - t0
            msg = f"[{tag}] step {step}/{steps} loss {acc/n:.4f} lr {lr_at(step):.2e} {seen/el/1e3:.1f}k tok/s elapsed {el/60:.1f}m eta {(steps-step-1)*el/max(1,step+1)/60:.0f}m"
            print(msg, flush=True); log.write(time.strftime("%F %T ") + msg + "\n"); log.flush(); acc = 0.0; n = 0
        if ck_every and step > 0 and step % ck_every == 0:
            model.save_pretrained(ck_dir / f"hf_step{step}", safe_serialization=True)
    return model


def cmd_bench(a):
    dev = device(); mm = shards()
    cfg = config(a.layers, a.d, a.inter, a.heads, a.kv); model = build(cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s = Sampler(mm, 0.5, a.batch)
    t0 = time.time(); seen = 0; step = 0
    while time.time() - t0 < a.seconds:
        x, y = s.batch(dev)
        with torch.autocast("mps", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); seen += x.numel(); step += 1
        if step == 3:
            t0 = time.time(); seen = 0  # warm-up excluded
    el = time.time() - t0
    print(f"[bench] layers {a.layers} d {a.d} batch {a.batch}: {seen/el/1e3:.1f}k tok/s -> 1B tokens in {1e9/(seen/el)/3600:.1f} h; loss {out.loss.item():.3f}")


def cmd_pretrain(a):
    dev = device(); mm = shards()
    cfg = config(a.layers, a.d, a.inter, a.heads, a.kv); model = build(cfg, dev)
    decay = [p for p in model.parameters() if p.ndim >= 2]; nod = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nod, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    steps = int(a.tokens // (a.batch * SEQ))
    ck = VOL / "ckpt" / "pre"; ck.mkdir(parents=True, exist_ok=True)
    log = open(VOL / "pretrain_local.log", "a")
    print(f"[local] {steps} steps of {a.batch}x{SEQ} tokens, p_ko {a.p_ko}", flush=True)
    model = train_loop(model, opt, Sampler(mm, a.p_ko, a.batch), dev, steps, a.lr, min(500, steps // 20), log, ck, a.ckpt_every, "pre")
    model.save_pretrained(ck / "final", safe_serialization=True)
    sm.hf_tokenizer(str(VOL / "tok")).save_pretrained(ck / "final")
    print("[local] pretrain DONE", flush=True)


def cmd_sft(a):
    from transformers import LlamaForCausalLM
    dev = device(); tok = sm.hf_tokenizer(str(VOL / "tok"))
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    model = LlamaForCausalLM.from_pretrained(a.init).to(dev)
    if model.get_input_embeddings().num_embeddings < len(tok):
        model.resize_token_embeddings(len(tok))
    pad = tok.pad_token_id
    def encode(ex):
        p, c = sm.render(ex)
        pi = tok(p, add_special_tokens=True)["input_ids"]; ci = tok(c, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        return (pi + ci)[: a.max_len], ([-100] * len(pi) + ci)[: a.max_len]
    enc = [encode(r) for r in rows]
    enc.sort(key=lambda e: len(e[0]))  # length-bucketed batches, shuffled per epoch below
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    nb = len(enc) // a.batch; steps = a.epochs * nb; s = 0; log = open(VOL / "sft_local.log", "a"); t0 = time.time()
    model.train()
    for ep in range(a.epochs):
        order = list(range(nb)); random.Random(ep).shuffle(order)
        for bi in order:
            chunk = enc[bi * a.batch:(bi + 1) * a.batch]; L = max(len(x[0]) for x in chunk)
            ids = torch.full((len(chunk), L), pad); lab = torch.full((len(chunk), L), -100); att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, (x, y) in enumerate(chunk):
                ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
            ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / max(1, steps)))
            with torch.autocast("mps", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=att, labels=lab)
            out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            if s % 20 == 0:
                msg = f"[sft] ep {ep} step {s}/{steps} loss {out.loss.item():.4f} elapsed {(time.time()-t0)/60:.1f}m"
                print(msg, flush=True); log.write(time.strftime("%F %T ") + msg + "\n"); log.flush()
            s += 1
    out_dir = VOL / "ckpt" / "sft" / "final"; model.save_pretrained(out_dir, safe_serialization=True); tok.save_pretrained(out_dir)
    print("[sft] DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "pretrain", "sft"])
    ap.add_argument("--layers", type=int, default=12); ap.add_argument("--d", type=int, default=512); ap.add_argument("--inter", type=int, default=1408)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--tokens", type=float, default=4e8); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--p-ko", type=float, default=0.5)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--data"); ap.add_argument("--init"); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()
    {"bench": cmd_bench, "pretrain": cmd_pretrain, "sft": cmd_sft}[a.cmd](a)
