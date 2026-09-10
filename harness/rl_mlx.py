"""Similarity-reward RL for Songgot (GRPO-style, on-policy, MLX).

After SFT the model is refined with group-relative policy optimisation against a continuous reward
computed from the gold call, following the reward design of STAR (Ni et al., 2026, arXiv:2602.03022):

    R = (R_format - 1) + R_format * sim(pred, gold)

R_format is 1 when the output parses as a JSON object with a "name" and an "arguments" object,
else 0 (so a malformed output scores -1). sim is 0 when the tool name differs; otherwise the mean over
the union of argument keys of a per-key score: exact match for numbers and booleans, character-level
LCS F1 (ROUGE-L on characters, which suits Korean) for strings, 1 for matching "none" calls.

For each prompt G completions are sampled at temperature T; advantages are group-normalised; the loss
is -A * mean token log-prob of the completion plus a KL penalty to the frozen SFT reference. One
optimiser step per batch of prompts (on-policy, no importance ratio needed). No teacher, no closed
model: the reward comes only from our own gold labels.

    SONGGOT_VOL=$PWD/vol .venv/bin/python harness/rl_mlx.py --init vol/ckpt/sft_e2 --data data/sft_train_v2.jsonl \
        --steps 400 --prompts 8 --group 8 --lr 1e-6 --out vol/ckpt/rl_final
"""
from __future__ import annotations

import argparse, json, math, os, pathlib, random, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import importlib
from mlx.utils import tree_flatten

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_mlx as T  # noqa: E402
import songgot_modal as sm  # noqa: E402

G = importlib.import_module("mlx_lm.generate")
S = importlib.import_module("mlx_lm.sample_utils")


# ---------------- reward ----------------
def lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0]
        for j, bj in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ch == bj else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def str_sim(p: str, g: str) -> float:
    if p == g:
        return 1.0
    l = lcs_len(p, g)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(g)
    return 2 * prec * rec / (prec + rec)


def key_sim(pv, gv) -> float:
    if isinstance(gv, bool) or isinstance(gv, (int, float)):
        try:
            return 1.0 if (isinstance(pv, bool) == isinstance(gv, bool)) and float(pv) == float(gv) else 0.0
        except (TypeError, ValueError):
            return 0.0
    if isinstance(gv, str):
        return str_sim(str(pv), gv) if isinstance(pv, (str, int, float)) else 0.0
    return 1.0 if json.dumps(pv, sort_keys=True, ensure_ascii=False) == json.dumps(gv, sort_keys=True, ensure_ascii=False) else 0.0


def reward(text: str, gold: dict) -> tuple[float, bool]:
    """Returns (reward, exact). Format failures score -1."""
    t = text.split("<|end|>")[0].strip()
    try:
        p = json.loads(t)
    except json.JSONDecodeError:
        return -1.0, False
    if not isinstance(p, dict) or "name" not in p or not isinstance(p.get("arguments", {}), dict):
        return -1.0, False
    if p["name"] != gold["name"]:
        return 0.0, False
    pa, ga = p.get("arguments", {}) or {}, gold.get("arguments", {}) or {}
    if not pa and not ga:
        return 1.0, True
    keys = set(pa) | set(ga)
    s = sum(key_sim(pa[k], ga[k]) for k in keys if k in pa and k in ga) / len(keys)
    return s, s >= 0.999


# ---------------- sampling and log-probs ----------------
def sample_completion(model, prompt_ids, sp, end_ids, sampler, max_new):
    toks = []
    for tok, _ in G.generate_step(mx.array(prompt_ids), model, max_tokens=max_new, sampler=sampler):
        t = int(tok)
        if t in end_ids:
            break
        toks.append(t)
    return toks


def seq_logprobs(model, ids: mx.array, mask: mx.array):
    """Per-token log-prob of ids[:,1:] under model, masked to completion positions."""
    logits = model(ids[:, :-1]).astype(mx.float32)
    lp = nn.log_softmax(logits, axis=-1)
    tgt = ids[:, 1:]
    tok_lp = mx.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0]
    return tok_lp * mask[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--prompts", type=int, default=8); ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--kl", type=float, default=0.02); ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=96); ap.add_argument("--max-prompt", type=int, default=900); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    init = pathlib.Path(a.init); cfg = json.load(open(init / "config.json"))
    margs = T.args_for(cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"])
    model = T.load_hf(init, margs); ref = T.load_hf(init, margs); ref.freeze()
    model.set_dtype(mx.bfloat16); ref.set_dtype(mx.bfloat16)
    from songgot_tok import Tok
    tok = Tok()

    class _SP:  # llama.cpp tokenizer behind the SentencePiece-shaped calls used below
        def encode(self, s): return tok.encode(s)
        def decode(self, ids): return tok.decode(ids, keep_special=True)
        def bos_id(self): return tok.bos_id
        def eos_id(self): return tok.eos_id
    sp = _SP()
    end_ids = {tok.eos_id, tok.end_id, tok.pad_id}
    pad = tok.pad_id
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    rng = random.Random(a.seed); rng.shuffle(rows)
    if a.limit:
        rows = rows[: a.limit]
    rows = [r for r in rows if len(sp.encode(sm.render(r)[0])) <= a.max_prompt]
    sampler = S.make_sampler(temp=a.temp, top_p=0.95)
    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.99], weight_decay=0.0)

    def loss_fn(model, ids, mask, adv, ref_lp):
        lp = seq_logprobs(model, ids, mask)  # (B, L-1)
        n = mx.maximum(mask[:, 1:].sum(axis=1), 1)
        pg = -(adv[:, None] * lp).sum(axis=1) / n
        # k3 KL estimator, per token, masked
        d = ref_lp - lp
        kl = ((mx.exp(d) - d - 1) * mask[:, 1:]).sum(axis=1) / n
        return (pg + a.kl * kl).mean()

    vg = nn.value_and_grad(model, loss_fn)
    log = open(T.VOL / "rl_mlx.log", "a"); t0 = time.time(); ptr = 0
    print(f"[rl] {len(rows)} prompts, {a.steps} steps, group {a.group}", flush=True)
    for step in range(a.steps):
        batch = rows[ptr: ptr + a.prompts]; ptr = (ptr + a.prompts) % max(1, len(rows) - a.prompts)
        seqs, masks, advs, stats = [], [], [], []
        for r in batch:
            p, _ = sm.render(r); pids = [sp.bos_id()] + sp.encode(p)
            comps = [sample_completion(model, pids, sp, end_ids, sampler, a.max_new) for _ in range(a.group)]
            rs = [reward(sp.decode(c), r["call"])[0] for c in comps]
            mean, std = float(np.mean(rs)), float(np.std(rs))
            stats.append((mean, sum(1 for c in comps if reward(sp.decode(c), r["call"])[1]) / a.group))
            if std < 1e-6:
                continue  # no learning signal in this group
            for c, rw in zip(comps, rs):
                ids = pids + c + [sp.eos_id()]
                seqs.append(ids); masks.append([0] * len(pids) + [1] * (len(c) + 1)); advs.append((rw - mean) / (std + 1e-4))
        if not seqs:
            print(f"[rl] step {step} no signal", flush=True); continue
        Lm = max(len(s) for s in seqs)
        ids = np.full((len(seqs), Lm), pad, dtype=np.int32); msk = np.zeros((len(seqs), Lm), dtype=np.float32)
        for j, (s_, m_) in enumerate(zip(seqs, masks)):
            ids[j, :len(s_)] = s_; msk[j, :len(m_)] = m_
        X, M, A = mx.array(ids), mx.array(msk), mx.array(np.array(advs, dtype=np.float32))
        ref_lp = seq_logprobs(ref, X, M); mx.eval(ref_lp)
        loss, grads = vg(model, X, M, A, ref_lp); grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads); mx.eval(model.parameters(), opt.state, loss)
        if step % 5 == 0:
            mr = float(np.mean([s[0] for s in stats])); ex = float(np.mean([s[1] for s in stats]))
            msg = f"[rl] step {step}/{a.steps} loss {loss.item():.4f} mean_reward {mr:.3f} exact {ex:.3f} seqs {len(seqs)} elapsed {(time.time()-t0)/60:.1f}m"
            print(msg, flush=True); log.write(time.strftime("%F %T ") + msg + "\n"); log.flush()
    T.save_hf(model, margs, pathlib.Path(a.out)); print("[rl] DONE", flush=True)


if __name__ == "__main__":
    main()
