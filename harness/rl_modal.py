"""Similarity-reward RL for Songgot on Modal (PyTorch, one H100). Same recipe as harness/rl_mlx.py:
GRPO-style on-policy updates against a continuous reward computed from the gold call (format term
plus argument similarity, after STAR, Ni et al. 2026), KL to the frozen SFT reference. Only our own
labels produce the reward.

    modal run harness/rl_modal.py --init sft/final --out rl/final --steps 200
"""
import json
import os
import random
import time

import modal

app = modal.App("songgot-rl")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
SPECIAL = ["<|system|>", "<|user|>", "<|call|>", "<|end|>", "<|pad|>"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "huggingface_hub[hf_transfer]", "numpy<2.3", "safetensors", "accelerate")
    .pip_install("llama-cpp-python", extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)


def render(example: dict):
    tools = json.dumps(example["tools"], ensure_ascii=False, separators=(",", ":"))
    return f"<|system|>\n{tools}\n<|user|>\n{example['query']}\n<|call|>\n"


# ---------------- reward (identical to rl_mlx.py) ----------------
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


def reward(text: str, gold: dict):
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


@app.function(image=image, gpu="H100", volumes={V: vol}, timeout=60 * 60 * 3, memory=65536)
def rl(init: str = "sft/final", out: str = "rl/final", data: str = "sft/train.jsonl", steps: int = 200, prompts: int = 16, group: int = 8,
       lr: float = 1e-6, kl: float = 0.02, temp: float = 0.8, max_new: int = 96, max_prompt: int = 900, seed: int = 0):
    import copy
    import numpy as np
    import torch
    from transformers import LlamaForCausalLM
    from llama_cpp import Llama
    vol.reload()
    vocab = f"{V}/tok/songgot-vocab.gguf"
    if not os.path.exists(vocab):
        from huggingface_hub import hf_hub_download
        vocab = hf_hub_download("palette-lab/songgot", "songgot-vocab.gguf")
    tok = Llama(model_path=vocab, vocab_only=True, verbose=False)
    enc = lambda s: tok.tokenize(s.encode("utf-8"), add_bos=False, special=True)
    dec = lambda ids: tok.detokenize([int(i) for i in ids], special=True).decode("utf-8", "replace")
    bos, eos = tok.token_bos(), tok.token_eos(); end_id = 3 + SPECIAL.index("<|end|>"); pad = 3 + SPECIAL.index("<|pad|>")
    dev = torch.device("cuda")
    policy = LlamaForCausalLM.from_pretrained(f"{V}/ckpt/{init}", torch_dtype=torch.bfloat16).to(dev)
    ref = copy.deepcopy(policy).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    rows = [json.loads(l) for l in open(f"{V}/{data}", encoding="utf-8")]
    rng = random.Random(seed); rng.shuffle(rows)
    rows = [r for r in rows[:60000] if len(enc(render(r))) <= max_prompt]
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    print(f"[rl] {len(rows)} prompts, {steps} steps, {prompts}x{group} per step", flush=True); t0 = time.time(); ptr = 0

    def seq_logprobs(model, ids, mask):
        logits = model(input_ids=ids, attention_mask=(ids != pad).long()).logits[:, :-1].float()
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp = lp.gather(-1, ids[:, 1:, None])[..., 0]
        return tok_lp * mask[:, 1:]

    for step in range(steps):
        batch = rows[ptr: ptr + prompts]; ptr = (ptr + prompts) % max(1, len(rows) - prompts)
        seqs, masks, advs, mean_r, exact_r = [], [], [], [], []
        policy.eval()
        with torch.no_grad():
            for r in batch:
                pids = [bos] + enc(render(r))
                inp = torch.tensor([pids], device=dev)
                gen = policy.generate(input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=temp, top_p=0.95, max_new_tokens=max_new,
                                      num_return_sequences=group, eos_token_id=[end_id, eos], pad_token_id=pad)
                comps = []
                for g in gen[:, len(pids):].tolist():
                    c = []
                    for t in g:
                        if t in (end_id, eos, pad):
                            break
                        c.append(t)
                    comps.append(c)
                rs = [reward(dec(c), r["call"]) for c in comps]
                vals = [x[0] for x in rs]; mean, std = float(np.mean(vals)), float(np.std(vals))
                mean_r.append(mean); exact_r.append(sum(x[1] for x in rs) / group)
                if std < 1e-6:
                    continue
                for c, (rw, _) in zip(comps, rs):
                    ids = pids + c + [eos]
                    seqs.append(ids); masks.append([0] * len(pids) + [1] * (len(c) + 1)); advs.append((rw - mean) / (std + 1e-4))
        if not seqs:
            print(f"[rl] step {step} no signal", flush=True); continue
        policy.train()
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long); msk = torch.zeros((len(seqs), L), dtype=torch.float32)
        for j, (s_, m_) in enumerate(zip(seqs, masks)):
            ids[j, :len(s_)] = torch.tensor(s_); msk[j, :len(m_)] = torch.tensor(m_)
        ids, msk = ids.to(dev), msk.to(dev); adv = torch.tensor(advs, device=dev, dtype=torch.float32)
        with torch.no_grad():
            ref_lp = seq_logprobs(ref, ids, msk)
        # micro-batches of 32 sequences to bound memory
        opt.zero_grad(set_to_none=True); total = 0.0
        for i in range(0, len(seqs), 32):
            sl = slice(i, i + 32)
            lp = seq_logprobs(policy, ids[sl], msk[sl])
            n = msk[sl, 1:].sum(dim=1).clamp(min=1)
            pg = -(adv[sl, None] * lp).sum(dim=1) / n
            d = ref_lp[sl] - lp
            klt = ((torch.exp(d) - d - 1) * msk[sl, 1:]).sum(dim=1) / n
            loss = (pg + kl * klt).mean() * (min(32, len(seqs) - i) / len(seqs))
            loss.backward(); total += loss.item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        if step and step % 100 == 0:  # checkpoint every 100 steps so a cancelled container loses little
            policy.save_pretrained(f"{V}/ckpt/{out}", safe_serialization=True); vol.commit()
        if step % 5 == 0:
            msg = f"[rl] step {step}/{steps} loss {total:.4f} mean_reward {np.mean(mean_r):.3f} exact {np.mean(exact_r):.3f} seqs {len(seqs)} elapsed {(time.time()-t0)/60:.1f}m"
            print(msg, flush=True); open(f"{V}/rl.log", "a").write(time.strftime("%F %T ") + msg + "\n")
    out_dir = f"{V}/ckpt/{out}"; policy.save_pretrained(out_dir, safe_serialization=True)
    import shutil
    for f in ("tokenizer.model", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json"):
        src = f"{V}/ckpt/{init}/{f}"
        if os.path.exists(src):
            shutil.copy(src, f"{out_dir}/{f}")
    vol.commit(); print("[rl] DONE", flush=True)
    return {"steps": steps}


@app.local_entrypoint()
def main(init: str = "sft/final", out: str = "rl/final", data: str = "sft/train.jsonl", steps: int = 200, prompts: int = 16, group: int = 8, lr: float = 1e-6, kl: float = 0.02, temp: float = 0.8):
    print(rl.remote(init, out, data, steps, prompts, group, lr, kl, temp))
