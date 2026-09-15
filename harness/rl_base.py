"""GRPO-style similarity-reward RL for line B (Qwen3.5 post-trained checkpoints) on Modal, one H100. Port of
harness/rl_modal.py to the base model's own chat template and tool-call format: prompts are rendered with
tokenizer.apply_chat_template(tools=..., enable_thinking=False), completions parsed like sft_base_fast.parse_call,
reward = STAR-style argument similarity from our own gold labels (format -1, wrong tool 0, else mean per-key
similarity; extra or missing keys count against), KL to the frozen SFT reference. The prompt set must be disjoint
from the SFT set (harness/make_rl_pool.py); on memorised prompts the group reward has no variance and nothing is learnt.

    SONGGOT_TRANSFORMERS=5.17.0 modal deploy harness/rl_base.py
    .venv/bin/python harness/spawn.py songgot-rl-base rl init=base_q35_08b_v11/final out=rl_q35_08b_v11/final data=rl/pool_v11.jsonl steps=300
"""
import copy
import json
import os
import random
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_base_fast import CACHE, NONE_REPLY, TV, V, hf_cache, image as base_image, parse_call, vol  # noqa: E402
from rl_modal import key_sim  # noqa: E402

image = base_image.pip_install("flash-linear-attention").add_local_python_source("sft_base_fast", "rl_modal")
app = modal.App("songgot-rl-base")


def reward(text: str, gold: dict):
    """(reward, exact). Parsed like the scorer; a none-gold row rewards the plain refusal and punishes a call."""
    if gold["name"] == "none":
        return (1.0, True) if "<tool_call>" not in text and NONE_REPLY[:8] in text else (0.0, False)
    if "<tool_call>" not in text:
        return -1.0, False
    try:
        p = json.loads(parse_call(text))
    except Exception:
        return -1.0, False
    if not isinstance(p, dict) or p.get("name") != gold["name"]:
        return 0.0, False
    pa, ga = p.get("arguments") or {}, gold.get("arguments") or {}
    if isinstance(ga, str):
        try:
            ga = json.loads(ga) if ga.strip() else {}
        except Exception:
            ga = {}
    if not pa and not ga:
        return 1.0, True
    keys = set(pa) | set(ga)
    s = sum(key_sim(pa[k], ga[k]) for k in keys if k in pa and k in ga) / len(keys)
    return s, s >= 0.999


@app.function(image=image, gpu="H100", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 12, memory=65536)
def rl(init: str = "base_q35_08b_v11/final", out: str = "rl_q35_08b_v11/final", data: str = "rl/pool_v11.jsonl", steps: int = 300, prompts: int = 16, group: int = 8,
       lr: float = 1e-6, kl: float = 0.02, temp: float = 0.8, max_new: int = 160, max_prompt: int = 1400, seed: int = 0, save_every: int = 50):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    vol.reload()
    path = f"{V}/ckpt/{init}"
    tok = AutoTokenizer.from_pretrained(path)
    dt = {"dtype": torch.bfloat16} if TV.startswith("5") else {"torch_dtype": torch.bfloat16}
    policy = AutoModelForCausalLM.from_pretrained(path, **dt).cuda()
    policy.gradient_checkpointing_enable(); policy.config.use_cache = False
    ref = copy.deepcopy(policy).eval(); ref.gradient_checkpointing_disable()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>"); eos = tok.eos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    rows = [json.loads(l) for l in open(f"{V}/{data}", encoding="utf-8")]
    rng = random.Random(seed); rng.shuffle(rows)

    def render(r):
        tl = [{"type": "function", "function": t} for t in r["tools"]]
        return tok.apply_chat_template([{"role": "user", "content": r["query"]}], tools=tl, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    rows = [r for r in rows if len(tok(render(r))["input_ids"]) <= max_prompt]
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    log = open(f"{V}/rl_base.log", "a")
    say = lambda m: (print(m, flush=True), log.write(time.strftime("%F %T ") + f"[{out}] " + m + "\n"), log.flush())
    say(f"[rl] init {init}: {len(rows)} prompts, {steps} steps, {prompts}x{group} per step, lr {lr}, kl {kl}, temp {temp}")
    t0 = time.time(); ptr = 0; dev = torch.device("cuda")

    def seq_logprobs(model, ids, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=(ids != pad).long()).logits[:, :-1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return lp.gather(-1, ids[:, 1:, None])[..., 0] * mask[:, 1:]

    for step in range(steps):
        batch = rows[ptr: ptr + prompts]; ptr = (ptr + prompts) % max(1, len(rows) - prompts)
        seqs, masks, advs, mean_r, exact_r = [], [], [], [], []
        policy.eval(); policy.config.use_cache = True
        with torch.no_grad():
            for r in batch:
                pids = tok(render(r))["input_ids"]
                inp = torch.tensor([pids], device=dev)
                gen = policy.generate(input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=temp, top_p=0.95, max_new_tokens=max_new,
                                      num_return_sequences=group, eos_token_id=[im_end, eos], pad_token_id=pad)
                comps = []
                for g in gen[:, len(pids):].tolist():
                    c = []
                    for t in g:
                        if t in (im_end, eos, pad):
                            break
                        c.append(t)
                    comps.append(c)
                rs = [reward(tok.decode(c, skip_special_tokens=False), r["call"]) for c in comps]
                vals = [x[0] for x in rs]; mean, std = float(np.mean(vals)), float(np.std(vals))
                mean_r.append(mean); exact_r.append(sum(x[1] for x in rs) / group)
                if std < 1e-6:
                    continue
                for c, (rw, _) in zip(comps, rs):
                    ids = pids + c + [im_end]
                    seqs.append(ids); masks.append([0] * len(pids) + [1] * (len(c) + 1)); advs.append((rw - mean) / (std + 1e-4))
        policy.config.use_cache = False
        if not seqs:
            say(f"[rl] step {step} no signal (mean_reward {np.mean(mean_r):.3f})"); continue
        policy.train()
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long); msk = torch.zeros((len(seqs), L), dtype=torch.float32)
        for j, (s_, m_) in enumerate(zip(seqs, masks)):
            ids[j, :len(s_)] = torch.tensor(s_); msk[j, :len(m_)] = torch.tensor(m_)
        ids, msk = ids.to(dev), msk.to(dev); adv = torch.tensor(advs, device=dev, dtype=torch.float32)
        with torch.no_grad():
            ref_lp = torch.cat([seq_logprobs(ref, ids[i:i + 8], msk[i:i + 8]) for i in range(0, len(seqs), 8)])
        opt.zero_grad(set_to_none=True); total = 0.0
        for i in range(0, len(seqs), 8):  # the 248k vocab makes fp32 logits ~1 GB per sequence: micro-batches of 8
            sl = slice(i, i + 8)
            lp = seq_logprobs(policy, ids[sl], msk[sl])
            n = msk[sl, 1:].sum(dim=1).clamp(min=1)
            pg = -(adv[sl, None] * lp).sum(dim=1) / n
            d = ref_lp[sl] - lp
            klt = ((torch.exp(d) - d - 1) * msk[sl, 1:]).sum(dim=1) / n
            loss = (pg + kl * klt).mean() * (min(8, len(seqs) - i) / len(seqs))
            loss.backward(); total += loss.item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        if step and step % save_every == 0:
            policy.save_pretrained(f"{V}/ckpt/{out}", safe_serialization=True); tok.save_pretrained(f"{V}/ckpt/{out}"); vol.commit()
        if step % 5 == 0:
            say(f"[rl] step {step}/{steps} loss {total:.4f} mean_reward {np.mean(mean_r):.3f} exact {np.mean(exact_r):.3f} seqs {len(seqs)} elapsed {(time.time()-t0)/60:.1f}m")
    policy.save_pretrained(f"{V}/ckpt/{out}", safe_serialization=True); tok.save_pretrained(f"{V}/ckpt/{out}")
    for f in ("chat_template.jinja", "generation_config.json"):
        src = f"{path}/{f}"
        if os.path.exists(src) and not os.path.exists(f"{V}/ckpt/{out}/{f}"):
            import shutil; shutil.copy(src, f"{V}/ckpt/{out}/{f}")
    vol.commit(); say(f"[rl] DONE {steps} steps -> ckpt/{out}")
    return {"steps": steps, "out": out}
