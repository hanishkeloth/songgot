"""FAST variant (app songgot-base2): no gradient checkpointing, optional row limit, one epoch by default.
Post-train an open sub-1B base (default Qwen3-0.6B, Apache 2.0) on the Songgot v5 tool-calling set, using
the base's own chat template with tools so its tool-calling prior carries over; then score it on
FunctionChat-Bench SingleCall on the same GPU. Lineage is declared on every card that ships these weights.

    modal run --detach harness/sft_base_modal.py --base Qwen/Qwen3-0.6B --out base_qwen06/final --name qwen06_sft
"""
import json
import os
import random
import re
import time

import modal

app = modal.App("songgot-base2")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V = "/vol"; CACHE = "/root/.cache/huggingface"
NONE_REPLY = "이 요청에 맞는 도구가 없어요."
CONDITIONS = ["exact", "4_random", "4_close", "8_random", "8_close"]

TV = os.environ.get("SONGGOT_TRANSFORMERS", "4.51.3")  # 4.51.3 for Qwen3, 5.17.0 for Qwen3.5 (set in the launching shell only)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", f"transformers=={TV}", "huggingface_hub[hf_transfer]", "numpy<2.3", "safetensors", "accelerate", "pillow")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)


def bench_items():
    out = []
    for line in open(f"{V}/bench/functionchat_singlecall.jsonl", encoding="utf-8"):
        r = json.loads(line)
        tools_by = {t["type"]: t["content"] for t in r["tools"]}
        for q in r["query"]:
            for cond in CONDITIONS:
                out.append({"id": f"{r['function_name']}#{q['serial_num']}#{cond}", "query": q["content"], "tools": tools_by[cond]})
    return out


def prompt_and_completion(tok, query: str, tools: list, call: dict):
    """Prompt = template with generation prompt (thinking off). Completion = the assistant turn the template itself
    renders for this tool call, minus the header and any empty think block, so training matches decoding exactly."""
    tl = [{"type": "function", "function": t} for t in tools]
    user = [{"role": "user", "content": query}]
    prompt = tok.apply_chat_template(user, tools=tl, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    asst = {"role": "assistant", "content": NONE_REPLY} if call["name"] == "none" else \
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": call["name"], "arguments": call.get("arguments", {})}}]}
    full = tok.apply_chat_template(user + [asst], tools=tl, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    turn = full[full.rfind("<|im_start|>assistant"):]
    turn = re.sub(r"^<\|im_start\|>assistant\n(<think>\s*</think>\s*)?", "", turn)
    return prompt, turn


def parse_call(out: str) -> str:
    """<tool_call>{json}</tool_call> (Qwen3) or <tool_call><function=name><parameter=k>v</parameter>...</function></tool_call> (Qwen3.5 style)."""
    m = re.search(r"<tool_call>(.*?)</tool_call>", out, re.S)
    if not m:
        return out.strip()
    body = m.group(1).strip()
    fm = re.search(r"<function=([^>\n]+)>(.*)", body, re.S)
    if not fm:
        return body
    args = {}
    for k, v in re.findall(r"<parameter=([^>\n]+)>(.*?)</parameter>", fm.group(2), re.S):
        v = v.strip()
        try:
            args[k] = json.loads(v)
        except Exception:
            args[k] = v
    return json.dumps({"name": fm.group(1).strip(), "arguments": args}, ensure_ascii=False)


@app.function(image=image, gpu="H100", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 24, memory=65536)
def sft(base: str = "Qwen/Qwen3-0.6B", out: str = "base_qwen06/final", data: str = "sft/train.jsonl", epochs: int = 1, lr: float = 1e-5,
        batch: int = 16, accum: int = 2, max_len: int = 1024, seed: int = 0, limit: int = 0):
    import math
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    vol.reload()
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, **({"dtype": torch.float32} if TV.startswith("5") else {"torch_dtype": torch.float32})).cuda()
    model.gradient_checkpointing_enable(); model.config.use_cache = False  # the 248k-token vocab makes fp32 logits ~1 GB per sequence; without checkpointing batch 8 OOMs on 80 GB
    rows = [json.loads(l) for l in open(f"{V}/{data}", encoding="utf-8")]
    rng = random.Random(seed); rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    log = open(f"{V}/base.log", "a")
    def say(m):
        print(m, flush=True); log.write(time.strftime("%F %T ") + m + "\n"); log.flush()

    def encode(r):
        prompt, comp = prompt_and_completion(tok, r["query"], r["tools"], r["call"])
        p = tok(prompt, add_special_tokens=False)["input_ids"]; c = tok(comp, add_special_tokens=False)["input_ids"]
        return (p + c)[:max_len], ([-100] * len(p) + c)[:max_len]
    enc = [encode(r) for r in rows]
    say(f"[base] {base}: {len(enc)} examples, mean len {sum(len(x[0]) for x in enc)/len(enc):.0f}; sample completion: {prompt_and_completion(tok, rows[0]['query'], rows[0]['tools'], rows[0]['call'])[1][:160]!r}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps = epochs * (len(enc) // (batch * accum)); warm = max(1, steps // 30); s = 0; t0 = time.time()
    lr_at = lambda k: lr * (k + 1) / warm if k < warm else lr * 0.1 + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * (k - warm) / max(1, steps - warm)))
    model.train()
    for ep in range(epochs):
        rng.shuffle(enc)
        for i in range(0, len(enc) - batch * accum + 1, batch * accum):
            for g in opt.param_groups:
                g["lr"] = lr_at(s)
            tot = 0.0
            for a in range(accum):
                chunk = enc[i + a * batch: i + (a + 1) * batch]; L = max(len(x[0]) for x in chunk)
                ids = torch.full((len(chunk), L), pad, dtype=torch.long); lab = torch.full((len(chunk), L), -100, dtype=torch.long); att = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, (x, y) in enumerate(chunk):
                    ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss / accum
                loss.backward(); tot += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); s += 1
            if s % 50 == 0 or s == steps:
                say(f"[base] ep {ep} step {s}/{steps} loss {tot:.4f} lr {lr_at(s):.2e} elapsed {(time.time()-t0)/60:.1f}m")
    out_dir = f"{V}/ckpt/{out}"; model.save_pretrained(out_dir, safe_serialization=True); tok.save_pretrained(out_dir)
    vol.commit(); say("[base] DONE")
    return {"steps": steps}


@app.function(image=image, gpu="H100", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60, memory=32768)
def predict(ckpt: str = "base_qwen06/final", name: str = "qwen06_sft", max_new: int = 200):
    """Greedy decoding with the base's chat template (thinking off), <tool_call> parsed like eval/functionchat_exact.py."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    vol.reload()
    path = f"{V}/ckpt/{ckpt}" if not ckpt.count("/") == 1 or os.path.exists(f"{V}/ckpt/{ckpt}") else ckpt
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, **({"dtype": torch.bfloat16} if TV.startswith("5") else {"torch_dtype": torch.bfloat16})).cuda().eval()
    os.makedirs(f"{V}/preds", exist_ok=True); rows = bench_items(); t0 = time.time()
    with open(f"{V}/preds/{name}.jsonl", "w", encoding="utf-8") as f:
        for i, it in enumerate(rows):
            tools = [{"type": "function", "function": t["function"]} for t in it["tools"]]
            text = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
            out = parse_call(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
            f.write(json.dumps({"id": it["id"], "output": out}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"[predict] {i}/{len(rows)} {out[:90]}", flush=True)
    vol.commit(); print(f"[predict] DONE {len(rows)} in {time.time()-t0:.0f}s", flush=True)
    return {"n": len(rows)}


@app.local_entrypoint()
def main(base: str = "Qwen/Qwen3-0.6B", out: str = "base_qwen06/final", name: str = "qwen06_sft", data: str = "sft/train.jsonl", epochs: int = 2, lr: float = 1e-5, skip_sft: bool = False):
    if not skip_sft:
        print(sft.remote(base, out, data, epochs, lr))
    print(predict.remote(out, name))
