"""Held-out human test: predict tool calls for a rows file in the benchmark item shape ({id, query, tools:[{function}]})
with a fine-tuned checkpoint, in its own Modal app so a training run on songgot-base2 is never redeployed under.
Built for bench/massive_ko_test_500.jsonl: 500 MASSIVE 1.1 ko-KR test utterances (never trained on; dev/test were held
out of every set), each with the gold function plus 3 random massive-agents declarations. Score locally with
eval/score_massive.py against bench/massive_ko_test_500_gold.jsonl.

    SONGGOT_TRANSFORMERS=5.17.0 modal deploy harness/eval_massive.py
    .venv/bin/python harness/spawn.py songgot-eval-massive predict_rows ckpt=base_q35_08b_v10/final rows=bench/massive_ko_test_500.jsonl name=q35_08b_v10_massive500
"""
import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_base_fast import CACHE, TV, V, hf_cache, image as base_image, parse_call, vol  # noqa: E402

# flash-linear-attention gives Qwen3.5 its fused gated-delta kernels: the reference PyTorch path decodes ~12 items/min on an
# H100, the fused one several times faster. Numerics differ at bf16 rounding level; the benchmark table stays on the base2
# predictor (reference kernels) so published numbers keep one code path.
image = base_image.pip_install("flash-linear-attention").add_local_python_source("sft_base_fast")
# Kanana-2 carries a pre-5.x rope_scaling block that transformers 5.17 refuses ("'int' object has no attribute 'get'"),
# so a second deployment of this file with SONGGOT_TRANSFORMERS=4.51.3 SONGGOT_EVAL_APP=songgot-eval-massive-tf4 serves it.
app = modal.App(os.environ.get("SONGGOT_EVAL_APP", "songgot-eval-massive"))


@app.function(image=image, gpu="H100", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 6, memory=65536)
def predict_rows(ckpt: str = "base_q35_08b_v10/final", rows: str = "bench/massive_ko_test_500.jsonl", name: str = "q35_08b_v10_massive500", max_new: int = 200):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    vol.reload()
    path = f"{V}/ckpt/{ckpt}" if os.path.exists(f"{V}/ckpt/{ckpt}") else ckpt
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, **({"dtype": torch.bfloat16} if TV.startswith("5") else {"torch_dtype": torch.bfloat16})).cuda().eval()
    items = [json.loads(l) for l in open(f"{V}/{rows}", encoding="utf-8")]
    os.makedirs(f"{V}/preds", exist_ok=True); t0 = time.time()
    with open(f"{V}/preds/{name}.jsonl", "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            tools = [{"type": "function", "function": t["function"]} for t in it["tools"]]
            text = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
            out = parse_call(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
            f.write(json.dumps({"id": it["id"], "output": out}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"[predict-rows] {i}/{len(items)} {out[:90]}", flush=True)
    vol.commit(); print(f"[predict-rows] DONE {len(items)} in {time.time()-t0:.0f}s -> preds/{name}.jsonl", flush=True)
    return {"n": len(items)}


@app.function(image=image, gpu="H100", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 6, memory=65536)
def predict_rows_batched(ckpt: str = "soup_q35_08b_v10v11v12/final", rows: str = "bench/rl_pool_v12_sample6k.jsonl", name: str = "soup3_pool6k_fast", max_new: int = 160, batch: int = 32):
    """Same as predict_rows but generates `batch` prompts per call with left padding, sorted by prompt length so padding is
    small. One prompt at a time runs ~12-15 items/min on an H100 whichever kernels are installed; the autoregressive loop,
    not the matmuls, is the cost. Greedy decoding, so results equal the unbatched path up to bf16 batching noise."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    vol.reload()
    path = f"{V}/ckpt/{ckpt}" if os.path.exists(f"{V}/ckpt/{ckpt}") else ckpt
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, **({"dtype": torch.bfloat16} if TV.startswith("5") else {"torch_dtype": torch.bfloat16})).cuda().eval()
    items = [json.loads(l) for l in open(f"{V}/{rows}", encoding="utf-8")]
    texts = []
    for it in items:
        tools = [{"type": "function", "function": t["function"]} for t in it["tools"]]
        texts.append(tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    order = sorted(range(len(items)), key=lambda i: len(texts[i]))
    outs = [None] * len(items); t0 = time.time(); done = 0
    os.makedirs(f"{V}/preds", exist_ok=True)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        L = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            outs[i] = parse_call(tok.decode(g[j][L:], skip_special_tokens=True))
        done += len(idx)
        if (s // batch) % 10 == 0:
            print(f"[predict-batched] {done}/{len(items)} {time.time()-t0:.0f}s {outs[idx[0]][:80]}", flush=True)
    with open(f"{V}/preds/{name}.jsonl", "w", encoding="utf-8") as f:
        for it, o in zip(items, outs):
            f.write(json.dumps({"id": it["id"], "output": o}, ensure_ascii=False) + "\n")
    vol.commit(); print(f"[predict-batched] DONE {len(items)} in {time.time()-t0:.0f}s -> preds/{name}.jsonl", flush=True)
    return {"n": len(items)}
