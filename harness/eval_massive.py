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

image = base_image.add_local_python_source("sft_base_fast")
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
