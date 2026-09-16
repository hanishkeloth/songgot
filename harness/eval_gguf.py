"""FunctionChat-Bench SingleCall on the published GGUF through llama.cpp with a chosen KV-cache type, so the number a
Pocket or Desktop user actually gets (Q4_K_M weights, f16 / q8_0 / q4_0 KV) is measured rather than inferred from the
bf16 transformers predictor. Same prompt rendering (tokenizer chat template, tools, thinking off), same parse_call, same
{id, output} rows, scored by eval/functionchat_exact.py. CPU only: a 0.8B Q4 model prefers 32 cores to a GPU build here,
and the CPU path is the one the browser and the laptop run.

    modal deploy harness/eval_gguf.py
    .venv/bin/python harness/spawn.py songgot-eval-gguf predict_gguf gguf=export_x_soup4/songgot-x-q4_k_m.gguf kv=q8_0 name=soup4_q4km_kv_q8_0
    modal volume get songgot preds/soup4_q4km_kv_q8_0.jsonl eval/preds_soup4_q4km_kv_q8_0.jsonl
    .venv/bin/python eval/functionchat_exact.py eval/preds_soup4_q4km_kv_q8_0.jsonl

llama.cpp quantizes only the attention layers' K/V; the Gated DeltaNet recurrent state of the 18 linear-attention layers
stays f32 whatever cache type is chosen (llama-memory-hybrid.cpp filters on !is_recurrent), so the exposed surface is
6 of 24 layers. A quantized V cache needs flash attention on; it is switched on for every variant so f16 is a fair base.
"""
import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V = "/vol"; CACHE = "/root/.cache/huggingface"
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "git")
    .env({"CMAKE_ARGS": "-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON", "FORCE_CMAKE": "1"})
    .pip_install("llama-cpp-python==0.3.35", "transformers==5.17.0", "jinja2", "huggingface_hub", "numpy")
    .add_local_python_source("sft_base_fast")
)
app = modal.App("songgot-eval-gguf")
KV = {"f16": "GGML_TYPE_F16", "q8_0": "GGML_TYPE_Q8_0", "q5_1": "GGML_TYPE_Q5_1", "q4_0": "GGML_TYPE_Q4_0"}


@app.function(image=image, cpu=32, memory=32768, volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 4)
def predict_gguf(gguf: str = "export_x_soup4/songgot-x-q4_k_m.gguf", tok_id: str = "palette-lab/songgot-x-0.8b", kv: str = "f16",
                 name: str = "soup4_q4km_kv_f16", n_ctx: int = 4096, max_new: int = 200, threads: int = 32):
    import llama_cpp
    from llama_cpp import Llama
    from transformers import AutoTokenizer
    from sft_base_fast import bench_items, parse_call
    vol.reload()
    tok = AutoTokenizer.from_pretrained(tok_id)
    t = getattr(llama_cpp, KV[kv])
    llm = Llama(model_path=f"{V}/{gguf}", n_ctx=n_ctx, n_batch=512, n_threads=threads, n_threads_batch=threads,
                type_k=t, type_v=t, flash_attn=True, verbose=False)
    rows = bench_items(); os.makedirs(f"{V}/preds", exist_ok=True); t0 = time.time(); ptoks = 0; gtoks = 0
    with open(f"{V}/preds/{name}.jsonl", "w", encoding="utf-8") as f:
        for i, it in enumerate(rows):
            tools = [{"type": "function", "function": t_["function"]} for t_ in it["tools"]]
            text = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            r = llm.create_completion(text, max_tokens=max_new, temperature=0.0, stop=["<|im_end|>"])
            u = r.get("usage") or {}; ptoks += u.get("prompt_tokens", 0); gtoks += u.get("completion_tokens", 0)
            out = parse_call(r["choices"][0]["text"])
            f.write(json.dumps({"id": it["id"], "output": out}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"[gguf {kv}] {i}/{len(rows)} {time.time()-t0:.0f}s {out[:90]}", flush=True)
    vol.commit()
    secs = time.time() - t0
    print(f"[gguf {kv}] DONE {len(rows)} in {secs:.0f}s prompt_tokens {ptoks} gen_tokens {gtoks} -> preds/{name}.jsonl", flush=True)
    return {"n": len(rows), "kv": kv, "secs": round(secs), "prompt_tokens": ptoks, "gen_tokens": gtoks}
