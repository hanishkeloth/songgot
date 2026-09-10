"""Korean queries over thousands of tools: our own teacher (palette-lab/palette-k-midm) rewrites English
glaive requests into natural Korean while the tool schemas and the gold call stay exactly as they are.

    modal run harness/teacher_translate.py --limit 12000

Reads /vol/data/glaive_en_calls.jsonl, writes /vol/sft/ko_trans.jsonl rows
{"lang": "ko", "query": <korean>, "tools": [...], "call": {...}, "cond": "en_ko"}.
A row is kept only if the Korean text contains Hangul and still carries every ASCII argument value
verbatim (names, codes, numbers), so the call remains the correct label for the sentence.
No closed model produces labels; the labels are glaive's (Apache 2.0), the wording is ours.
"""
import json
import os
import random
import re
import time

import modal

app = modal.App("songgot-teacher-translate")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V = "/vol"
CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.27.1", "huggingface_hub[hf_transfer]", "transformers")
    .run_commands("pip uninstall -y flashinfer flashinfer-python || true")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
)

STYLES = ["짧은 반말", "정중한 존댓말", "회사에서 쓰는 격식체", "친구에게 말하듯 편한 말투", "오타나 띄어쓰기 실수가 섞인 말투",
          "아주 짧게 단어만 나열하는 말투", "급하게 부탁하는 말투", "길게 상황을 설명한 뒤 요청하는 말투"]


def prompt_for(query: str, call: dict, style: str) -> str:
    vals = [str(v) for v in call.get("arguments", {}).values() if isinstance(v, (str, int, float))]
    keep = ", ".join(f'"{v}"' for v in vals[:6]) or "없음"
    return (
        "다음 영어 요청을 한국어 음성 비서에게 말하듯 자연스러운 한국어 한 문장으로 바꾸세요.\n"
        f"말투: {style}.\n"
        f"반드시 그대로 유지할 값(영문 이름, 숫자, 코드 등): {keep}. 이 값들은 번역하거나 바꾸지 말고 문장 안에 그대로 넣으세요.\n"
        "설명이나 따옴표 없이 한국어 문장만 한 줄로 출력하세요.\n\n"
        f"영어 요청: {query}"
    )


@app.function(image=image, gpu="H100:2", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 2, memory=65536)
def translate(limit: int = 12000, model_id: str = "palette-lab/palette-k-midm"):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open(f"{V}/data/glaive_en_calls.jsonl", encoding="utf-8")][:limit]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True, max_model_len=2048,
              gpu_memory_utilization=0.9, dtype="bfloat16", enforce_eager=True)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=160, stop=["<|im_end|>", "<|endoftext|>", "\n"])
    rng = random.Random(11)
    texts = [tok.apply_chat_template([{"role": "user", "content": prompt_for(r["query"], r["call"], rng.choice(STYLES))}], tokenize=False, add_generation_prompt=True) for r in rows]
    print(f"[translate] {len(texts)} prompts", flush=True); t0 = time.time()
    outs = llm.generate(texts, sp)
    os.makedirs(f"{V}/sft", exist_ok=True)
    n_ok = n_bad = 0
    hangul = re.compile(r"[가-힣]")
    with open(f"{V}/sft/ko_trans.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            ko = o.outputs[0].text.strip().strip("`\"'“”")
            vals = [str(v) for v in r["call"].get("arguments", {}).values() if isinstance(v, (str, int, float))]
            ascii_vals = [v for v in vals if v.isascii() and len(v) > 1]
            if not (3 <= len(ko) <= 200) or not hangul.search(ko) or any(v not in ko for v in ascii_vals):
                n_bad += 1; continue
            f.write(json.dumps({"lang": "ko", "query": ko, "tools": r["tools"], "call": r["call"], "cond": "en_ko"}, ensure_ascii=False) + "\n"); n_ok += 1
    vol.commit()
    msg = f"[translate] ok {n_ok} bad {n_bad} in {time.time()-t0:.0f}s"
    print(msg, flush=True); open(f"{V}/teacher.log", "a").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    return {"ok": n_ok, "bad": n_bad}


@app.local_entrypoint()
def main(limit: int = 12000):
    print(translate.remote(limit))
