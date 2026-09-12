"""Korean document QA for Songgot-V stage 2, written by our own teacher (Palette-K-Midm) from the page MARKDOWN of
ko-vdr pages (CC BY 4.0). The teacher never sees the image; the questions are answerable from the page text, so the
(image, question, answer) triple is a valid supervision signal for a model that must read the page.

Two question types per page: a short factual lookup (number, name, date, title) and one about structure (a table
cell, a list item, a heading). Answers are short. Rows: {"image", "question", "answer", "kind", "src"}.

    modal deploy harness/teacher_docqa.py
    .venv/bin/python harness/spawn.py songgot-teacher-docqa docqa pages=60000
"""
import json
import os
import random
import re
import time

import modal

app = modal.App("songgot-teacher-docqa")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V, CACHE = "/vol", "/root/.cache/huggingface"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.27.1", "huggingface_hub[hf_transfer]", "transformers")
    .run_commands("pip uninstall -y flashinfer flashinfer-python || true")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
)


def prompt_for(md: str) -> str:
    return ("아래는 한국어 문서 한 페이지의 내용입니다. 이 페이지의 이미지만 보고 답할 수 있는 질문 2개를 만드세요.\n"
            "1) 사실 확인형: 숫자, 날짜, 이름, 제목 등 페이지에 적힌 값을 묻는 질문. 답은 페이지에 적힌 표현 그대로 짧게.\n"
            "2) 구조형: 표의 특정 칸, 목록의 항목, 소제목 등 페이지 구조를 읽어야 답할 수 있는 질문. 답은 짧게.\n"
            "페이지에 없는 내용은 묻지 마세요. 설명 없이 JSON 배열만 출력: [{\"kind\":\"fact\",\"question\":\"...\",\"answer\":\"...\"},{\"kind\":\"structure\",\"question\":\"...\",\"answer\":\"...\"}]\n\n"
            f"페이지 내용:\n{md[:3500]}")


def first_json_list(text: str):
    text = re.sub(r"```(?:json)?", "", text); i = text.find("[")
    if i < 0:
        return None
    depth = 0; in_s = False; esc = False
    for j in range(i, len(text)):
        c = text[j]
        if in_s:
            esc = (c == "\\") if not esc else False
            if c == '"' and not esc:
                in_s = False
            continue
        if c == '"':
            in_s = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    return None
    return None


@app.function(image=image, gpu="H100:2", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 4, memory=65536)
def docqa(pages: int = 60000, model_id: str = "palette-lab/palette-k-midm", src: str = "corpora/ko_vdr_pages.jsonl", dst: str = "corpora/ko_vdr_docqa.jsonl", seed: int = 12):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    vol.reload(); t0 = time.time()
    rows = [json.loads(l) for l in open(f"{V}/{src}", encoding="utf-8")]
    rows = [r for r in rows if r.get("text") and not r.get("dup_page") and len(r["text"]) >= 200]
    random.Random(seed).shuffle(rows); rows = rows[:pages]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True, max_model_len=6144, gpu_memory_utilization=0.9, dtype="bfloat16", enforce_eager=True)
    texts = [tok.apply_chat_template([{"role": "user", "content": prompt_for(r["text"])}], tokenize=False, add_generation_prompt=True) for r in rows]
    print(f"[docqa] {len(texts)} pages", flush=True)
    outs = llm.generate(texts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=400, stop=["<|im_end|>", "<|endoftext|>"]))
    kept = bad = 0
    with open(f"{V}/{dst}", "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            arr = first_json_list(o.outputs[0].text)
            if not isinstance(arr, list):
                bad += 1; continue
            for it in arr:
                if not isinstance(it, dict):
                    continue
                q, a = str(it.get("question", "")).strip(), str(it.get("answer", "")).strip()
                if 4 <= len(q) <= 200 and 1 <= len(a) <= 120 and re.search(r"[가-힣]", q):
                    f.write(json.dumps({"image": r["image"], "question": q, "answer": a, "kind": it.get("kind", ""), "src": r.get("src")}, ensure_ascii=False) + "\n"); kept += 1
    vol.commit()
    msg = f"[docqa] DONE {kept} QA pairs from {len(rows)} pages ({bad} unparseable) in {time.time()-t0:.0f}s -> {dst}"
    print(msg, flush=True); open(f"{V}/teacher.log", "a").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    return {"kept": kept, "bad": bad}
