"""Synthetic Korean tool-calling data from our own teacher (imcapsule/palette-k-midm) on Modal.

    modal run harness/teacher_gen.py --per-tool 24 --n-negative 1500

Writes /vol/sft/ko_raw.jsonl on the songgot volume: {"tool": name, "query": ..., "call": {...}}.
Only our own open-weight model produces labels (no closed-model outputs). FunctionChat-Bench
tool names and queries are never shown to the teacher; the assembly step applies a
disjointness gate against the benchmark anyway.
"""
import json
import os
import random
import time

import modal

app = modal.App("songgot-teacher")
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

STYLES = [
    "짧은 반말", "정중한 존댓말", "회사에서 쓰는 격식체", "친구에게 말하듯 편한 말투", "오타나 띄어쓰기 실수가 섞인 말투",
    "경상도 사투리", "전라도 사투리", "아주 짧게 단어만 나열하는 말투", "길게 상황을 설명한 뒤 요청하는 말투",
    "급하게 부탁하는 말투", "아이가 말하듯 쉬운 말투", "노년층이 말하듯 천천히 또박또박한 말투",
]
CONTEXTS = ["서울 강남", "부산 해운대", "대구", "제주도", "인천공항", "회사 사무실", "집 거실", "출근길 지하철", "주말 아침", "야근 중",
            "비 오는 저녁", "명절 연휴", "학교", "병원 대기실", "카페"]


def prompt_for(tool: dict, n: int, style: str, ctx: str) -> str:
    schema = json.dumps({"name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]}, ensure_ascii=False)
    return (
        "당신은 한국어 음성 비서용 학습 데이터를 만드는 전문가입니다.\n"
        f"아래 함수를 사용자가 실제로 부를 법한 한국어 요청 {n}개를 만들고, 각 요청에 대한 정확한 함수 호출 JSON을 붙이세요.\n"
        f"말투: {style}. 상황: {ctx}.\n"
        "규칙: 요청마다 값(장소, 사람 이름, 날짜, 시간, 금액, 수량)을 구체적이고 서로 다르게 쓰세요. 날짜는 YYYY-MM-DD, 시간은 24시간 HH:MM으로 정규화하세요. "
        "'내일', '모레', '다음 주 월요일'처럼 상대 표현이 나오면 오늘이 2026-09-10 (목요일)이라고 가정해 실제 날짜로 바꾸세요. "
        "enum이 있는 인자는 반드시 enum 값만 쓰세요. 필수 인자는 빠뜨리지 말고, 요청에 없는 선택 인자는 넣지 마세요. "
        "실제 상표명이나 실존 인물 이름은 쓰지 말고 평범한 이름(민수, 지은, 김 과장 등)과 일반 명사를 쓰세요.\n"
        "출력 형식: 한 줄에 하나씩, `요청 ||| {\"name\": \"...\", \"arguments\": {...}}`. 다른 말은 쓰지 마세요.\n\n"
        f"함수: {schema}"
    )


NEG_PROMPT = (
    "당신은 한국어 음성 비서용 학습 데이터를 만드는 전문가입니다. 비서가 가진 어떤 도구로도 처리할 수 없는, 그냥 대화나 잡담, 감상, 의견 질문, "
    "지식 질문 같은 한국어 발화를 {n}개 만드세요. 말투: {style}. 상황: {ctx}. 한 줄에 하나씩, 번호 없이 문장만 쓰세요. 실제 상표나 실존 인물은 쓰지 마세요."
)


@app.function(image=image, gpu="H100:2", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 3, memory=65536)
def generate(per_tool: int = 24, n_negative: int = 1500, model_id: str = "imcapsule/palette-k-midm"):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tools = json.load(open(f"{V}/data/tools_ko.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=0.9, dtype="bfloat16", enforce_eager=True)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1800, stop=["<|im_end|>", "<|endoftext|>"])
    rng = random.Random(7)
    jobs = []
    for t in tools:
        for _ in range(per_tool):
            jobs.append(("tool", t, prompt_for(t, 12, rng.choice(STYLES), rng.choice(CONTEXTS))))
    for _ in range(n_negative // 15):
        jobs.append(("neg", None, NEG_PROMPT.format(n=15, style=rng.choice(STYLES), ctx=rng.choice(CONTEXTS))))
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for _, _, p in jobs]
    print(f"[teacher] {len(jobs)} prompts", flush=True); t0 = time.time()
    outs = llm.generate(texts, sp)
    os.makedirs(f"{V}/sft", exist_ok=True)
    n_ok = n_bad = n_neg = 0
    with open(f"{V}/sft/ko_raw.jsonl", "w", encoding="utf-8") as f:
        for (kind, t, _), o in zip(jobs, outs):
            for line in o.outputs[0].text.splitlines():
                line = line.strip().lstrip("-•*0123456789.) ").strip()
                if not line:
                    continue
                if kind == "neg":
                    if 4 <= len(line) <= 120 and "|||" not in line:
                        f.write(json.dumps({"tool": None, "query": line, "call": {"name": "none", "arguments": {}}}, ensure_ascii=False) + "\n"); n_neg += 1
                    continue
                if "|||" not in line:
                    n_bad += 1; continue
                q, _, js = line.partition("|||")
                q = q.strip().strip("`\"'"); js = js.strip().strip("`")
                try:
                    call = json.loads(js)
                    assert call.get("name") == t["name"] and isinstance(call.get("arguments"), dict)
                    props = t["parameters"].get("properties", {})
                    assert all(k in props for k in call["arguments"])
                    for k, v in call["arguments"].items():
                        if "enum" in props[k]:
                            assert v in props[k]["enum"]
                    assert all(k in call["arguments"] for k in t["parameters"].get("required", []))
                except Exception:
                    n_bad += 1; continue
                if 3 <= len(q) <= 200:
                    f.write(json.dumps({"tool": t["name"], "query": q, "call": {"name": t["name"], "arguments": call["arguments"]}}, ensure_ascii=False) + "\n"); n_ok += 1
    vol.commit()
    msg = f"[teacher] ok {n_ok} bad {n_bad} neg {n_neg} in {time.time()-t0:.0f}s"
    print(msg, flush=True); open(f"{V}/teacher.log", "a").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    return {"ok": n_ok, "bad": n_bad, "neg": n_neg}


@app.local_entrypoint()
def main(per_tool: int = 24, n_negative: int = 1500):
    print(generate.remote(per_tool, n_negative))
