"""Synthetic Korean tool-calling data at catalogue scale, with our own teacher (palette-lab/palette-k-midm) and
labels checked by a second pass. Three stages, all on the same vLLM engine:
  0. invent tool schemas per domain and naming style (the v5 set has only 645 distinct real tools; the
     benchmark's 25 are all new to the model, so catalogue breadth is the lever);
  1. for every tool write three Korean requests in different registers plus the exact call JSON;
  2. re-derive the call from the request with the target tool among three distractors, keep only the pairs
     where name, keys and values agree.
Output rows {"domain","tool","query","call"} go to /vol/sft/synth_v6.jsonl; distractor sets, restyling and the
benchmark guard are applied locally by harness/assemble_sft_v6.py. No closed model touches the data.

    modal run --detach harness/teacher_synth.py --passes 3
"""
import json
import os
import random
import re
import time

import modal

app = modal.App("songgot-teacher-synth")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V = "/vol"; CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.27.1", "huggingface_hub[hf_transfer]", "transformers")
    .run_commands("pip uninstall -y flashinfer flashinfer-python || true")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
)

DOMAINS = ["배달 음식 주문", "카페 주문과 적립", "온라인 쇼핑", "중고 거래", "은행과 송금", "카드와 결제", "주식과 투자", "가계부와 지출 관리",
           "대중교통과 길찾기", "택시와 대리운전", "항공권과 기차표", "호텔과 숙소 예약", "여행 일정", "지도와 장소 검색", "주차와 주유",
           "일정과 캘린더", "알림과 타이머", "메모와 할 일", "이메일", "메신저와 문자", "전화와 연락처", "화상회의", "파일과 클라우드",
           "사진과 앨범", "음악 재생", "동영상과 OTT", "팟캐스트와 라디오", "뉴스와 날씨", "스포츠 경기 정보", "게임", "도서와 전자책",
           "번역과 사전", "학습과 강의", "학교와 학사", "직장과 인사", "채용과 이력서", "회의록과 문서", "고객 상담과 티켓", "재고와 물류",
           "매장 운영과 POS", "병원 예약과 진료", "약국과 복약", "운동과 헬스", "수면과 건강 데이터", "식단과 칼로리", "스마트홈 조명과 온도",
           "가전 제어", "자동차와 차량 관리", "부동산과 임대", "공공서비스와 민원", "세금과 연말정산", "보험", "통신 요금과 데이터",
           "구독 관리", "영화와 공연 예매", "미용실과 네일 예약", "반려동물", "육아와 아이 일정", "장보기와 식료품", "쿠폰과 멤버십"]
NAME_STYLES = ["snake_case", "camelCase", "PascalCase", "snake_case 뒤에 숫자 접미사(예: _v2, _2)", "동사+명사 camelCase(예: fetchOrders)",
               "짧은 snake_case 약어(예: get_bal)"]
STYLES = ["짧은 반말", "정중한 존댓말", "회사에서 쓰는 격식체", "친구에게 말하듯 편한 말투", "오타나 띄어쓰기 실수가 섞인 말투",
          "아주 짧게 단어만 나열하는 말투", "급하게 부탁하는 말투", "길게 상황을 설명한 뒤 요청하는 말투"]


SUBAREAS = ["검색과 조회", "생성과 등록", "수정과 취소", "알림과 구독", "통계와 리포트", "결제와 환불", "공유와 초대", "설정과 권한",
            "추천과 즐겨찾기", "이력과 기록", "위치 기반 기능", "관리자 기능", "고객용 기능", "일괄 처리", "실시간 상태"]


def p_schema(domain, style, k, sub="", avoid=()):
    extra = (f" 특히 '{sub}' 쪽 기능을 다루세요." if sub else "") + (f" 다음 이름과 겹치는 기능은 피하세요: {', '.join(avoid)}." if avoid else "")
    return (f"'{domain}' 분야의 앱이나 서비스가 제공할 법한 함수(도구) {k}개를 JSON 배열로 만드세요.{extra}\n"
            f"각 항목은 name(영문, {style}), description(한국어 한 문장), parameters(JSON Schema: \"type\":\"object\", properties, required)를 가집니다.\n"
            "properties는 0~4개, 각 속성에 type(string/integer/number/boolean/array)과 한국어 description을 쓰고, 일부 속성은 enum을 가지세요. "
            "매개변수가 전혀 없는 도구도 가끔 넣으세요(그때 properties는 {}). 서로 다른 기능이어야 하며 설명이나 코드 블록 없이 JSON 배열만 출력하세요.")


def p_siblings(tool, k=4):
    return (f"아래 도구와 혼동하기 쉬운, 같은 서비스 안의 비슷하지만 다른 기능의 도구 {k}개를 만드세요. 이름은 원래 도구와 같은 표기법(예: snake_case면 snake_case)으로, "
            "기능은 서로 겹치지 않아야 합니다(예: 조회 vs 수정 vs 취소 vs 목록). 각 항목은 name, description(한국어 한 문장), parameters(JSON Schema)를 가지며 JSON 배열만 출력하세요.\n\n"
            f"원래 도구: {json.dumps(tool, ensure_ascii=False)}")


def p_pairs(tool, styles):
    return ("아래 도구 스키마를 보고, 이 도구를 호출하게 만드는 서로 다른 한국어 사용자 요청 3개와 각 요청에 정확히 대응하는 호출 JSON을 만드세요.\n"
            f"요청 1의 말투: {styles[0]}. 요청 2의 말투: {styles[1]}. 요청 3의 말투: {styles[2]}.\n"
            "규칙: 호출의 인자 값은 요청 문장에서 그대로 알 수 있어야 합니다. 날짜와 시간이 필요하면 요청에 구체적으로 말하고(예: 9월 12일 오후 3시) 호출에는 2026-09-12, 15:00처럼 씁니다. "
            "required 속성은 반드시 채우고, 요청에 없는 값을 지어내지 마세요. 매개변수가 없는 도구면 arguments는 {}입니다.\n"
            "출력: JSON 배열만. 각 항목은 {\"query\": \"...\", \"call\": {\"name\": \"...\", \"arguments\": {...}}} 형식.\n\n"
            f"도구: {json.dumps(tool, ensure_ascii=False)}")


def p_verify(query, tools):
    return ("사용 가능한 도구 목록과 사용자 요청입니다. 요청을 수행할 도구 호출을 JSON으로만 출력하세요: {\"name\": \"...\", \"arguments\": {...}}. "
            "인자 값은 요청에서 그대로 가져오고, 날짜는 2026-09-12, 시간은 15:00 형식으로 씁니다. 맞는 도구가 없으면 {\"name\": \"none\", \"arguments\": {}}.\n\n"
            f"도구 목록: {json.dumps(tools, ensure_ascii=False)}\n\n요청: {query}")


def first_json(text, kind):
    """First JSON array/object in the text, tolerant of code fences and trailing prose."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("[" if kind == "list" else "{")
    if start < 0:
        return None
    depth = 0; in_str = False; esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\") if not esc else False
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def norm(v):
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, (int, float)):
        return ("n", float(v))
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return ("n", float(s))
        except ValueError:
            pass
        digits = re.sub(r"\D", "", s)
        return ("d", digits) if len(digits) >= 4 else ("s", re.sub(r"[\s\-_.,:/]", "", s))
    return ("j", json.dumps(v, sort_keys=True, ensure_ascii=False))


def agree(a: dict, b: dict) -> bool:
    if a.get("name") != b.get("name"):
        return False
    aa, ba = a.get("arguments") or {}, b.get("arguments") or {}
    if set(aa) != set(ba):
        return False
    return all(norm(aa[k]) == norm(ba[k]) for k in aa)


def valid_tool(t) -> bool:
    if not isinstance(t, dict) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{2,60}", str(t.get("name", ""))):
        return False
    p = t.get("parameters")
    if not isinstance(p, dict) or not isinstance(p.get("properties", {}), dict):
        return False
    if not isinstance(t.get("description"), str) or not re.search(r"[가-힣]", t["description"]):
        return False
    return all(isinstance(v, dict) and "type" in v for v in p.get("properties", {}).values())


@app.function(image=image, gpu="H100:2", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 3, memory=65536)
def synth(passes: int = 3, tools_per_prompt: int = 5, model_id: str = "palette-lab/palette-k-midm", dst: str = "sft/synth_v6.jsonl", seed: int = 6, sibling_share: float = 0.0):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.9, dtype="bfloat16", enforce_eager=True)
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    log = open(f"{V}/teacher.log", "a")
    def say(m):
        print(m, flush=True); log.write(time.strftime("%F %T ") + m + "\n"); log.flush()
    t0 = time.time()

    # stage 0: schemas
    prompts, meta = [], []
    existing = json.load(open(f"{V}/data/tool_catalogue.json", encoding="utf-8")) if os.path.exists(f"{V}/data/tool_catalogue.json") else []
    known = [t["name"] for t in existing if isinstance(t, dict) and "name" in t]
    for ps in range(passes):
        for d in DOMAINS:
            for st in NAME_STYLES:
                sub = rng.choice(SUBAREAS) if ps else ""
                prompts.append(chat(p_schema(d, st, tools_per_prompt, sub, rng.sample(known, 4) if known and ps else ()))); meta.append(d)
    # no fixed sampling seed: with one, identical prompts across passes return identical tools and dedupe to nothing
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1200, stop=["<|im_end|>", "<|endoftext|>"]))
    tools, seen = [], set()
    for t in existing:
        if valid_tool(t) or (isinstance(t, dict) and "name" in t):
            tools.append({"domain": "catalogue", "tool": t}); seen.add(t["name"].lower())
    n_bad = 0
    for d, o in zip(meta, outs):
        arr = first_json(o.outputs[0].text, "list")
        if not isinstance(arr, list):
            n_bad += 1; continue
        for t in arr:
            if valid_tool(t) and t["name"].lower() not in seen:
                seen.add(t["name"].lower()); tools.append({"domain": d, "tool": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}})
    say(f"[synth] stage 0: {len(prompts)} prompts -> {len(tools)} tools ({len(existing)} from the catalogue, {n_bad} unparseable) in {time.time()-t0:.0f}s")
    # stage 0b: confusable siblings for a share of the invented tools, so "close" distractor sets are hard
    if sibling_share > 0:
        base_idx = [i for i, e in enumerate(tools) if e["domain"] != "catalogue"]
        pick = rng.sample(base_idx, int(len(base_idx) * sibling_share))
        outs = llm.generate([chat(p_siblings(tools[i]["tool"])) for i in pick], SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1000, stop=["<|im_end|>", "<|endoftext|>"]))
        n_sib = 0
        for i, o in zip(pick, outs):
            arr = first_json(o.outputs[0].text, "list")
            for t in (arr if isinstance(arr, list) else []):
                if valid_tool(t) and t["name"].lower() not in seen:
                    seen.add(t["name"].lower()); tools.append({"domain": tools[i]["domain"], "tool": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}, "sibling_of": tools[i]["tool"]["name"]}); n_sib += 1
        say(f"[synth] stage 0b: {len(pick)} parents -> {n_sib} sibling tools in {time.time()-t0:.0f}s")
    json.dump(tools, open(f"{V}/{dst}".replace(".jsonl", "_tools.json"), "w", encoding="utf-8"), ensure_ascii=False); vol.commit()

    # stage 1: requests + calls
    prompts, meta = [], []
    for i, e in enumerate(tools):
        prompts.append(chat(p_pairs(e["tool"], rng.sample(STYLES, 3)))); meta.append(i)
    outs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=700, stop=["<|im_end|>", "<|endoftext|>"]))
    pairs = []
    for i, o in zip(meta, outs):
        arr = first_json(o.outputs[0].text, "list")
        if not isinstance(arr, list):
            continue
        for it in arr:
            if not isinstance(it, dict) or not isinstance(it.get("query"), str) or not isinstance(it.get("call"), dict):
                continue
            q = it["query"].strip(); c = it["call"]
            if not (4 <= len(q) <= 300) or not re.search(r"[가-힣]", q) or c.get("name") != tools[i]["tool"]["name"] or not isinstance(c.get("arguments", {}), dict):
                continue
            pairs.append({"i": i, "query": q, "call": {"name": c["name"], "arguments": c.get("arguments") or {}}})
    say(f"[synth] stage 1: {len(pairs)} candidate pairs from {len(tools)} tools in {time.time()-t0:.0f}s")

    # stage 2: verify with distractors present
    prompts = []
    for p in pairs:
        fam = [t for t in tools if t is not tools[p["i"]] and (t.get("sibling_of") == tools[p["i"]]["tool"]["name"] or tools[p["i"]].get("sibling_of") in (t["tool"]["name"], t.get("sibling_of")))]
        others = rng.sample(fam, min(2, len(fam))) + rng.sample([t for t in tools if t is not tools[p["i"]] and t not in fam], 3 - min(2, len(fam)))
        shown = [tools[p["i"]]["tool"]] + [t["tool"] for t in others]; rng.shuffle(shown)
        prompts.append(chat(p_verify(p["query"], shown)))
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=250, stop=["<|im_end|>", "<|endoftext|>"]))
    kept = 0; n_name = 0
    with open(f"{V}/{dst}", "w", encoding="utf-8") as f:
        for p, o in zip(pairs, outs):
            got = first_json(o.outputs[0].text, "dict")
            if not isinstance(got, dict):
                continue
            if got.get("name") == p["call"]["name"]:
                n_name += 1
            if agree(got, p["call"]):
                e = tools[p["i"]]
                f.write(json.dumps({"domain": e["domain"], "tool": e["tool"], "query": p["query"], "call": p["call"]}, ensure_ascii=False) + "\n"); kept += 1
    vol.commit()
    say(f"[synth] stage 2: kept {kept} of {len(pairs)} (name agreed {n_name}) -> {dst} in {time.time()-t0:.0f}s; DONE")
    return {"tools": len(tools), "pairs": len(pairs), "kept": kept}


@app.local_entrypoint()
def main(passes: int = 3, dst: str = "sft/synth_v6.jsonl", seed: int = 6, sibling_share: float = 0.0):
    print(synth.remote(passes, 5, "palette-lab/palette-k-midm", dst, seed, sibling_share))
