"""Targeted synthesis v10: the four argument-value patterns that separate Songgot-X from the FunctionChat leader
(error analysis 2026-09-14: of 79 items Kanana-2 gets and Songgot-X misses, 56 are values present verbatim in the
query, 10 are Korean number words, 5 are two-digit years). Each mode asks the teacher for requests that exercise one
pattern over existing tools, then keeps a pair only if (a) an independent verification pass agrees and (b) a
deterministic rule for the pattern holds. No benchmark tool names or queries are used (assemble step re-checks).

    modal deploy harness/teacher_synth_v10.py
    .venv/bin/python harness/spawn.py songgot-teacher-v10 synth_targeted dst=sft/synth_v10.jsonl per_mode=30000

Modes
  copy    names, titles, places copied verbatim: no spacing change, no added or dropped particles, parentheses,
          Latin letters and digits kept as written ("로맨틱홀리데이(2006)", "stella(김윤경)", "GS25 역삼점")
  roles   two-slot tools bound by particles: A에서 B까지 / A부터 B로 / A를 B로 / B까지 A에서 (reversed order),
          origin/destination, from/to, sender/recipient, old/new, start/end, source/target
  numeral Korean number words to digits: 오만원 -> 50000, 십오프로 -> 15, 삼십 분 -> 30, 두 시간 반 -> 2.5, 백이십 명 -> 120
  year    two-digit and spoken years: 25년 6월 1일 -> 2025-06-01, 이천이십오년 -> 2025, 24년 말 -> 2024-12-31
"""
import json
import os
import random
import re
import time

import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_synth import agree, first_json, image as base_image, p_verify, valid_tool  # noqa: E402

image = base_image.add_local_python_source("teacher_synth")

app = modal.App("songgot-teacher-v10")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V = "/vol"; CACHE = "/root/.cache/huggingface"

STYLES = ["짧은 반말", "정중한 존댓말", "친구에게 말하듯 편한 말투", "급하게 부탁하는 말투", "띄어쓰기 실수가 섞인 말투", "길게 상황을 설명한 뒤 요청하는 말투"]

MODE_RULES = {
    "copy": ("요청에는 고유명사(사람 이름, 가게·장소 이름, 영화·책·노래 제목, 상품명)를 하나 이상 넣으세요. 그 이름은 일부러 까다롭게 쓰세요: "
             "띄어쓰기 없이 붙여 쓴 제목, 괄호가 붙은 이름(예: 로맨틱홀리데이(2006), stella(김윤경)), 영문과 한글이 섞인 이름(예: GS25 역삼점, iPhone 17 프로), "
             "숫자가 들어간 이름. 호출의 인자 값은 요청에 적힌 글자 그대로(띄어쓰기, 괄호, 대소문자까지) 복사해야 합니다. 조사(은/는/이/가/을/를/에서/까지)는 값에 넣지 않습니다."),
    "roles": ("이 도구에는 출발/도착, 보낸 사람/받는 사람, 이전 값/새 값, 시작/끝처럼 짝을 이루는 매개변수가 있습니다. 요청 3개는 각각 다른 어순으로 쓰세요: "
              "하나는 'A에서 B까지', 하나는 'B까지 A에서'처럼 순서를 뒤집어서, 하나는 'A를 B로 바꿔줘'처럼. 호출에서는 조사(에서/부터/를)로 역할을 정확히 판단해 알맞은 매개변수에 넣어야 합니다."),
    "numeral": ("요청의 모든 숫자(금액, 비율, 개수, 시간 길이, 거리)는 반드시 한글 숫자말로만 쓰세요(예: 오만원, 십오프로, 삼십 분, 두 시간 반, 백이십 명, 천오백 원, 만 이천 원). 아라비아 숫자를 요청에 쓰면 안 됩니다. "
                "호출에서는 그 값을 숫자로 바꿔 씁니다(오만원 -> 50000, 십오프로 -> 15, 두 시간 반 -> 2.5, 삼십 분 -> 30). 단위나 '프로', '퍼센트' 글자는 값에 넣지 않습니다."),
    "year": ("요청의 날짜는 두 자리 연도나 말로 쓴 연도로만 쓰세요(예: 25년 6월 1일, 24년 말, 이천이십오년 3월 15일, 26년 1월 첫날). 호출에서는 2025-06-01처럼 네 자리 연도의 ISO 날짜로 씁니다. "
             "25년은 2025년, 24년은 2024년입니다. 절대 다른 연도로 바꾸지 마세요. '말'은 그 달이나 해의 마지막 날, '첫날'은 1일입니다."),
}


def p_mode(tool, styles, mode):
    return ("아래 도구 스키마를 보고, 이 도구를 호출하게 만드는 서로 다른 한국어 사용자 요청 3개와 각 요청에 정확히 대응하는 호출 JSON을 만드세요.\n"
            f"요청 1의 말투: {styles[0]}. 요청 2의 말투: {styles[1]}. 요청 3의 말투: {styles[2]}.\n"
            f"이번 규칙: {MODE_RULES[mode]}\n"
            "공통 규칙: required 속성은 반드시 채우고, 요청에 없는 값을 지어내지 마세요. 요청에 있는 값은 빠짐없이 인자에 넣으세요.\n"
            "출력: JSON 배열만. 각 항목은 {\"query\": \"...\", \"call\": {\"name\": \"...\", \"arguments\": {...}}} 형식.\n\n"
            f"도구: {json.dumps(tool, ensure_ascii=False)}")


# ---------------- deterministic pattern checks ----------------
DIGITS = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
NATIVE = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "스무": 20, "스물": 20, "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80, "아흔": 90}
SMALL = {"십": 10, "백": 100, "천": 1000}
BIG = {"만": 10_000, "억": 100_000_000}


def ko_number(s: str):
    """Sino-Korean number words to int: 오만원 -> 50000, 십오 -> 15, 천오백 -> 1500, 만 이천 -> 12000, 백이십 -> 120. None if not parseable."""
    t = re.sub(r"[\s,원명개분번프로퍼센트%시간분초km킬로미터]", "", s)
    if not t or any(ch not in "영공일이삼사오육칠팔구십백천만억" for ch in t):
        return None
    total = 0; section = 0; num = 0
    for ch in t:
        if ch in DIGITS:
            num = DIGITS[ch]
        elif ch in SMALL:
            section += (num or 1) * SMALL[ch]; num = 0
        elif ch in BIG:
            total += (section + (num or (0 if section else 1))) * BIG[ch] if (section or num) else BIG[ch]; section = 0; num = 0
    return total + section + num


def native_number(s: str):
    s = s.strip()
    for k in sorted(NATIVE, key=len, reverse=True):
        if s.startswith(k):
            rest = s[len(k):]
            if not rest:
                return NATIVE[k]
            for k2 in sorted(NATIVE, key=len, reverse=True):
                if rest.startswith(k2) and NATIVE[k] >= 10 and NATIVE[k2] < 10:
                    return NATIVE[k] + NATIVE[k2]
    return None


def rule_ok(mode: str, query: str, call: dict) -> bool:
    args = call.get("arguments") or {}
    if not args:
        return False
    if mode == "copy":
        strs = [v for v in args.values() if isinstance(v, str) and not re.fullmatch(r"[\d:.\-T ]+", v)]
        return bool(strs) and all(v in query for v in strs)
    if mode == "roles":
        strs = [v for v in args.values() if isinstance(v, str) and len(v) >= 2]
        return len(strs) >= 2 and all(v in query for v in strs)
    if mode == "numeral":
        if re.search(r"\d", query):
            return False
        nums = [v for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not nums:
            return False
        words = re.findall(r"[영공일이삼사오육칠팔구십백천만억]{1,12}|[한두세네]\s?시간\s?반|[한두세네]|하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|열|스무|스물|서른|마흔|쉰", query)
        cands = set()
        for w in words:
            if "시간" in w and "반" in w:
                base = native_number(w[0]); cands.add(base + 0.5 if base else None)
            v = ko_number(w); cands.add(v)
            v2 = native_number(w); cands.add(v2)
            if v is not None:
                cands.add(v * 60)  # 두 시간 -> 120 minutes is also accepted downstream
        cands.discard(None)
        return all(n in cands for n in nums)
    if mode == "year":
        dates = [v for v in args.values() if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)]
        if not dates:
            return False
        for d in dates:
            yy = d[2:4]
            if not (re.search(rf"(^|[^\d]){yy}년", query) or re.search(r"[이일]천[영공일이삼사오육칠팔구십]*년", query)):
                return False
            if d[:4] not in ("2024", "2025", "2026", "2027", "2023"):
                return False
        return True
    return False


def pick_tools(tools, mode, n, rng):
    """Tools whose schema suits the mode: string params for copy, paired params for roles, numeric params for numeral, date-like params for year."""
    def props(t):
        p = t.get("parameters") or {}
        return p.get("properties") if isinstance(p.get("properties"), dict) else {k: v for k, v in p.items() if isinstance(v, dict) and "type" in v}
    ROLE = re.compile(r"(origin|destination|from|to|source|target|sender|recipient|old|new|start|end|departure|arrival)", re.I)
    out = []
    for t in tools:
        p = props(t)
        if not p:
            continue
        types = [(k, str((v or {}).get("type", "")), str((v or {}).get("description", "") or "")) for k, v in p.items()]
        if mode == "copy" and any(ty == "string" for _, ty, _ in types):
            out.append(t)
        elif mode == "roles" and sum(1 for k, ty, _ in types if ty == "string" and ROLE.search(k)) >= 2:
            out.append(t)
        elif mode == "numeral" and any(ty in ("number", "integer") for _, ty, _ in types):
            out.append(t)
        elif mode == "year" and any(("date" in k.lower() or "날짜" in d or "date" in d.lower()) and ty == "string" for k, ty, d in types):
            out.append(t)
    rng.shuffle(out)
    return out[:n]


@app.function(image=image, gpu="H100:2", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 6, memory=65536)
def synth_targeted(tools: str = "sft/synth_tools_v9_all.json", dst: str = "sft/synth_v10.jsonl", per_mode: int = 12000, model_id: str = "palette-lab/palette-k-midm", seed: int = 10):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    vol.reload()
    rng = random.Random(seed)
    import glob
    raw = []
    for path in sorted(glob.glob(f"{V}/{tools}")):
        raw += json.load(open(path, encoding="utf-8"))
    seen = set(); pool = []
    for e in raw:
        t = e["tool"] if isinstance(e, dict) and "tool" in e else e
        if valid_tool(t) and t["name"].lower() not in seen:
            seen.add(t["name"].lower()); pool.append(t)
    print(f"[v10] {len(pool)} distinct tools from {tools}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.9, dtype="bfloat16", enforce_eager=True)
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=900, stop=["<|im_end|>", "<|endoftext|>"])
    spv = SamplingParams(temperature=0.0, max_tokens=300, stop=["<|im_end|>", "<|endoftext|>"])
    log = open(f"{V}/teacher.log", "a")
    say = lambda m: (print(m, flush=True), log.write(time.strftime("%F %T ") + m + "\n"), log.flush())
    kept = []
    for mode in ("copy", "roles", "numeral", "year"):
        chosen = pick_tools(pool, mode, per_mode // 3 + 1, rng)
        say(f"[v10] mode {mode}: {len(chosen)} tools -> {len(chosen) * 3} candidate pairs")
        prompts = [chat(p_mode(t, rng.sample(STYLES, 3), mode)) for t in chosen]
        outs = llm.generate(prompts, sp)
        cands = []
        for t, o in zip(chosen, outs):
            arr = first_json(o.outputs[0].text, "list") or []
            for it in arr[:3]:
                if not isinstance(it, dict) or not isinstance(it.get("query"), str) or not isinstance(it.get("call"), dict):
                    continue
                call = it["call"]
                if call.get("name") != t["name"] or not isinstance(call.get("arguments"), dict):
                    continue
                if not rule_ok(mode, it["query"], call):
                    continue
                cands.append({"tool": t, "query": it["query"].strip(), "call": call})
        say(f"[v10] mode {mode}: {len(cands)} pass the rule; verifying")
        # verification: the teacher, shown the tool plus 4 random distractors, must produce the same call
        vprompts = []
        for c in cands:
            distract = rng.sample(pool, 4)
            tools_shown = [c["tool"]] + distract; rng.shuffle(tools_shown)
            vprompts.append(chat(p_verify(c["query"], tools_shown)))
        vouts = llm.generate(vprompts, spv)
        n_ok = 0
        # copy/roles: the verifier must reproduce the exact call. numeral/year: the deterministic rule already fixed the
        # values (the teacher itself often writes 2026 for "25년", which is the error being corrected), so only the
        # tool choice is verified.
        for c, o in zip(cands, vouts):
            v = first_json(o.outputs[0].text, "dict")
            if v and (agree(v, c["call"]) if mode in ("copy", "roles") else v.get("name") == c["call"]["name"]):
                kept.append({"lang": "ko", "query": c["query"], "tools": [c["tool"]], "call": c["call"], "cond": f"v10_{mode}"}); n_ok += 1
        say(f"[v10] mode {mode}: {n_ok} agreed and kept (total {len(kept)})")
        os.makedirs(os.path.dirname(f"{V}/{dst}"), exist_ok=True)
        with open(f"{V}/{dst}", "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        vol.commit()
    say(f"[v10] wrote {dst}: {len(kept)} rows")
    return {"rows": len(kept)}
