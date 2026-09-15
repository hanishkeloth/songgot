"""Targeted synthesis round v12: follow the parameter description. The v10 Songgot-X misses on FunctionChat that remain
after v11 are conventions the schema itself states (paper section 6): AddAlarm.time says "사용자의 표현 그대로 추출"
and the model normalises spacing; CreateEvent.start_datetime gives no format and the model writes ISO; memo is
optional and the model fills ""; QueryCalendar.query says "일정의 이름" and the model keeps "이번달"; send_message
wants the message as it would be sent and the model copies reported speech. Five modes, each with a deterministic
rule and a teacher verification pass, as in teacher_synth_v10.py / v11.py:

  descfmt    a date/time string parameter carries a format hint injected into its description ((YYYY-MM-DD), (HH:MM),
             (YYYY-MM-DD HH:MM), (사용자의 표현 그대로 추출. 예: ...)); the request states the value another way; the
             call follows the hint. The hinted schema is what the row stores.
  verbatimdt a date/time string parameter carries NO format hint; the call copies the request's expression verbatim.
  omit       a tool with optional parameters; the request mentions only some; the call omits the rest (no "" / null).
  narrow     a parameter whose description names a thing (이름, 제목, 상품명); the request wraps it in modifiers
             (이번달 테크세미나, 기영이 결혼식 날짜); the call carries only the named thing.
  direct     a message-sending tool; the request reports what to say (…늦는다고 문자 보내줘); the call carries the
             message as the recipient would read it (직접 화법).

    modal deploy harness/teacher_synth_v12.py
    .venv/bin/python harness/spawn.py songgot-teacher-v12 synth_targeted_v12 dst=sft/synth_v12.jsonl per_mode=6000
"""
import copy
import json
import os
import random
import re
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_synth import agree, first_json, image as base_image, p_verify, valid_tool  # noqa: E402
from teacher_synth_v10 import CACHE, STYLES, V, hf_cache, vol  # noqa: E402

image = base_image.add_local_python_source("teacher_synth", "teacher_synth_v10")
app = modal.App("songgot-teacher-v12")

HINTS = {
    "ymd": "(YYYY-MM-DD)",
    "hm": "(HH:MM, 24시간)",
    "ymdhm": "(YYYY-MM-DD HH:MM)",
    "verbatim": "(사용자의 표현 그대로 추출. 예: 5분 후, 내일 아침 7시, 6월 3일 오전 9시)",
}
DT_KEY = re.compile(r"(date|time|day|when|start|end|due|deadline|schedule|at$|_on$|일자|날짜|시각|시간|기한)", re.I)
DT_DESC = re.compile(r"(date|time|날짜|일자|시각|시간|기한|일시)", re.I)
NAME_DESC = re.compile(r"(이름|제목|상품명|명칭|title|name|subject)", re.I)
MSG_TOOL = re.compile(r"(send|message|sms|text|chat|kakao|talk|mail|notify|dm|reply|post|comment|remind|alert|announce|slack|line|memo|note)", re.I)
MSG_KEY = re.compile(r"(message|content|text|body|msg|내용|본문)", re.I)
MODIFIER = re.compile(r"^(이번\s?(주|달|해|년)|다음\s?(주|달|해|년)|지난\s?(주|달|해)|오늘|내일|모레|어제|올해|작년|내년)\s|\s(날짜|일정|시간|언제|정보|내용)$")
REPORTED = re.compile(r"(다고|라고|냐고|자고|달라고|하라고|겠다고)\s")
TIMEWORD = re.compile(r"(\d|오전|오후|아침|저녁|밤|새벽|정오|자정|시|분|월|일|내일|모레|다음|이번|주|말)")

MODE_RULES = {
    "descfmt": ("이 도구의 날짜/시간 매개변수 설명에는 형식 지정이 괄호로 적혀 있습니다. 요청에서는 그 값을 형식과 다르게 자연스럽게 쓰세요"
                "(예: 형식이 YYYY-MM-DD이면 요청에는 '6월 3일'이나 '다음 달 15일', 형식이 HH:MM이면 요청에는 '오후 세 시 반', 형식이 '사용자의 표현 그대로'이면 요청에는 '내일 아침 7시'). "
                "호출에서는 반드시 설명의 형식을 따르세요: YYYY-MM-DD면 2026-06-03처럼(연도가 없으면 2026년), HH:MM이면 15:30처럼, '사용자의 표현 그대로'면 요청에 적힌 글자 그대로 복사합니다."),
    "verbatimdt": ("이 도구의 날짜/시간 매개변수 설명에는 형식 지정이 없습니다. 요청의 날짜/시간은 '6월 3일 오전 9시', '다음 주 월요일 저녁 7시', '10분후', '2023.11.3 오후5시'처럼 자연스럽게 쓰고, "
                   "호출에서는 그 표현을 요청에 적힌 글자 그대로(띄어쓰기까지) 복사합니다. 절대 ISO 형식이나 숫자 형식으로 바꾸지 마세요."),
    "omit": ("이 도구에는 필수가 아닌 매개변수가 있습니다. 요청에서는 그중 일부만 언급하세요. 호출에는 요청에 있는 값만 넣고, 언급되지 않은 선택 매개변수는 키 자체를 넣지 마세요. "
             "빈 문자열이나 null을 넣는 것은 틀린 것입니다."),
    "narrow": ("이 도구의 매개변수 설명은 특정한 것의 이름/제목만 요구합니다(예: '일정의 이름', '상품 이름'). 요청에서는 그 이름 앞뒤에 수식어를 붙여 쓰세요"
               "(예: '이번달 테크세미나가 언제였더라?', '기영이 결혼식 날짜 찾아줘', '부모님 드릴 건강식품 좀 검색해줘'). 호출의 값에는 수식어와 '날짜/일정/언제' 같은 말을 빼고 이름만 넣습니다"
               "(테크세미나, 기영이 결혼식, 건강식품)."),
    "direct": ("이 도구는 메시지를 보냅니다. 요청은 전달할 말을 간접 화법으로 쓰세요(예: '엄마한테 오늘 저녁 먹고 들어간다고 문자 보내줘', '클로이한테 저번 회식 때 간 식당 이름이 뭐였냐고 물어봐줘'). "
               "호출의 메시지 값은 받는 사람이 읽을 문장으로 직접 화법으로 바꿔 씁니다(예: '오늘 저녁 먹고 들어가요.', '저번 회식 때 간 식당 이름이 뭐였죠?'). "
               "요청의 '보내줘/물어봐줘/전해줘' 같은 지시어를 메시지에 넣으면 틀린 것입니다."),
}


def props_of(tool):
    p = tool.get("parameters") or {}
    return p.get("properties") if isinstance(p.get("properties"), dict) else {k: v for k, v in p.items() if isinstance(v, dict) and "type" in v}


def required_of(tool):
    r = (tool.get("parameters") or {}).get("required")
    return set(r) if isinstance(r, list) else set()  # some teacher schemas write "required": true


def dt_params(tool):
    return [k for k, v in props_of(tool).items() if str((v or {}).get("type", "")) == "string" and (DT_KEY.search(k) or DT_DESC.search(str((v or {}).get("description", "") or "")))]


def p_mode(tool, styles, mode):
    return ("아래 도구 스키마를 보고, 이 도구를 호출하게 만드는 서로 다른 한국어 사용자 요청 3개와 각 요청에 정확히 대응하는 호출 JSON을 만드세요.\n"
            f"요청 1의 말투: {styles[0]}. 요청 2의 말투: {styles[1]}. 요청 3의 말투: {styles[2]}.\n"
            f"이번 규칙: {MODE_RULES[mode]}\n"
            "공통 규칙: required 속성은 반드시 채우고, 요청에 없는 값을 지어내지 마세요. 요청에 있는 값은 빠짐없이 인자에 넣으세요.\n"
            "출력: JSON 배열만. 각 항목은 {\"query\": \"...\", \"call\": {\"name\": \"...\", \"arguments\": {...}}} 형식.\n\n"
            f"도구: {json.dumps(tool, ensure_ascii=False)}")


def prepare(tool, mode, rng):
    """Return the tool as the row will store it (with an injected hint for descfmt, hints stripped for verbatimdt) and the hint kind."""
    t = copy.deepcopy(tool); pr = props_of(t)
    if mode == "descfmt":
        keys = dt_params(t)
        if not keys:
            return None, None
        kind = rng.choice(["ymd", "hm", "ymdhm", "verbatim", "verbatim"])
        for k in keys:
            d = re.sub(r"\((YYYY|HH|사용자의 표현)[^)]*\)", "", str(pr[k].get("description", "") or "")).strip()
            pr[k]["description"] = (d + " " + HINTS[kind]).strip()
        return t, kind
    if mode == "verbatimdt":
        keys = dt_params(t)
        if not keys:
            return None, None
        for k in keys:
            pr[k]["description"] = re.sub(r"\((YYYY|HH|ISO|사용자의 표현)[^)]*\)", "", str(pr[k].get("description", "") or "")).strip() or "날짜 및 시간"
        return t, "verbatim"
    return t, None


def rule_ok(mode, kind, tool, query, call):
    args = call.get("arguments") or {}
    if not args or any(v is None or (isinstance(v, str) and not v.strip()) for v in args.values()):
        return False
    pr = props_of(tool); required = required_of(tool)
    if mode in ("descfmt", "verbatimdt"):
        keys = [k for k in dt_params(tool) if k in args]
        if not keys:
            return False
        for k in keys:
            v = str(args[k]).strip()
            if kind == "ymd":
                m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v)
                if not m or re.search(r"\d{4}-\d{2}-\d{2}", query):
                    return False
                if not re.search(rf"(^|[^\d]){int(m.group(2))}\s?월", query) or not re.search(rf"(^|[^\d]){int(m.group(3))}\s?일", query):
                    return False  # month and day must be stated in the request; 말/첫날 forms are left to v10
            elif kind == "hm":
                m = re.fullmatch(r"(\d{2}):(\d{2})", v)
                if not m or re.search(r"\d{1,2}:\d{2}", query) or not re.search(r"시", query):
                    return False
            elif kind == "ymdhm":
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", v) or re.search(r"\d{4}-\d{2}-\d{2}", query):
                    return False
            else:  # verbatim: every token of the value is in the request (range ends may repeat the date: "6월 3일 오전 11시")
                if not TIMEWORD.search(v) or re.fullmatch(r"[\d:\-T ]+", v) or re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}", v) and not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}", query):
                    return False
                if not all(tk in query for tk in v.split()):
                    return False
        return all(str(v) in query for k, v in args.items() if k not in keys and isinstance(v, str))
    if mode == "omit":
        optional = [k for k in pr if k not in required]
        if len(pr) < 3 or not optional or all(k in args for k in optional):
            return False
        return all((not isinstance(v, str)) or v.strip() in query or re.fullmatch(r"[\d.:\-]+", v.strip()) for v in args.values())
    if mode == "narrow":
        keys = [k for k, v in pr.items() if k in args and isinstance(args[k], str) and NAME_DESC.search(str((v or {}).get("description", "") or ""))]
        if not keys:
            return False
        for k in keys:
            v = args[k].strip()
            if v not in query or len(v) > 24 or MODIFIER.search(v) or v == query.strip():
                return False
            i = query.index(v); around = query[max(0, i - 8):i] + "|" + query[i + len(v):i + len(v) + 8]
            if not re.search(r"(이번|다음|지난|오늘|내일|모레|올해|작년|날짜|일정|언제|드릴|위한|좀|쫌)", around):
                return False  # the request must actually carry a modifier that the value dropped
        return all(str(v) in query for k, v in args.items() if k not in keys and isinstance(v, str))
    if mode == "direct":
        if not MSG_TOOL.search(tool["name"]) or not REPORTED.search(query):
            return False
        keys = [k for k in args if MSG_KEY.search(k) and isinstance(args[k], str)]
        if not keys:
            return False
        v = args[keys[0]].strip()
        if v in query or len(v) < 4 or re.search(r"(보내|전해|문자|메시지|메세지|물어봐|알려줘|해줘)\s*[.!?~]*$", v) or REPORTED.search(v + " "):
            return False
        return all(str(o) in query for k, o in args.items() if k not in keys and isinstance(o, str))
    return False


def pick_tools(tools, mode, n, rng):
    out = []
    for t in tools:
        pr = props_of(t)
        if not pr:
            continue
        if mode in ("descfmt", "verbatimdt") and dt_params(t):
            out.append(t)
        elif mode == "omit" and len(pr) >= 3 and any(k not in required_of(t) for k in pr):
            out.append(t)
        elif mode == "narrow" and any(str((v or {}).get("type", "")) == "string" and NAME_DESC.search(str((v or {}).get("description", "") or "")) for v in pr.values()):
            out.append(t)
        elif mode == "direct" and MSG_TOOL.search(t["name"]) and any(MSG_KEY.search(k) and str((v or {}).get("type", "")) == "string" for k, v in pr.items()):
            out.append(t)
    rng.shuffle(out)
    return out[:n]


@app.function(image=image, gpu="H100:2", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 6, memory=65536)
def synth_targeted_v12(tools: str = "sft/synth_tools_v9_all.json", dst: str = "sft/synth_v12.jsonl", per_mode: int = 6000, model_id: str = "palette-lab/palette-k-midm", seed: int = 12, modes: str = "descfmt,verbatimdt,omit,narrow,direct"):
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
    print(f"[v12] {len(pool)} distinct tools from {tools}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.9, dtype="bfloat16", enforce_eager=True)
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=900, stop=["<|im_end|>", "<|endoftext|>"])
    spv = SamplingParams(temperature=0.0, max_tokens=300, stop=["<|im_end|>", "<|endoftext|>"])
    log = open(f"{V}/teacher.log", "a")
    say = lambda m: (print(m, flush=True), log.write(time.strftime("%F %T ") + m + "\n"), log.flush())
    kept = []
    for mode in [m.strip() for m in modes.split(",") if m.strip()]:
        chosen = pick_tools(pool, mode, per_mode // 3 + 1, rng)
        if len(chosen) * 3 < per_mode:
            reps = per_mode // max(1, len(chosen) * 3) + 1
            chosen = [t for t in chosen for _ in range(reps)][: per_mode // 3 + 1]
        prepared = [prepare(t, mode, rng) for t in chosen]
        chosen = [(t, k) for t, k in prepared if t]
        say(f"[v12] mode {mode}: {len(chosen)} tool prompts -> {len(chosen) * 3} candidate pairs")
        prompts = [chat(p_mode(t, rng.sample(STYLES, 3), mode)) for t, _ in chosen]
        outs = llm.generate(prompts, sp)
        cands = []
        for (t, kind), o in zip(chosen, outs):
            arr = first_json(o.outputs[0].text, "list") or []
            for it in arr[:3]:
                if not isinstance(it, dict) or not isinstance(it.get("query"), str) or not isinstance(it.get("call"), dict):
                    continue
                call = it["call"]
                if call.get("name") != t["name"] or not isinstance(call.get("arguments"), dict):
                    continue
                if not rule_ok(mode, kind, t, it["query"], call):
                    continue
                cands.append({"tool": t, "query": it["query"].strip(), "call": call})
        say(f"[v12] mode {mode}: {len(cands)} pass the rule; verifying")
        vprompts = []
        for c in cands:
            distract = rng.sample(pool, 4)
            tools_shown = [c["tool"]] + distract; rng.shuffle(tools_shown)
            vprompts.append(chat(p_verify(c["query"], tools_shown)))
        vouts = llm.generate(vprompts, spv)
        n_ok = 0
        # descfmt/verbatimdt: the rule pins the value; verify tool choice only (the teacher itself normalises dates, the
        # error being corrected). omit/narrow/direct: full agreement, the boundary of the value is the point.
        for c, o in zip(cands, vouts):
            v = first_json(o.outputs[0].text, "dict")
            if v and (v.get("name") == c["call"]["name"] if mode in ("descfmt", "verbatimdt") else agree(v, c["call"])):
                kept.append({"lang": "ko", "query": c["query"], "tools": [c["tool"]], "call": c["call"], "cond": f"v12_{mode}"}); n_ok += 1
        say(f"[v12] mode {mode}: {n_ok} agreed and kept (total {len(kept)})")
        os.makedirs(os.path.dirname(f"{V}/{dst}"), exist_ok=True)
        with open(f"{V}/{dst}", "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        vol.commit()
    say(f"[v12] wrote {dst}: {len(kept)} rows")
    return {"rows": len(kept)}
