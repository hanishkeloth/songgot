"""Targeted synthesis round v13: the four miss classes left in the published three-way soup on FunctionChat SingleCall
(paper section 6, 2026-09-16), each with a deterministic rule and a teacher verification pass as in v10-v12.

  verbatimtime  alarm/timer/reminder-style tools with a free-text time parameter; the request carries an expression such
                as "10분후", "3시간 뒤", "내일 아침 6시반", "16일 오후 두시" and the call copies it as an EXACT substring
                (v12 accepted token-level matches, which let "10분 후" through for "10분후").
  memoomit      tools with an optional free-text parameter (memo, note, description, comment, detail) beside required
                structured ones; the request gives only the structured values; the call omits the free-text key.
  oldnew        update/rename tools with an old/new parameter pair (name/new_name, old_x/new_x, current/target); the
                request states both; the call puts each in its own slot.
  intent        a query-type tool (when / how many days / is there) and a create-type tool (add / register / set) with
                the same entity are BOTH shown; the request asks a question or gives a command; the call picks by intent.

    modal deploy harness/teacher_synth_v13.py
    .venv/bin/python harness/spawn.py songgot-teacher-v13 synth_targeted_v13 dst=sft/synth_v13.jsonl per_mode=6000
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
from teacher_synth_v12 import props_of, required_of  # noqa: E402

image = base_image.add_local_python_source("teacher_synth", "teacher_synth_v10", "teacher_synth_v12")
app = modal.App("songgot-teacher-v13")

TIME_TOOL = re.compile(r"(alarm|timer|remind|wake|알람|타이머|리마인드)", re.I)
TIME_KEY = re.compile(r"(^time$|when|at$|_at|time$|시간|시각)", re.I)
MEMO_KEY = re.compile(r"(memo|note|desc|comment|detail|remark|additional|extra|reason|메모|설명|비고|상세)", re.I)
OLDNEW = [(re.compile(r"^(name|old_name|current_name|from_name)$", re.I), re.compile(r"^(new_name|to_name|target_name)$", re.I)),
          (re.compile(r"^old_", re.I), re.compile(r"^new_", re.I)), (re.compile(r"^current_", re.I), re.compile(r"^(new_|target_)", re.I))]
QUERY_TOOL = re.compile(r"(query|search|find|get|check|info|inform|dday|d_day|count|list|lookup|status|remain|조회|검색|확인)", re.I)
CREATE_TOOL = re.compile(r"(create|add|set|register|schedule|insert|new|make|book|추가|등록|생성|설정)", re.I)
ASK = re.compile(r"(언제|며칠|몇\s?일|얼마나\s?남|남았|있어\?|있나요|있니|인가요|일까|였더라|뭐야|뭐지|알려줘|확인해)")
CMD = re.compile(r"(추가해|등록해|잡아|만들어|넣어|설정해|생성해|잡아줘|추가|등록|저장해)")

MODE_RULES = {
    "verbatimtime": ("이 도구의 시간 매개변수 설명에는 '사용자의 표현 그대로 추출'이라고 적혀 있습니다. 요청에는 시간을 '10분후', '3시간 뒤', '내일 아침 6시반', '16일 오후 두시', '삼십분 후에', "
                     "'모레 저녁 7시 반'처럼 다양한 띄어쓰기와 말로 쓰세요(세 요청은 서로 다른 형태로). 호출의 시간 값은 요청에 적힌 글자를 띄어쓰기까지 그대로, '뒤/후/전'까지 포함해서 복사합니다. "
                     "절대 띄어쓰기를 고치거나 숫자로 바꾸거나 '에/에는' 같은 조사를 붙이지 마세요."),
    "memoomit": ("이 도구에는 memo/note/description 같은 자유 서술 매개변수가 있지만 필수는 아닙니다. 요청에는 이름, 날짜, 시간, 장소 같은 값만 쓰고 별도의 메모 내용은 쓰지 마세요. "
                 "호출에는 요청에 있는 값만 넣고, memo/note/description 키는 아예 넣지 마세요(빈 문자열도 안 됩니다). 날짜/시간 값은 요청에 적힌 표현 그대로 복사합니다."),
    "oldnew": ("이 도구는 값을 바꾸는 도구로, 이전 값과 새 값을 받는 매개변수 짝이 있습니다(예: name과 new_name, old_email과 new_email). 요청 3개는 각각 다른 어순으로 쓰세요: "
               "'A를 B로 바꿔줘', 'B로 바꿔줘, 지금은 A야', 'A 있잖아, 그거 B로'. 호출에서는 어순과 관계없이 지금 값이 이전 슬롯에, 바꿀 값이 새 슬롯에 들어가야 합니다."),
    "intent": ("도구 목록에 같은 대상을 다루는 조회용 도구와 등록용 도구가 함께 있습니다. 요청 3개 중 2개는 질문(언제였더라, 며칠 남았어, 있나요)으로, 1개는 명령(추가해줘, 등록해줘, 잡아줘)으로 쓰세요. "
               "질문이면 조회용 도구를, 명령이면 등록용 도구를 호출합니다. 질문에 나온 날짜는 조회 도구의 검색어에 넣지 말고 대상 이름만 넣습니다."),
}


def p_mode(tools, styles, mode):
    shown = tools if isinstance(tools, list) else [tools]
    return ("아래 도구 스키마를 보고, 이 도구(들)를 호출하게 만드는 서로 다른 한국어 사용자 요청 3개와 각 요청에 정확히 대응하는 호출 JSON을 만드세요.\n"
            f"요청 1의 말투: {styles[0]}. 요청 2의 말투: {styles[1]}. 요청 3의 말투: {styles[2]}.\n"
            f"이번 규칙: {MODE_RULES[mode]}\n"
            "공통 규칙: required 속성은 반드시 채우고, 요청에 없는 값을 지어내지 마세요. 요청에 있는 값은 빠짐없이 인자에 넣으세요.\n"
            "출력: JSON 배열만. 각 항목은 {\"query\": \"...\", \"call\": {\"name\": \"...\", \"arguments\": {...}}} 형식.\n\n"
            f"도구: {json.dumps(shown, ensure_ascii=False)}")


def time_params(tool):
    return [k for k, v in props_of(tool).items() if str((v or {}).get("type", "")) == "string" and TIME_KEY.search(k)]


def memo_params(tool):
    req = required_of(tool)
    return [k for k, v in props_of(tool).items() if str((v or {}).get("type", "")) == "string" and MEMO_KEY.search(k) and k not in req]


def oldnew_pairs(tool):
    keys = list(props_of(tool)); out = []
    for a, b in OLDNEW:
        olds = [k for k in keys if a.search(k)]; news = [k for k in keys if b.search(k)]
        if olds and news:
            out.append((olds[0], news[0]))
    return out


def prepare(tool, mode):
    t = copy.deepcopy(tool); pr = props_of(t)
    if mode == "verbatimtime":
        for k in time_params(t):
            d = re.sub(r"\((YYYY|HH|ISO|사용자의 표현)[^)]*\)", "", str(pr[k].get("description", "") or "")).strip()
            pr[k]["description"] = (d + " (사용자의 표현 그대로 추출. 예: 5분 후, 내일 아침 7시)").strip()
    return t


def rule_ok(mode, tools, query, call):
    args = call.get("arguments") or {}
    if not args or any(v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v) for v in args.values()):
        return False
    tool = next((t for t in tools if t["name"] == call.get("name")), None)
    if tool is None:
        return False
    grounded = all((not isinstance(v, str)) or v.strip() in query or re.fullmatch(r"[\d.:\-]+", v.strip()) for v in args.values())
    if mode == "verbatimtime":
        keys = [k for k in time_params(tool) if k in args]
        if not keys or not grounded:
            return False
        for k in keys:
            v = str(args[k]).strip()
            if v not in query or not re.search(r"(\d|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|삼십|십|오전|오후|아침|저녁|밤|새벽|정오|자정)", v) or re.fullmatch(r"[\d:\-T ]+", v):
                return False
            if re.search(r"[에는은이가을를]$", v) and (v[:-1] in query) and not re.search(r"(후|뒤|전|시|분|반)$", v):
                return False  # a particle glued to the expression
            tail = query[query.index(v) + len(v):]
            if re.match(r"\s?(뒤|후|전|반)", tail):
                return False  # the request continues the expression ("3시간 뒤"); the value must carry it
        return True
    if mode == "memoomit":
        mk = memo_params(tool)
        if not mk or any(k in args for k in mk) or not grounded:
            return False
        return len(args) >= 2
    if mode == "oldnew":
        pairs = oldnew_pairs(tool)
        if not pairs or not grounded:
            return False
        for a, b in pairs:
            if a in args and b in args and isinstance(args[a], str) and isinstance(args[b], str) and args[a].strip() != args[b].strip():
                return True
        return False
    if mode == "intent":
        if len(tools) < 2 or not grounded:
            return False
        is_q = bool(ASK.search(query)); is_c = bool(CMD.search(query))
        if is_q == is_c:
            return False
        want_query = is_q
        if want_query != bool(QUERY_TOOL.search(tool["name"]) and not CREATE_TOOL.search(tool["name"])) and want_query != (not CREATE_TOOL.search(tool["name"])):
            return False
        if want_query and any(re.search(r"\d{4}-\d{2}-\d{2}|\d+\s?월\s?\d+\s?일", str(v)) for v in args.values()):
            return False  # question: entity only, no date in the search term
        return True
    return False


def pick(pool, mode, n, rng):
    out = []
    if mode == "intent":
        qs = [t for t in pool if QUERY_TOOL.search(t["name"]) and not CREATE_TOOL.search(t["name"])]
        cs = [t for t in pool if CREATE_TOOL.search(t["name"]) and not QUERY_TOOL.search(t["name"])]
        # pair by shared name words so the two tools plausibly concern the same thing
        def words(name): return {w for w in re.split(r"[^a-z가-힣]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()) if len(w) > 2 and w not in {"get", "set", "add", "new", "create", "query", "search", "find", "check", "info", "list", "the", "and"}}
        idx = {}
        for t in cs:
            for w in words(t["name"]):
                idx.setdefault(w, []).append(t)
        for q in qs:
            cands = [c for w in words(q["name"]) for c in idx.get(w, [])]
            if cands:
                out.append([q, rng.choice(cands)])
        rng.shuffle(out); return out[:n]
    for t in pool:
        if mode == "verbatimtime" and TIME_TOOL.search(t["name"]) and time_params(t):
            out.append(t)
        elif mode == "memoomit" and memo_params(t) and len(props_of(t)) >= 2:
            out.append(t)
        elif mode == "oldnew" and oldnew_pairs(t):
            out.append(t)
    rng.shuffle(out)
    return out[:n]


@app.function(image=image, gpu="H100:2", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 6, memory=65536)
def synth_targeted_v13(tools: str = "sft/synth_tools_v9_all.json", dst: str = "sft/synth_v13.jsonl", per_mode: int = 6000, model_id: str = "palette-lab/palette-k-midm", seed: int = 13, modes: str = "verbatimtime,memoomit,oldnew,intent"):
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
    print(f"[v13] {len(pool)} distinct tools", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.9, dtype="bfloat16", enforce_eager=True)
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=900, stop=["<|im_end|>", "<|endoftext|>"])
    spv = SamplingParams(temperature=0.0, max_tokens=300, stop=["<|im_end|>", "<|endoftext|>"])
    log = open(f"{V}/teacher.log", "a")
    say = lambda m: (print(m, flush=True), log.write(time.strftime("%F %T ") + m + "\n"), log.flush())
    kept = []
    for mode in [m.strip() for m in modes.split(",") if m.strip()]:
        chosen = pick(pool, mode, per_mode // 3 + 1, rng)
        if len(chosen) * 3 < per_mode:
            reps = per_mode // max(1, len(chosen) * 3) + 1
            chosen = [t for t in chosen for _ in range(reps)][: per_mode // 3 + 1]
        units = [(x if isinstance(x, list) else [prepare(x, mode)]) for x in chosen]
        say(f"[v13] mode {mode}: {len(units)} tool prompts -> {len(units) * 3} candidate pairs")
        prompts = [chat(p_mode(u if len(u) > 1 else u[0], rng.sample(STYLES, 3), mode)) for u in units]
        outs = llm.generate(prompts, sp)
        cands = []
        for u, o in zip(units, outs):
            arr = first_json(o.outputs[0].text, "list") or []
            for it in arr[:3]:
                if not isinstance(it, dict) or not isinstance(it.get("query"), str) or not isinstance(it.get("call"), dict):
                    continue
                call = it["call"]
                if not isinstance(call.get("arguments"), dict) or call.get("name") not in {t["name"] for t in u}:
                    continue
                if not rule_ok(mode, u, it["query"], call):
                    continue
                cands.append({"tools": u, "query": it["query"].strip(), "call": call})
        say(f"[v13] mode {mode}: {len(cands)} pass the rule; verifying")
        vprompts = []
        for c in cands:
            distract = rng.sample(pool, 4)
            shown = list(c["tools"]) + distract; rng.shuffle(shown)
            vprompts.append(chat(p_verify(c["query"], shown)))
        vouts = llm.generate(vprompts, spv)
        n_ok = 0
        for c, o in zip(cands, vouts):
            v = first_json(o.outputs[0].text, "dict")
            # verbatimtime: rule pins the value; verify tool only. others: full agreement.
            if v and (v.get("name") == c["call"]["name"] if mode == "verbatimtime" else agree(v, c["call"])):
                kept.append({"lang": "ko", "query": c["query"], "tools": c["tools"], "call": c["call"], "cond": f"v13_{mode}"}); n_ok += 1
        say(f"[v13] mode {mode}: {n_ok} agreed and kept (total {len(kept)})")
        os.makedirs(os.path.dirname(f"{V}/{dst}"), exist_ok=True)
        with open(f"{V}/{dst}", "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        vol.commit()
    say(f"[v13] wrote {dst}: {len(kept)} rows")
    return {"rows": len(kept)}
