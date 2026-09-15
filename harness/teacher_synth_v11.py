"""Targeted synthesis round v11: the two data gaps that the v10 Songgot-X exposed on FunctionChat SingleCall
(eval/score_q35_08b_v10.json, paper section 6 "Targeted round v10").

  payload    the request is "instruction + delimiter + payload" (이 문장 몇 단어인지 세줘: ..., 다음 내용 메모해줘\\n...,
             "..." 라고 보내줘) and the text argument carries the payload only, verbatim, never the instruction. v10 had no
             such row, and the v10 model copied the instruction prefix into count_words (19 items lost against v8).
  year_hist  the request states a four-digit year between 1950 and 2015 (생일, 기념일, 창립일, 디데이) and the call keeps
             it. No row of the v10 set carries a 19xx year, and the v10 model wrote 2026 for "1989년 7월 22일".

Same teacher (palette-lab/palette-k-midm), same deterministic-rule-then-verify recipe as teacher_synth_v10.py; rows
are tagged cond v11_<mode> and assembled with harness/assemble_sft_v10.py --synth2.

    modal deploy harness/teacher_synth_v11.py
    .venv/bin/python harness/spawn.py songgot-teacher-v11 synth_targeted_v11 dst=sft/synth_v11.jsonl per_mode=8000
"""
import json
import os
import random
import re
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_synth import agree, first_json, image as base_image, p_verify, valid_tool  # noqa: E402
from teacher_synth_v10 import CACHE, STYLES, V, hf_cache, pick_tools as pick_tools_v10, vol  # noqa: E402

image = base_image.add_local_python_source("teacher_synth", "teacher_synth_v10")
app = modal.App("songgot-teacher-v11")

MODE_RULES = {
    "payload": ("요청은 반드시 '지시문 + 구분자 + 본문' 형태로 쓰세요. 지시문은 도구가 할 일(예: 이 문장 몇 단어인지 세줘, 다음 내용 메모해줘, "
                "아래 글 영어로 번역해줘, 이 메시지 엄마한테 보내줘, 이거 요약해줘)이고, 구분자는 콜론(:), 줄바꿈, 큰따옴표, '다음 내용', '아래 문장' 중 하나입니다. "
                "본문은 실제 한국어 문장 한두 개(뉴스 한 줄, 메모 내용, 메시지 본문, 일기 한 줄)로, 8글자 이상이어야 합니다. 세 요청 중 하나는 지시문이 본문 뒤에 오게 쓰세요"
                "(예: \"내일 회의 10시로 변경\" 이거 메모해줘). 호출의 본문 인자(text, content, message, memo, body, sentence 등)에는 지시문과 구분자를 빼고 "
                "본문만 글자 그대로(띄어쓰기, 문장부호까지) 넣습니다. 지시문의 단어가 인자 값에 들어가면 틀린 것입니다."),
    "year_hist": ("요청의 날짜에는 1950년부터 2015년 사이의 네 자리 연도를 반드시 명시하세요(예: 1989년 7월 22일, 2003년 3월 1일, 1975년 12월 25일). "
                  "생일, 결혼기념일, 창립일, 입사일, 디데이 계산, 옛 기록 조회처럼 과거 연도가 자연스러운 요청으로 쓰세요. 세 요청은 서로 다른 연도를 쓰세요. "
                  "호출에서는 요청에 적힌 연도를 그대로 ISO 날짜(1989-07-22)로 씁니다. 절대 다른 연도로 바꾸거나 올해로 옮기지 마세요."),
}

TEXT_PARAM = re.compile(r"(text|content|message|memo|body|sentence|note|msg|comment|description|quote|caption|input|paragraph|본문|내용|문장|메시지)", re.I)
INSTRUCTION_TAIL = re.compile(r"(세\s?줘|해\s?줘|주세요|줄래|주실래|보내|저장|기록|등록|번역|요약|바꿔|메모해|추가해|올려|읽어)\s*[.!?~]*$")
DELIM = re.compile(r"[:：\n\"“”'‘’「」『』]|다음\s?(내용|문장|글|텍스트|메시지)|아래\s?(내용|문장|글|텍스트|메시지)|이\s?(문장|내용|글|메시지|텍스트|거)")


def p_mode(tool, styles, mode):
    return ("아래 도구 스키마를 보고, 이 도구를 호출하게 만드는 서로 다른 한국어 사용자 요청 3개와 각 요청에 정확히 대응하는 호출 JSON을 만드세요.\n"
            f"요청 1의 말투: {styles[0]}. 요청 2의 말투: {styles[1]}. 요청 3의 말투: {styles[2]}.\n"
            f"이번 규칙: {MODE_RULES[mode]}\n"
            "공통 규칙: required 속성은 반드시 채우고, 요청에 없는 값을 지어내지 마세요. 요청에 있는 값은 빠짐없이 인자에 넣으세요.\n"
            "출력: JSON 배열만. 각 항목은 {\"query\": \"...\", \"call\": {\"name\": \"...\", \"arguments\": {...}}} 형식.\n\n"
            f"도구: {json.dumps(tool, ensure_ascii=False)}")


def rule_ok(mode: str, query: str, call: dict) -> bool:
    args = call.get("arguments") or {}
    if not args:
        return False
    if mode == "payload":
        strs = [(k, v) for k, v in args.items() if isinstance(v, str) and len(v.strip()) >= 8]
        if not strs:
            return False
        k, v = max(strs, key=lambda kv: len(kv[1]))
        v = v.strip()
        if v not in query or v == query.strip():
            return False
        i = query.index(v); prefix, suffix = query[:i], query[i + len(v):]
        if len(prefix.strip()) < 3 and len(suffix.strip()) < 3:
            return False  # nothing outside the payload: no instruction to separate from
        if not DELIM.search(prefix + " " + suffix):
            return False
        if INSTRUCTION_TAIL.search(v) or DELIM.match(v[-1]):
            return False  # the payload swallowed the instruction or a closing delimiter
        if v[0] in ":：\"“'‘「『" or v[-1] in "\"”'’」』":
            return False
        return all((not isinstance(o, str)) or o in query for kk, o in args.items() if kk != k)
    if mode == "year_hist":
        dates = [v for v in args.values() if isinstance(v, str) and re.search(r"(19[5-9]\d|20[01]\d)-\d{2}-\d{2}", v)]
        if not dates:
            return False
        for d in dates:
            y = re.search(r"(19[5-9]\d|20[01]\d)-\d{2}-\d{2}", d).group(1)
            if not (1950 <= int(y) <= 2015) or not re.search(rf"(^|[^\d]){y}\s?년", query):
                return False
        # no other four-digit year in the query may have been dropped or rewritten
        years_q = set(re.findall(r"(?<!\d)((?:19|20)\d\d)\s?년", query))
        years_c = {re.search(r"((?:19|20)\d\d)-\d{2}-\d{2}", d).group(1) for d in dates}
        return years_c <= years_q
    return False


def pick_tools(tools, mode, n, rng):
    if mode == "year_hist":
        return pick_tools_v10(tools, "year", n, rng)
    out = []
    for t in tools:
        p = t.get("parameters") or {}
        props = p.get("properties") if isinstance(p.get("properties"), dict) else {k: v for k, v in p.items() if isinstance(v, dict) and "type" in v}
        if any(str((v or {}).get("type", "")) == "string" and (TEXT_PARAM.search(k) or TEXT_PARAM.search(str((v or {}).get("description", "") or ""))) for k, v in (props or {}).items()):
            out.append(t)
    rng.shuffle(out)
    return out[:n]


@app.function(image=image, gpu="H100:2", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 6, memory=65536)
def synth_targeted_v11(tools: str = "sft/synth_tools_v9_all.json", dst: str = "sft/synth_v11.jsonl", per_mode: int = 8000, model_id: str = "palette-lab/palette-k-midm", seed: int = 11):
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
    print(f"[v11] {len(pool)} distinct tools from {tools}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.9, dtype="bfloat16", enforce_eager=True)
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=900, stop=["<|im_end|>", "<|endoftext|>"])
    spv = SamplingParams(temperature=0.0, max_tokens=300, stop=["<|im_end|>", "<|endoftext|>"])
    log = open(f"{V}/teacher.log", "a")
    say = lambda m: (print(m, flush=True), log.write(time.strftime("%F %T ") + m + "\n"), log.flush())
    kept = []
    for mode in ("payload", "year_hist"):
        chosen = pick_tools(pool, mode, per_mode // 3 + 1, rng)
        if len(chosen) * 3 < per_mode:  # few eligible tools: ask each tool more than once with different styles
            reps = per_mode // max(1, len(chosen) * 3) + 1
            chosen = [t for t in chosen for _ in range(reps)][: per_mode // 3 + 1]
        say(f"[v11] mode {mode}: {len(chosen)} tool prompts -> {len(chosen) * 3} candidate pairs")
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
        say(f"[v11] mode {mode}: {len(cands)} pass the rule; verifying")
        vprompts = []
        for c in cands:
            distract = rng.sample(pool, 4)
            tools_shown = [c["tool"]] + distract; rng.shuffle(tools_shown)
            vprompts.append(chat(p_verify(c["query"], tools_shown)))
        vouts = llm.generate(vprompts, spv)
        n_ok = 0
        # payload: the verifier must reproduce the exact call (the payload boundary is the point). year_hist: the rule has
        # already pinned the year to the one written in the query, so only the tool choice is verified (the teacher itself
        # tends to rewrite years, which is the error being corrected).
        for c, o in zip(cands, vouts):
            v = first_json(o.outputs[0].text, "dict")
            if v and (agree(v, c["call"]) if mode == "payload" else v.get("name") == c["call"]["name"]):
                kept.append({"lang": "ko", "query": c["query"], "tools": [c["tool"]], "call": c["call"], "cond": f"v11_{mode}"}); n_ok += 1
        say(f"[v11] mode {mode}: {n_ok} agreed and kept (total {len(kept)})")
        os.makedirs(os.path.dirname(f"{V}/{dst}"), exist_ok=True)
        with open(f"{V}/{dst}", "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        vol.commit()
    say(f"[v11] wrote {dst}: {len(kept)} rows")
    return {"rows": len(kept)}
