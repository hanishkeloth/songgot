"""Post-training set v3. Built after the first 12-layer model scored 0.2 percent on FunctionChat-Bench:
it copied made-up snake_case names perfectly and could not copy a single camelCase name, because
every tool name in v2 was snake_case. Three fixes over v2 (which this file imports):

1. Tool-name style augmentation: a share of rows get every tool renamed in a random style
   (camelCase, PascalCase, snake_case, with digit or word suffixes, verb synonyms), call name updated
   to match, so the model learns to copy the name it is shown rather than recall one it memorised.
2. glaive parsing fixed: the dataset quotes the arguments object with single quotes, so 97 percent
   of its calls failed json.loads in v1 and v2. Fixed, the English part grows from 3.5k to tens of
   thousands of rows over thousands of distinct tools.
3. "No tool" capped at about 6 percent (the benchmark has none; v2's 9.3 percent over-triggered it).

    .venv/bin/python harness/assemble_sft_v3.py --ko data/ko_raw.jsonl --out data/sft_train_v3.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from assemble_sft import TOOLS, bench_guard, schema
from assemble_sft_v2 import pick_tools

VERBS = {"get": ["fetch", "retrieve", "lookup", "query", "find", "check", "read", "show"], "set": ["create", "add", "schedule", "register", "make", "configure"],
         "send": ["dispatch", "post", "deliver", "push", "submit"], "create": ["make", "add", "new", "register", "open"], "add": ["create", "insert", "append", "register", "put"],
         "search": ["find", "lookup", "query", "browse"], "play": ["start", "stream", "run", "queue"], "book": ["reserve", "schedule", "make", "request"],
         "order": ["buy", "purchase", "request", "place"], "cancel": ["remove", "delete", "stop", "abort"], "check": ["get", "verify", "inspect", "view"],
         "compose": ["write", "draft", "create"], "start": ["begin", "launch", "run", "trigger"], "turn": ["switch", "toggle", "set"], "make": ["create", "place", "do"],
         "request": ["ask", "apply", "submit"], "translate": ["convert", "render"], "convert": ["change", "transform"], "calculate": ["compute", "estimate", "eval"],
         "record": ["log", "track", "save"], "define": ["explain", "lookup"], "read": ["fetch", "list", "show"], "list": ["show", "enumerate", "get"], "update": ["edit", "modify", "change"],
         "delete": ["remove", "erase", "drop"], "pay": ["transfer", "settle"], "transfer": ["send", "move", "wire"], "control": ["set", "adjust", "operate"], "find": ["search", "locate", "get"]}
SUFFIX_WORDS = ["Now", "Info", "Data", "Request", "Action", "Task", "Item", "Service", "Api", "Helper", "ForUser", "ByName", "V2", "Ext", "Quick"]


def restyle(name: str, rng: random.Random) -> str:
    words = [w for w in re.split(r"[_\-\s]+", name) if w]
    if not words:
        return name
    if rng.random() < 0.5 and words[0].lower() in VERBS:
        words[0] = rng.choice(VERBS[words[0].lower()])
    if rng.random() < 0.25:
        words.append(rng.choice(SUFFIX_WORDS).lower())
    if rng.random() < 0.15:
        words.insert(0, rng.choice(["api", "do", "run", "user", "my", "app"]))
    style = rng.choices(["camel", "pascal", "snake", "snake_num", "camel_num", "pascal_num"], weights=[30, 15, 25, 10, 12, 8])[0]
    if style.startswith("camel"):
        out = words[0].lower() + "".join(w[:1].upper() + w[1:].lower() for w in words[1:])
    elif style.startswith("pascal"):
        out = "".join(w[:1].upper() + w[1:].lower() for w in words)
    else:
        out = "_".join(w.lower() for w in words)
    if style.endswith("num"):
        out += (("_" if style.startswith("snake") else "") + str(rng.choice([1, 2, 3, 7, 10, 24, 2026])))
    return out


FORBIDDEN: set = set()  # benchmark tool names, set in main()


def restyle_row(row: dict, rng: random.Random) -> dict:
    mapping = {}
    used = set()
    for t in row["tools"]:
        new = restyle(t["name"], rng)
        while new in used or new in FORBIDDEN:
            new = restyle(t["name"], rng) + str(rng.randint(2, 99))
        used.add(new); mapping[t["name"]] = new
    tools = [dict(t, name=mapping[t["name"]]) for t in row["tools"]]
    call = dict(row["call"]); call["name"] = mapping.get(call["name"], call["name"])
    return dict(row, tools=tools, call=call, restyled=True)


def glaive_rows_v3(limit: int, rng: random.Random):
    out, none_out = [], []
    from datasets import load_dataset
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    idx = list(range(len(ds))); rng.shuffle(idx)
    turn_re = re.compile(r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*?)(?=<\|endoftext\|>|USER:|$)", re.S)
    fix_args = lambda s: re.sub(r"""(["'])arguments\1\s*:\s*'(.*?)'\s*}""", lambda m: '"arguments": ' + json.dumps(m.group(2)) + "}", s, flags=re.S)
    for i in idx:
        r = ds[i]
        sysm, chat = r["system"], r["chat"]
        m = re.search(r"functions.*?(\[.*\]|\{.*\})\s*$", sysm, re.S)
        funcs = []
        if m:
            try:
                j = json.loads(m.group(1)); funcs = j if isinstance(j, list) else [j]
            except json.JSONDecodeError:
                funcs = []
        tools = [{"name": f.get("name"), "description": f.get("description", ""), "parameters": f.get("parameters", {"type": "object", "properties": {}})} for f in funcs if f.get("name")]
        if not tools:
            continue
        for ti, (query, ans) in enumerate(turn_re.findall(chat)):
            query, ans = query.strip(), ans.strip()
            if not query or len(query) > 300:
                continue
            fm = re.search(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)", ans, re.S)
            if fm:
                try:
                    call = json.loads(fix_args(fm.group(1)))
                    args = call.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    call = {"name": call["name"], "arguments": args if isinstance(args, dict) else {}}
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                if not any(t["name"] == call["name"] for t in tools):
                    continue
                out.append({"lang": "en", "query": query, "tools": tools, "call": call})
            elif ti == 0 and len(ans) <= 300:
                none_out.append({"lang": "en", "query": query, "tools": tools, "call": {"name": "none", "arguments": {}}})
        if len(out) >= limit:
            break
    return out, none_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ko", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--en-limit", type=int, default=40000); ap.add_argument("--none-share", type=float, default=0.06)
    ap.add_argument("--p-restyle-ko", type=float, default=0.6); ap.add_argument("--p-restyle-en", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by_name = {t["name"]: t for t in TOOLS}
    bench_names, bench_queries = bench_guard()
    assert not (set(by_name) & bench_names)
    global FORBIDDEN; FORBIDDEN = set(bench_names)

    ko = [json.loads(l) for l in open(a.ko, encoding="utf-8")]
    rows, ko_none = [], []
    for r in ko:
        q = r["query"]
        if re.sub(r"\s+", "", q) in bench_queries:
            raise SystemExit(f"query collision with the benchmark: {q}")
        target = by_name.get(r["tool"]) if r["tool"] else None
        k = rng.choices([1, 3, 4, 5, 6, 8], weights=[22, 8, 25, 8, 7, 30])[0] if target else rng.choices([3, 4, 6, 8], weights=[2, 4, 2, 4])[0]
        close = k > 1 and rng.random() < 0.5
        picked = pick_tools(target, rng, close, k)
        row = {"lang": "ko", "query": q, "tools": [schema(t) for t in picked], "call": r["call"], "cond": ("close" if close else "random") if k > 1 else "exact"}
        if rng.random() < a.p_restyle_ko:
            row = restyle_row(row, rng)
        (ko_none if row["call"]["name"] == "none" else rows).append(row)
    en, en_none = glaive_rows_v3(a.en_limit, rng)
    en = [r for r in en if not any(t["name"] in bench_names for t in r["tools"])]
    en_none = [r for r in en_none if not any(t["name"] in bench_names for t in r["tools"])]
    for r in en:
        r["cond"] = "en"
    en = [restyle_row(r, rng) if rng.random() < a.p_restyle_en else r for r in en]
    rows.extend(en)
    # negatives: Korean first, then English, capped to the share
    n_none = int(a.none_share * len(rows) / (1 - a.none_share))
    rng.shuffle(en_none)
    negs = ko_none[:n_none] + [dict(r, cond="en") for r in en_none[: max(0, n_none - len(ko_none))]]
    rows.extend(negs)
    # final gate: no restyled or glaive name may equal a benchmark tool name
    bad = [t["name"] for r in rows for t in r["tools"] if t["name"] in bench_names]
    assert not bad, f"benchmark name collision after restyle: {set(bad)}"
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    names = {t["name"] for r in rows for t in r["tools"]}
    print(f"wrote {a.out}: {len(rows)} rows (ko {sum(r['lang']=='ko' for r in rows)}, en {sum(r['lang']=='en' for r in rows)}, none {len(negs)} = {100*len(negs)/len(rows):.1f}%, restyled {sum(bool(r.get('restyled')) for r in rows)}, distinct tool names {len(names)})")
    print("conditions", dict(collections.Counter(r["cond"] for r in rows)))


if __name__ == "__main__":
    main()
