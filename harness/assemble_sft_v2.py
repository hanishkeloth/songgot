"""Post-training set v2: same sources as v1 (template Korean calls, glaive English, Apache 2.0),
rebuilt to mirror the conditions a deployed tool caller meets:

- tool counts follow the benchmark's 1 / 4 / 8 (plus a little 3, 5, 6) instead of mostly 1;
- half of the multi-tool Korean prompts carry CLOSE distractors (same category as the target),
  the other half random ones, so the model learns to discriminate between look-alike tools;
- "no tool applies" is kept but capped at about 10 percent of rows (v1 had 28 percent, which
  teaches refusal more than calling); every Korean negative gets a close-looking tool set too;
- tool order is shuffled and the target is never at a fixed position.

Disjointness gate against FunctionChat-Bench SingleCall is unchanged (names and queries).

    .venv/bin/python harness/assemble_sft_v2.py --ko data/ko_raw.jsonl --out data/sft_train_v2.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from assemble_sft import BENCH, ROOT, TOOLS, bench_guard, schema


def glaive_rows_v2(limit: int, rng: random.Random):
    """glaive-function-calling-v2: keep EVERY (user turn -> immediate <functioncall>) pair in a chat,
    not only the first turn (v1 kept the first turn, which is mostly small talk, so 95 percent of its
    English rows were negatives). Negatives are taken only from single-turn chats with short answers."""
    out, none_out = [], []
    from datasets import load_dataset
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    idx = list(range(len(ds))); rng.shuffle(idx)
    turn_re = re.compile(r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*?)(?=<\|endoftext\|>|USER:|$)", re.S)
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
        pairs = turn_re.findall(chat)
        for ti, (query, ans) in enumerate(pairs):
            query, ans = query.strip(), ans.strip()
            if not query or len(query) > 300:
                continue
            fm = re.search(r"<functioncall>\s*(\{.*\})", ans, re.S)
            if fm:
                try:
                    call = json.loads(fm.group(1).replace("'", "'"))
                    args = call.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    call = {"name": call["name"], "arguments": args}
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

BY_CAT = collections.defaultdict(list)
for t in TOOLS:
    BY_CAT[t["category"]].append(t)


def pick_tools(target, rng, close: bool, k: int):
    """k tools including the target (if any). close=True fills from the target's category first."""
    others = [t for t in TOOLS if not target or t["name"] != target["name"]]
    chosen = []
    if close and target:
        same = [t for t in BY_CAT[target["category"]] if t["name"] != target["name"]]
        rng.shuffle(same)
        chosen.extend(same[: k - 1])
    if close and not target:
        cat = rng.choice(list(BY_CAT))
        same = list(BY_CAT[cat]); rng.shuffle(same); chosen.extend(same[:k])
    rest = [t for t in others if t not in chosen]
    rng.shuffle(rest)
    need = (k - 1 if target else k) - len(chosen)
    chosen.extend(rest[: max(0, need)])
    if target:
        chosen.append(target)
    rng.shuffle(chosen)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ko", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--en-limit", type=int, default=40000); ap.add_argument("--en-none-max", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by_name = {t["name"]: t for t in TOOLS}
    bench_names, bench_queries = bench_guard()
    assert not (set(by_name) & bench_names), f"tool-name collision with the benchmark: {set(by_name) & bench_names}"

    ko = [json.loads(l) for l in open(a.ko, encoding="utf-8")]
    rows = []
    close_n = 0
    for r in ko:
        q = r["query"]
        if re.sub(r"\s+", "", q) in bench_queries:
            raise SystemExit(f"query collision with the benchmark: {q}")
        target = by_name.get(r["tool"]) if r["tool"] else None
        k = rng.choices([1, 3, 4, 5, 6, 8], weights=[22, 8, 25, 8, 7, 30])[0] if target else rng.choices([3, 4, 6, 8], weights=[2, 4, 2, 4])[0]
        close = k > 1 and rng.random() < 0.5
        close_n += close
        picked = pick_tools(target, rng, close, k)
        rows.append({"lang": "ko", "query": q, "tools": [schema(t) for t in picked], "call": r["call"], "cond": ("close" if close else "random") if k > 1 else "exact"})
    n_ko = len(rows)
    en, en_none_rows = glaive_rows_v2(a.en_limit, rng)
    rng.shuffle(en_none_rows)
    for r in en + en_none_rows[: a.en_none_max]:
        if any(t["name"] in bench_names for t in r["tools"]):
            continue
        r["cond"] = "en"
        rows.append(r)
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_none = sum(1 for r in rows if r["call"]["name"] == "none")
    conds = collections.Counter(r["cond"] for r in rows)
    counts = collections.Counter(len(r["tools"]) for r in rows)
    print(f"wrote {a.out}: {len(rows)} rows (ko {n_ko}, en {len(rows)-n_ko}, none {n_none} = {100*n_none/len(rows):.1f}%)")
    print("conditions", dict(conds)); print("tools per prompt", sorted(counts.items()))


if __name__ == "__main__":
    main()
