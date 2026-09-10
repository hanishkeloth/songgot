"""Assemble the post-training set: Korean synthetic calls (our teacher) + English glaive
function-calling v2 (Apache 2.0) + negatives, each with distractor tools, in Songgot's format.

    .venv/bin/python harness/assemble_sft.py --ko data/ko_raw.jsonl --out data/sft_train.jsonl

Disjointness gate: no example may share a tool name or a query with FunctionChat-Bench
SingleCall (test only). The gate aborts the build if it finds one.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = json.load(open(ROOT / "data" / "tools_ko.json", encoding="utf-8"))
BENCH = ROOT / "eval" / "FunctionChat-Singlecall.jsonl"


def bench_guard():
    names, queries = set(), set()
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for t in r["tools"]:
            for f in t["content"]:
                names.add(f["function"]["name"])
        for q in r["query"]:
            queries.add(re.sub(r"\s+", "", q["content"]))
    return names, queries


def schema(t):
    return {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}


def glaive_rows(path: str, limit: int, rng: random.Random):
    """glaive-function-calling-v2: system lists functions; chat has USER/ASSISTANT turns with
    <functioncall> {...} for calls. Keep the first user turn and its direct answer."""
    out = []
    from datasets import load_dataset
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    idx = list(range(len(ds))); rng.shuffle(idx)
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
        um = re.search(r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|USER:|$)", chat, re.S)
        if not um:
            continue
        query, ans = um.group(1).strip(), um.group(2).strip()
        fm = re.search(r"<functioncall>\s*(\{.*\})", ans, re.S)
        if fm:
            try:
                call = json.loads(fm.group(1).replace("'", "'"))
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                call = {"name": call["name"], "arguments": args}
            except (json.JSONDecodeError, KeyError):
                continue
            if not any(f.get("name") == call["name"] for f in funcs):
                continue
        else:
            call = {"name": "none", "arguments": {}}
            if not funcs or len(ans) > 400:
                continue
        tools = [{"name": f.get("name"), "description": f.get("description", ""), "parameters": f.get("parameters", {"type": "object", "properties": {}})} for f in funcs if f.get("name")]
        if not tools or len(query) > 300:
            continue
        out.append({"lang": "en", "query": query, "tools": tools, "call": call})
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ko", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--en-limit", type=int, default=40000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by_name = {t["name"]: t for t in TOOLS}
    bench_names, bench_queries = bench_guard()
    assert not (set(by_name) & bench_names), f"tool-name collision with the benchmark: {set(by_name) & bench_names}"

    ko = [json.loads(l) for l in open(a.ko, encoding="utf-8")]
    rows = []
    for r in ko:
        q = r["query"]
        if re.sub(r"\s+", "", q) in bench_queries:
            raise SystemExit(f"query collision with the benchmark: {q}")
        target = by_name.get(r["tool"]) if r["tool"] else None
        k = rng.choice([1, 3, 4, 5, 8]) if target else rng.choice([3, 4, 6, 8])
        pool = [t for t in TOOLS if not target or t["name"] != target["name"]]
        picked = rng.sample(pool, k - 1 if target else k)
        if target:
            picked.append(target)
        rng.shuffle(picked)
        rows.append({"lang": "ko", "query": q, "tools": [schema(t) for t in picked], "call": r["call"]})
    n_ko = len(rows)
    en = glaive_rows("glaiveai/glaive-function-calling-v2", a.en_limit, rng)
    for r in en:
        if any(t["name"] in bench_names for t in r["tools"]):
            continue
        rows.append(r)
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_none = sum(1 for r in rows if r["call"]["name"] == "none")
    print(f"wrote {a.out}: {len(rows)} rows (ko {n_ko}, en {len(rows)-n_ko}, none {n_none})")


if __name__ == "__main__":
    main()
