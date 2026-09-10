"""Post-training set v6 = v5 plus the teacher-synthesised rows (harness/teacher_synth.py): thousands of invented
tools across 60 Korean service domains, three verified Korean requests each. Distractor sets are built here:
close = tools from the same domain, random = tools from other domains, k in {1,3,4,5,6,8}; 30 percent of rows
get tool names restyled (assemble_sft_v3.restyle_row); about 5 percent become negatives (the target tool is
withheld, call = none). Benchmark guard: no benchmark tool name, no benchmark query.

    .venv/bin/python harness/assemble_sft_v6.py --v5 data/sft_train_v5_all.jsonl --synth data/synth_v6.jsonl --tools data/synth_tools_v6.json --out data/sft_train_v6_all.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

import assemble_sft_v3 as V3
from assemble_sft import bench_guard


def words(name: str):
    return {w.lower() for w in re.split(r"[_\-\s]+|(?<=[a-z])(?=[A-Z])", name) if len(w) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True); ap.add_argument("--synth", required=True); ap.add_argument("--tools", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--p-restyle", type=float, default=0.3); ap.add_argument("--none-share", type=float, default=0.05); ap.add_argument("--seed", type=int, default=6)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bench_names, bench_queries = bench_guard(); V3.FORBIDDEN = set(bench_names)
    tools = [e for e in json.load(open(a.tools, encoding="utf-8")) if e["tool"]["name"] not in bench_names]
    by_domain = collections.defaultdict(list)
    for e in tools:
        by_domain[e["domain"]].append(e["tool"])
    names = {e["tool"]["name"] for e in tools}
    synth = [json.loads(l) for l in open(a.synth, encoding="utf-8")]
    synth = [r for r in synth if r["tool"]["name"] in names and "".join(r["query"].split()) not in bench_queries]

    def close_pool(r):
        pool = [t for t in by_domain[r["domain"]] if t["name"] != r["tool"]["name"]]
        if len(pool) < 7:  # catalogue tools have no domain: nearest by shared name words, then random
            w = words(r["tool"]["name"]); pool += sorted((e["tool"] for e in tools if e["tool"]["name"] != r["tool"]["name"] and e["tool"] not in pool), key=lambda t: -len(words(t["name"]) & w))[:20]
        return pool

    rows, negs = [], []
    for r in synth:
        k = rng.choices([1, 3, 4, 5, 6, 8], weights=[18, 8, 25, 8, 9, 32])[0]
        close = k > 1 and rng.random() < 0.5
        pool = close_pool(r) if close else [e["tool"] for e in tools if e["tool"]["name"] != r["tool"]["name"]]
        others = rng.sample(pool, min(k - 1, len(pool)))
        shown = others + [r["tool"]]; rng.shuffle(shown)
        row = {"lang": "ko", "query": r["query"], "tools": shown, "call": r["call"], "cond": ("close" if close else "random") if k > 1 else "exact", "src": "synth6"}
        if rng.random() < a.p_restyle:
            row = V3.restyle_row(row, rng)
        rows.append(row)
        if rng.random() < a.none_share:  # negative: same request, target withheld
            kk = rng.choice([3, 4, 6, 8]); neg_tools = rng.sample(pool, min(kk, len(pool)))
            negs.append({"lang": "ko", "query": r["query"], "tools": neg_tools, "call": {"name": "none", "arguments": {}}, "cond": "close" if close else "random", "src": "synth6_neg"})
    v5 = [json.loads(l) for l in open(a.v5, encoding="utf-8")]
    out = v5 + rows + negs
    bad = [t["name"] for r in out for t in r["tools"] if t["name"] in bench_names]
    assert not bad, f"benchmark name collision: {set(bad)}"
    rng.shuffle(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    distinct = {t["name"] for r in out for t in r["tools"]}
    print(f"wrote {a.out}: {len(out)} rows (v5 {len(v5)}, synth {len(rows)}, negatives {len(negs)}, distinct tool names {len(distinct)}, synth tools {len(names)})")
    print("conditions", dict(collections.Counter(r.get("cond") for r in out)))


if __name__ == "__main__":
    main()
