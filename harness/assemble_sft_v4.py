"""Post-training set v4 = v3 plus two additions aimed at the v3 model's measured failures:

1. Korean queries over thousands of tools: glaive requests rewritten in Korean by our own
   Palette-K-Midm (harness/teacher_translate.py), schemas and gold calls unchanged. v3's Korean rows
   covered 57 tools; the benchmark's 25 are unseen and mostly generic.
2. Tools without parameters: the v3 model invented arguments for a tool whose schema has none.
   Rows whose tool has no properties are upweighted (x4) and a share of them restyled, so that
   "empty schema -> empty arguments" is seen often enough to be learned.

    .venv/bin/python harness/assemble_sft_v4.py --v3 data/sft_train_v3_all.jsonl --trans data/ko_trans.jsonl --out data/sft_train_v4_all.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random

import assemble_sft_v3 as V3
from assemble_sft import bench_guard


def props(t):
    return (t.get("parameters") or {}).get("properties") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", required=True); ap.add_argument("--trans", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--p-restyle-trans", type=float, default=0.4); ap.add_argument("--noarg-weight", type=int, default=4); ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bench_names, bench_queries = bench_guard()
    V3.FORBIDDEN = set(bench_names)
    rows = [json.loads(l) for l in open(a.v3, encoding="utf-8")]
    trans = [json.loads(l) for l in open(a.trans, encoding="utf-8")] if a.trans and __import__("os").path.exists(a.trans) else []
    trans = [r for r in trans if not any(t["name"] in bench_names for t in r["tools"]) and "".join(r["query"].split()) not in bench_queries]
    trans = [V3.restyle_row(r, rng) if rng.random() < a.p_restyle_trans else r for r in trans]
    noarg = [r for r in rows if r["call"]["name"] != "none" and not r["call"]["arguments"] and not props(next(t for t in r["tools"] if t["name"] == r["call"]["name"]))]
    noarg_trans = [r for r in trans if r["call"]["name"] != "none" and not r["call"]["arguments"] and not props(next(t for t in r["tools"] if t["name"] == r["call"]["name"]))]
    extra = []
    for r in noarg + noarg_trans:
        for _ in range(a.noarg_weight - 1):
            extra.append(V3.restyle_row(r, rng) if rng.random() < 0.5 else dict(r))
    out = rows + trans + extra
    bad = [t["name"] for r in out for t in r["tools"] if t["name"] in bench_names]
    assert not bad, f"benchmark name collision: {set(bad)}"
    rng.shuffle(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_none = sum(r["call"]["name"] == "none" for r in out)
    print(f"wrote {a.out}: {len(out)} rows (v3 {len(rows)}, translated {len(trans)}, no-arg extra {len(extra)}, none {100*n_none/len(out):.1f}%, distinct names {len({t['name'] for r in out for t in r['tools']})})")
    print("conditions", dict(collections.Counter(r.get("cond") for r in out)))


if __name__ == "__main__":
    main()
