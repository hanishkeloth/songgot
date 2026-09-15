"""RL prompt pool for the line-B GRPO stage: v8 rows that entered NEITHER the v10 nor the v11 SFT sample.
Memory rule from the from-scratch runs (2026-09-10): on prompts the policy has memorised the group reward has
no variance and GRPO gives nothing; the RL prompts must be disjoint from SFT. assemble_sft_v10.py draws its 175k
v8 rows with random.Random(seed).shuffle over the bench-guarded list, seed 10 for v10 and 11 for v11, so the
complement is reproducible here. Distractors are added the same way (1, 4 or 8; name-related or random).

    .venv/bin/python harness/make_rl_pool.py --v8 data/sft_train_v8_all.jsonl --out data/rl_pool_v11.jsonl --n 24000
"""
import argparse
import collections
import json
import random
import re
import sys

sys.path.insert(0, "harness")
from assemble_sft import bench_guard  # noqa: E402


def words(name):
    return {w for w in re.split(r"[^a-z0-9가-힣]+", name.lower().replace("_", " ")) if len(w) > 2}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v8", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24000); ap.add_argument("--sample", type=int, default=175000); ap.add_argument("--seeds", default="10,11"); ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    bench_names, bench_queries = bench_guard()
    rows = [json.loads(l) for l in open(a.v8, encoding="utf-8")]
    rows = [r for r in rows if r.get("query") and r.get("tools") and r.get("call") and not any(t["name"] in bench_names for t in r["tools"]) and "".join(r["query"].split()) not in bench_queries]
    used = set()
    for s in [int(x) for x in a.seeds.split(",")]:
        v = list(rows); random.Random(s).shuffle(v); used |= {id(r) for r in v[: a.sample]}
    rest = [r for r in rows if id(r) not in used]
    rng = random.Random(a.seed); rng.shuffle(rest)
    pool = {}
    for r in rows:
        for t in r["tools"]:
            pool.setdefault(t["name"], t)
    pool_list = list(pool.values()); by_word = collections.defaultdict(list)
    for t in pool_list:
        for w in words(t["name"]):
            by_word[w].append(t)
    out = []
    for r in rest[: a.n]:
        target = next((t for t in r["tools"] if t["name"] == r["call"]["name"]), r["tools"][0])
        k = rng.choice([1, 4, 8]); close = rng.random() < 0.5; cands = []
        if k > 1 and close:
            seen = {target["name"]}
            for w in words(target["name"]):
                for t in by_word.get(w, []):
                    if t["name"] not in seen:
                        seen.add(t["name"]); cands.append(t)
            rng.shuffle(cands)
        if k > 1 and len(cands) < k - 1:
            cands += [t for t in rng.sample(pool_list, 40) if t["name"] != target["name"] and all(t["name"] != c["name"] for c in cands)]
        tools = [target] + cands[: k - 1]; rng.shuffle(tools)
        out.append({"query": r["query"], "tools": tools, "call": r["call"], "cond": ("exact" if k == 1 else f"{k}_{'close' if close else 'random'}")})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"v8 guarded rows {len(rows)}, used by SFT samples {len(used)}, unseen {len(rest)}, wrote {len(out)} -> {a.out}")
    print("conditions", collections.Counter(r["cond"] for r in out).most_common())


if __name__ == "__main__":
    main()
