"""Post-training set v10 (and v11 with --synth2 for the second targeted round, --massive-weight 1, --none-share 0.02) = v8 sample + the targeted synthetic round (copy fidelity, from/to roles, Korean number words,
two-digit years; harness/teacher_synth_v10.py) + human Korean voice-assistant calls (MASSIVE ko-KR train split,
harness/massive_to_calls.py) + KoSGD. Single-tool rows get distractor tools so the conditions mirror the benchmark
(exact: 1 tool, close: 4 or 8 with name-related siblings, random: 4 or 8 unrelated); 30 percent are restyled and 5
percent become negatives (target withheld, call "none"), as in v6. Benchmark tool names and queries are excluded.

    .venv/bin/python harness/assemble_sft_v10.py --v8 data/sft_train_v8_all.jsonl --synth data/synth_v10.jsonl \
        --massive data/massive_ko_calls.jsonl --kosgd data/kosgd_calls.jsonl --out data/sft_train_v10.jsonl --v8-sample 175000
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
    return {w for w in re.split(r"[^a-z0-9가-힣]+", name.lower().replace("_", " ")) if len(w) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True); ap.add_argument("--synth", default=""); ap.add_argument("--synth2", default="", help="second targeted round (v11), own weight"); ap.add_argument("--synth3", default="", help="third targeted round (v12), own weight"); ap.add_argument("--massive", default=""); ap.add_argument("--kosgd", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--v8-sample", type=int, default=175000)
    ap.add_argument("--synth-weight", type=int, default=2); ap.add_argument("--synth2-weight", type=int, default=2); ap.add_argument("--synth3-weight", type=int, default=2); ap.add_argument("--massive-weight", type=int, default=2)
    ap.add_argument("--p-restyle", type=float, default=0.3); ap.add_argument("--none-share", type=float, default=0.05); ap.add_argument("--seed", type=int, default=10)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bench_names, bench_queries = bench_guard(); V3.FORBIDDEN = set(bench_names)

    def load(path, split=None):
        if not path:
            return []
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if split:
            rows = [r for r in rows if r.get("split") == split]
        return [r for r in rows if r.get("query") and r.get("tools") and r.get("call") and not any(t["name"] in bench_names for t in r["tools"]) and "".join(r["query"].split()) not in bench_queries]

    v8 = load(a.v8); synth = load(a.synth); synth2 = load(a.synth2); synth3 = load(a.synth3); massive = load(a.massive, split="train"); kosgd = load(a.kosgd)
    rng.shuffle(v8); v8 = v8[: a.v8_sample]
    pool = {}
    for r in v8:
        for t in r["tools"]:
            pool.setdefault(t["name"], t)
    pool_list = list(pool.values()); by_word = collections.defaultdict(list)
    for t in pool_list:
        for w in words(t["name"]):
            by_word[w].append(t)

    def with_distractors(r):
        target = r["tools"][0]; k = rng.choice([1, 4, 8])
        if k == 1:
            return dict(r, tools=[target], cond=(r.get("cond") or "v10") + "_exact")
        close = rng.random() < 0.5
        cands = []
        if close:
            seen = {target["name"]}
            for w in words(target["name"]):
                for t in by_word.get(w, []):
                    if t["name"] not in seen:
                        seen.add(t["name"]); cands.append(t)
            rng.shuffle(cands)
        if len(cands) < k - 1:
            extra = [t for t in rng.sample(pool_list, min(len(pool_list), 40)) if t["name"] != target["name"] and all(t["name"] != c["name"] for c in cands)]
            cands += extra
        tools = [target] + cands[: k - 1]; rng.shuffle(tools)
        return dict(r, tools=tools, cond=(r.get("cond") or "v10") + ("_close" if close else "_random"))

    extra_rows = []
    for src, weight in ((synth, a.synth_weight), (synth2, a.synth2_weight), (synth3, a.synth3_weight), (massive, a.massive_weight), (kosgd, 1)):
        for r in src:
            for _ in range(weight):
                row = with_distractors(r)
                if rng.random() < a.p_restyle:
                    row = V3.restyle_row(row, rng)
                if rng.random() < a.none_share and len(row["tools"]) > 1:
                    row = dict(row, tools=[t for t in row["tools"] if t["name"] != row["call"]["name"]], call={"name": "none", "arguments": {}}, cond=row["cond"] + "_none")
                extra_rows.append(row)
    out = v8 + extra_rows
    bad = [t["name"] for r in out for t in r["tools"] if t["name"] in bench_names]
    assert not bad, f"benchmark name collision: {set(bad)}"
    rng.shuffle(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    conds = collections.Counter(r.get("cond") for r in out)
    print(f"wrote {a.out}: {len(out)} rows (v8 {len(v8)}, synth {len(synth)}x{a.synth_weight}, synth2 {len(synth2)}x{a.synth2_weight}, synth3 {len(synth3)}x{a.synth3_weight}, massive {len(massive)}x{a.massive_weight}, kosgd {len(kosgd)}, distinct names {len({t['name'] for r in out for t in r['tools']})})")
    print("conditions", conds.most_common(12))


if __name__ == "__main__":
    main()
