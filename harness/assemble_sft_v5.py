"""Post-training set v5 = v4 with the no-parameter schema shape randomised.

In v4 the tools without parameters were written as a bare {} in 6,180 rows and as
{"type": "object", "properties": {}, "required": []} in only 588; FunctionChat-Bench writes 45 of its
57 no-parameter tools in the second form, and the v4 model, having learned the shape rather than the
rule, invented arguments for them. Every no-parameter tool in the set is now rendered as one of the
three common shapes at random (bare {}, {"type":"object","properties":{}}, or with "required": []),
and tools with parameters get "required": [] added when it is missing 30 percent of the time.

    .venv/bin/python harness/assemble_sft_v5.py --v4 data/sft_train_v4_all.jsonl --out data/sft_train_v5_all.jsonl
"""
import argparse, collections, json, random

SHAPES = [lambda: {}, lambda: {"type": "object", "properties": {}}, lambda: {"type": "object", "properties": {}, "required": []}]

ap = argparse.ArgumentParser(); ap.add_argument("--v4", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args(); rng = random.Random(a.seed)
n = 0; shapes = collections.Counter()
with open(a.out, "w", encoding="utf-8") as f:
    for line in open(a.v4, encoding="utf-8"):
        r = json.loads(line)
        for t in r["tools"]:
            p = t.get("parameters") or {}
            if not p.get("properties"):
                t["parameters"] = rng.choice(SHAPES)(); shapes[json.dumps(t["parameters"])] += 1
            elif "required" not in p and rng.random() < 0.3:
                t["parameters"] = dict(p, required=[])
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"wrote {a.out}: {n} rows; no-parameter shapes {dict(shapes)}")
