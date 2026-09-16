"""Hard RL pool: the prompts of an unseen pool sample that a checkpoint gets wrong under greedy decoding. GRPO gives no
gradient on prompts whose sampled group all agrees (2026-09-16: round 1 on the plain unseen pool had 0-3 groups with
variance per step), so the pool is filtered to misses first.

    .venv/bin/python harness/make_hard_pool.py --preds vol/preds/soup3_pool6k_fast.jsonl --gold bench/rl_pool_v12_sample6k_gold.jsonl --out data/rl_pool_v12_hard.jsonl
"""
import argparse
import collections
import json
import random
import sys

sys.path.insert(0, "eval")
import functionchat_exact as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--preds", required=True); ap.add_argument("--gold", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    preds = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(a.preds, encoding="utf-8")}
    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    def norm_args(x):
        x = json.loads(x) if isinstance(x, str) and x.strip() else (x or {})
        return F._norm(x)
    cats = collections.Counter(); hard = []
    for g in gold:
        c = F.parse_call(preds.get(g["id"], "")); gc = g["call"]
        if c is None: k = "unparseable"
        elif c["name"] != gc["name"]: k = "wrong_name"
        elif c.get("arguments") is None or F._norm(c["arguments"]) != norm_args(gc.get("arguments")): k = "arg_mismatch"
        else: k = "ok"
        cats[k] += 1
        if k != "ok":
            hard.append({"query": g["query"], "tools": g["tools"], "call": gc, "cond": g.get("cond"), "miss": k})
    random.Random(a.seed).shuffle(hard)
    with open(a.out, "w", encoding="utf-8") as f:
        for h in hard:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print("checkpoint on the pool sample:", dict(cats), f"-> {len(hard)} hard prompts -> {a.out}")
    print("hard by cond:", collections.Counter(h["cond"] for h in hard).most_common(), "| by miss:", dict(collections.Counter(h["miss"] for h in hard)))


if __name__ == "__main__":
    main()
