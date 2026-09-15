"""Exact-match scorer for the MASSIVE ko held-out predictions: name accuracy, full-call accuracy (arguments compared after
whitespace normalisation, since MASSIVE spans are spoken text), and per-scenario breakdown.

    .venv/bin/python eval/score_massive.py vol/preds/q35_08b_v10_massive500.jsonl bench/massive_ko_test_500_gold.jsonl
"""
import collections
import json
import re
import sys

sys.path.insert(0, "eval")
from functionchat_exact import parse_call  # noqa: E402


def norm(v):
    return re.sub(r"\s+", "", str(v)).lower() if v is not None else None


def main(pred_path, gold_path):
    preds = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(pred_path, encoding="utf-8")}
    gold = [json.loads(l) for l in open(gold_path, encoding="utf-8")]
    n = nm = fl = 0; by = collections.defaultdict(lambda: [0, 0, 0]); fails = []
    for g in gold:
        c = parse_call(preds.get(g["id"], "")); gc = g["call"]
        n_ok = bool(c) and c["name"] == gc["name"]
        ga = {k: norm(v) for k, v in (gc.get("arguments") or {}).items()}
        pa = {k: norm(v) for k, v in ((c or {}).get("arguments") or {}).items()} if c else None
        f_ok = n_ok and pa == ga
        n += 1; nm += n_ok; fl += f_ok; s = by[g["scenario"]]; s[0] += 1; s[1] += n_ok; s[2] += f_ok
        if not f_ok and len(fails) < 15:
            fails.append({"q": g["query"], "gold": gc, "out": (preds.get(g["id"]) or "")[:140]})
    res = {"n": n, "call_acc": round(fl / n, 4), "name_acc": round(nm / n, 4),
           "by_scenario": {k: {"n": v[0], "name_acc": round(v[1] / v[0], 3), "call_acc": round(v[2] / v[0], 3)} for k, v in sorted(by.items())}, "fails_sample": fails}
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
