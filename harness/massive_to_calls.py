"""Human Korean voice-assistant requests as tool calls: Amazon MASSIVE 1.1 ko-KR (CC BY 4.0, 16,520 utterances with
intent and inline slot annotation) mapped onto the 55 function declarations that AmazonScience/massive-agents wrote
for the same intents (intent alarm_set -> function alarm.set, slot names = parameter names, values = the annotated
spans verbatim). Intents without a declaration (general_quirky, general_greet, ...) are dropped.

    .venv/bin/python harness/massive_to_calls.py --tar <amazon-massive-dataset-1.1.tar.gz> --out data/massive_ko_calls.jsonl

The result keeps MASSIVE's partitions: train (10,861) for training, dev (1,916), test (2,792) as a human held-out set.
Values are the spoken spans ("오전 다섯 시", "이번 주"), so this set teaches copying what the user said, not
normalising it; the benchmark's normalised conventions come from the synthetic sets.
"""
import argparse
import collections
import json
import re
import tarfile

from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tar", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with tarfile.open(a.tar) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("ko-KR.jsonl"))
        rows = [json.loads(l) for l in tf.extractfile(member).read().decode("utf-8").splitlines() if l.strip()]
    p = hf_hub_download("AmazonScience/massive-agents", "massive-full-converted-all-langs-with-id/ko-KR/input.jsonl", repo_type="dataset")
    funcs = json.loads(open(p, encoding="utf-8").readline())["function"]; fn = {f["name"]: f for f in funcs}
    out, miss, part = [], collections.Counter(), collections.Counter()
    for r in rows:
        name = r["intent"].replace("_", ".", 1)
        if name not in fn:
            miss[r["intent"]] += 1; continue
        props = (fn[name].get("parameters") or {}).get("properties") or {}
        args = {slot: val.strip() for slot, val in re.findall(r"\[([a-z_]+) : ([^\]]+)\]", r["annot_utt"]) if slot in props}
        out.append({"id": f"massive-ko-{r['id']}", "split": r["partition"], "query": r["utt"], "tools": [fn[name]], "call": {"name": name, "arguments": args},
                    "scenario": r["scenario"], "source": "AmazonScience/MASSIVE 1.1 ko-KR (CC BY 4.0) + massive-agents declarations"})
        part[r["partition"]] += 1
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: {len(out)} rows {dict(part)}; unmapped intents {miss.most_common(5)}")


if __name__ == "__main__":
    main()
