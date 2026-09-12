"""Stage 2b for Songgot-V: teach the multiple-choice answer format the Korean benchmarks use.

The stage-2 model scored at chance on K-DTCBench and K-MMBench (0.275 / 0.269 on 2026-09-12) because it had never
seen a question followed by lettered options and a one-letter answer. This turns the 113,844 teacher-written
(page, question, answer) triples into 4-way multiple choice: the gold answer plus three answers sampled from other
pages of the same question kind, shuffled, labelled A-D, rendered exactly the way harness/songgot_v2.eval_k renders
the benchmarks. Then it continues stage 2 from ckpt/v2/sft_docqa and re-scores.

    modal deploy harness/docqa_to_mc.py
    .venv/bin/python harness/spawn.py songgot-docqa-mc run
"""
import json
import os
import random
import time

import modal

app = modal.App("songgot-docqa-mc")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={V: vol}, cpu=2, memory=8192, timeout=60 * 60 * 12)
def run(src: str = "corpora/ko_vdr_docqa.jsonl", dst: str = "corpora/ko_vdr_docqa_mc.jsonl", steps: int = 2500, seed: int = 3):
    vol.reload(); rng = random.Random(seed)
    rows = [json.loads(l) for l in open(f"{V}/{src}", encoding="utf-8")]
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r.get("kind", ""), []).append(r["answer"])
    n = 0
    with open(f"{V}/{dst}", "w", encoding="utf-8") as f:
        for r in rows:
            pool = by_kind.get(r.get("kind", ""), []) or [x["answer"] for x in rows[:5000]]
            distractors = set()
            for _ in range(40):
                d = rng.choice(pool)
                if d != r["answer"] and len(d) <= 60:
                    distractors.add(d)
                if len(distractors) == 3:
                    break
            if len(distractors) < 3:
                continue
            opts = [r["answer"]] + sorted(distractors); rng.shuffle(opts)
            letters = "ABCD"; gold = letters[opts.index(r["answer"])]
            q = r["question"] + "\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts)) + "\n정답:"
            f.write(json.dumps({"image": r["image"], "question": q, "answer": gold, "kind": "mc", "src": r.get("src")}, ensure_ascii=False) + "\n"); n += 1
    vol.commit(); print(f"[mc] {n} multiple-choice rows -> {dst}", flush=True)
    log = lambda m: (print(m, flush=True), open(f"{V}/chains.log", "a").write(time.strftime("%F %T ") + m + "\n"), vol.commit())
    log(f"[chain] mc rows {n}; stage 2b from ckpt/v2/sft_docqa")
    r = modal.Function.from_name("songgot-v2", "sft").remote(init="ckpt/v2/sft_docqa", qa=dst, steps=steps, out="v2/sft_mc", lr_lm=2e-5, lr_proj=5e-5)
    log(f"[chain] v2 stage 2b {r}; scoring")
    s = modal.Function.from_name("songgot-v2", "eval_k").remote(ckpt="ckpt/v2/sft_mc", tag="songgot_v_mc", benches="kdtcbench,kmmbench", limit=1000)
    log(f"[chain] SONGGOT-V MC SCORED {s}")
    return s
