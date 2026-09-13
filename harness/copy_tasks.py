"""Copy-task bucket for pretraining: verbatim span copying from Korean context, rendered as a tool call.

Failure analysis of the published 12-layer model (2026-09-12): 203 of 500 misses pick the right tool and the right
keys but garble the VALUE, mostly a near-miss copy of a span in the query (kept particle, dropped syllable). This
bucket trains exactly that skill during pretraining: a paragraph, one sentence with a span masked, and the call
{"name":"fill","arguments":{"value":<span>}} where the span is copied verbatim, particle stripped when the mask
covers a particle-bearing word. Sources: Korean Wikipedia raw text and the user's own documents (both on the volume).

    modal deploy harness/copy_tasks.py
    .venv/bin/python harness/spawn.py songgot-copy run rows=300000
"""
import json
import os
import random
import re
import time

import modal

app = modal.App("songgot-copy")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11")
PARTICLES = ("으로부터", "에서부터", "이라고", "라고", "에서", "에게", "한테", "까지", "부터", "으로", "로", "의", "에", "은", "는", "이", "가", "을", "를", "와", "과", "도", "만")
TOOL = {"name": "fill", "description": "빈칸(___)에 들어갈 값을 본문에서 그대로 찾아 적습니다.", "parameters": {"type": "object", "properties": {"value": {"type": "string", "description": "본문에 적힌 표현 그대로"}}, "required": ["value"]}}
HANGUL = re.compile(r"[가-힣]")


def strip_particle(w):
    for p in PARTICLES:
        if w.endswith(p) and len(w) > len(p) + 1 and HANGUL.search(w[:-len(p)]):
            return w[:-len(p)], p
    return w, ""


def paragraphs(path, sep):
    buf = open(path, encoding="utf-8", errors="ignore").read()
    for para in buf.split(sep):
        para = re.sub(r"\s+", " ", para).strip()
        if 200 <= len(para) <= 700 and len(HANGUL.findall(para)) / len(para) > 0.3:
            yield para


def make_row(para, rng):
    sents = [s for s in re.split(r"(?<=[.!?다요])\s+", para) if 15 <= len(s) <= 160]
    if len(sents) < 2:
        return None
    si = rng.randrange(len(sents)); sent = sents[si]; words = sent.split()
    if len(words) < 4:
        return None
    n = rng.choice([1, 1, 2, 2, 3])
    i = rng.randrange(0, len(words) - n + 1); span_words = words[i:i + n]
    last, particle = strip_particle(span_words[-1])
    if not HANGUL.search(last) and not re.search(r"\d", last):
        return None
    span = " ".join(span_words[:-1] + [last]).strip("()[]\"'“”‘’,.")
    if len(span) < 2 or len(span) > 40:
        return None
    masked = " ".join(words[:i] + ["___" + particle] + words[i + n:])
    ctx = " ".join(sents[:si] + sents[si + 1:])
    query = f"{ctx}\n\n빈칸을 채우세요: {masked}"
    return {"lang": "ko", "query": query, "tools": [TOOL], "call": {"name": "fill", "arguments": {"value": span}}, "cond": "copy"}


@app.function(image=image, volumes={V: vol}, cpu=4, memory=32768, timeout=60 * 60 * 6)
def run(rows: int = 300000, seed: int = 5, then_pretrain: bool = True):
    vol.reload(); rng = random.Random(seed); t0 = time.time()
    srcs = [(f"{V}/raw/ko.txt", "\n\n")] + [(f"{V}/private/private_ko/{f}", "\n\x1e\n") for f in sorted(os.listdir(f"{V}/private/private_ko"))] if os.path.isdir(f"{V}/private/private_ko") else [(f"{V}/raw/ko.txt", "\n\n")]
    out = f"{V}/sft/copy_tasks.jsonl"; n = 0
    with open(out, "w", encoding="utf-8") as fo:
        for path, sep in srcs:
            if not os.path.exists(path):
                continue
            for para in paragraphs(path, sep):
                r = make_row(para, rng)
                if r:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
                    if n >= rows:
                        break
            if n >= rows:
                break
    print(f"[copy] {n} rows in {time.time()-t0:.0f}s", flush=True)
    # combined instruction stream: v8 rows + copy rows, shuffled
    comb = f"{V}/sft/train_v8_copy.jsonl"; lines = open(f"{V}/sft/train_v8.jsonl", encoding="utf-8").readlines() + open(out, encoding="utf-8").readlines()
    rng.shuffle(lines); open(comb, "w", encoding="utf-8").writelines(lines); vol.commit()
    print(f"[copy] combined {len(lines)} rows -> {comb}", flush=True)
    log = lambda m: (print(m, flush=True), open(f"{V}/chains.log", "a").write(time.strftime("%F %T ") + m + "\n"), vol.commit())
    log(f"[chain] copy bucket {n} rows; rebuilding inst shards in tok from v8+copy")
    r = modal.Function.from_name("songgot", "prep_inst").remote(data="sft/train_v8_copy.jsonl", tokd_name="tok")
    log(f"[chain] prep_inst {r}")
    if then_pretrain:
        c = modal.Function.from_name("songgot", "pretrain").spawn(tokens=6e9, p_ko=0.45, p_inst=0.15, ckpt_every=5000, tag="pre5", tokd="tok")
        log(f"[chain] pre5 pretrain spawned {c.object_id}")
    return {"copy_rows": n, "combined": len(lines)}
