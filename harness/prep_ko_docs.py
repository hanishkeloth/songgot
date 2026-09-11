"""Corpus v3 for pretraining: the user's own Korean documents plus the ko-vdr page markdown, tokenized with the
Songgot tokenizer into uint16 shards under /vol/tok3, next to copies of the corpus v1 shards (kowiki + fineweb-edu)
so the loader can sample all three. Quality gate matches ingest_korean.keep. Runs on CPU; spawn it after
songgot-ingest::extract has written /vol/private/private_ko.

    modal deploy harness/prep_ko_docs.py
    .venv/bin/python harness/spawn.py songgot-prep3 build
"""
import json
import os
import re
import shutil
import time

import modal

app = modal.App("songgot-prep3")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11").pip_install("sentencepiece", "numpy<2.3", "pyarrow")
HANGUL = re.compile(r"[가-힣]")


def _tok_texts(args):
    """Worker: tokenize a list of documents to one shard, EOS between documents."""
    import numpy as np
    import sentencepiece as spm
    docs, out, model = args
    sp_ = spm.SentencePieceProcessor(model_file=model); ids = []
    for d in docs:
        ids.extend(sp_.encode(d)); ids.append(2)
    np.array(ids, dtype=np.uint16).tofile(out)
    return out, len(ids)


def keep(t: str) -> bool:
    if len(t) < 300:
        return False
    if len(HANGUL.findall(t)) / max(1, len(t)) < 0.15:
        return False
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    return bool(lines) and sum(len(l) for l in lines) / len(lines) >= 12 and len(set(lines)) / len(lines) > 0.5


def vdr_docs(limit_rows: int = 0):
    """ko-vdr page markdown, one document per page, tables and headings kept as text."""
    import pyarrow.parquet as pq
    root = f"{V}/corpora/ko_vdr/data"; n = 0
    for f in sorted(os.listdir(root)):
        if not f.endswith(".parquet"):
            continue
        pf = pq.ParquetFile(f"{root}/{f}")
        cols = [c for c in ("markdown", "text") if c in pf.schema.names]
        if not cols:
            continue
        for rg in range(pf.num_row_groups):
            for md in pf.read_row_group(rg, columns=cols[:1]).column(0).to_pylist():
                if md and keep(md):
                    yield md; n += 1
                    if limit_rows and n >= limit_rows:
                        return


@app.function(image=image, volumes={V: vol}, cpu=32, memory=131072, timeout=60 * 60 * 6)
def build(tokd_name: str = "tok3", docs_per_shard: int = 20000, include_vdr: bool = True, workers: int = 32):
    from multiprocessing import Pool
    t0 = time.time(); tokd = f"{V}/{tokd_name}"; os.makedirs(tokd, exist_ok=True); model = f"{V}/tok/spm.model"
    log = open(f"{V}/prep3.log", "a")
    def say(m):
        print(m, flush=True); log.write(time.strftime("%F %T ") + m + "\n"); log.flush()
    # 1) corpus v1 shards (already tokenized) copied in so one loader sees everything
    for f in sorted(os.listdir(f"{V}/tok")):
        if f.endswith(".bin") and (f.startswith("ko_") or f.startswith("en_")) and not os.path.exists(f"{tokd}/{f}"):
            shutil.copy(f"{V}/tok/{f}", f"{tokd}/{f}")
    say(f"[prep3] corpus v1 shards copied ({time.time()-t0:.0f}s)")
    # 2) the user's documents (private_ko shards are plain text, docs separated by blank lines)
    jobs = []; batch = []; k = 0; n_docs = 0
    def flush(prefix):
        nonlocal batch, k
        if batch:
            jobs.append((list(batch), f"{tokd}/{prefix}_{k:04d}.bin", model)); k += 1; batch = []
    priv = f"{V}/private/private_ko"
    if os.path.isdir(priv):
        for f in sorted(os.listdir(priv)):
            for d in open(f"{priv}/{f}", encoding="utf-8").read().split("\n\n"):
                d = d.strip()
                if keep(d):
                    batch.append(d); n_docs += 1
                    if len(batch) >= docs_per_shard:
                        flush("priv")
        flush("priv"); say(f"[prep3] private docs: {n_docs} kept")
    # 3) ko-vdr markdown
    if include_vdr and os.path.isdir(f"{V}/corpora/ko_vdr/data"):
        k = 0; n_vdr = 0
        for d in vdr_docs():
            batch.append(d); n_vdr += 1
            if len(batch) >= docs_per_shard:
                flush("vdr")
        flush("vdr"); say(f"[prep3] ko-vdr pages kept: {n_vdr}")
    total = 0
    with Pool(workers) as pool:
        for out, n in pool.imap_unordered(_tok_texts, jobs):
            total += n
    vol.commit()
    counts = {}
    for f in os.listdir(tokd):
        if f.endswith(".bin"):
            counts[f.split("_")[0]] = counts.get(f.split("_")[0], 0) + os.path.getsize(f"{tokd}/{f}") // 2
    say(f"[prep3] DONE new tokens {total/1e9:.2f}B; shards by bucket (tokens): " + ", ".join(f"{b} {n/1e9:.2f}B" for b, n in sorted(counts.items())) + f" in {time.time()-t0:.0f}s")
    return counts
