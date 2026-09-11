"""ko-vdr parquet -> (page image, markdown, query) pairs for Songgot-V, one container per parquet file.

Resumable and preemption-safe: each file writes corpora/ko_vdr_pairs/NNN.jsonl only when complete (tmp + rename),
and a rerun skips files that already have one. build_all fans out with .map and then merges everything into
corpora/ko_vdr_pages.jsonl with image paths relative to corpora/ko_vdr_pages (NNN/xxxxx.jpg).
The first version wrote one file from one container and restarted from zero on preemption (2026-09-11).

    modal deploy harness/vdr_pairs.py
    .venv/bin/python harness/spawn.py songgot-vdr-pairs build_all
"""
import io
import json
import os
import time

import modal

app = modal.App("songgot-vdr-pairs")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11").pip_install("pyarrow", "pillow", "numpy<2.3")
ROOT = f"{V}/corpora/ko_vdr/data"


def files():
    return sorted(f for f in os.listdir(ROOT) if f.endswith(".parquet"))


@app.function(image=image, volumes={V: vol}, cpu=4, memory=16384, timeout=60 * 60 * 2, retries=3)
def page_file(idx: int, max_side: int = 1280):
    import pyarrow.parquet as pq
    from PIL import Image
    vol.reload()
    fs = files(); f = fs[idx]
    done = f"{V}/corpora/ko_vdr_pairs/{idx:03d}.jsonl"
    if os.path.exists(done):
        return {"idx": idx, "skipped": True}
    outdir = f"{V}/corpora/ko_vdr_pages/{idx:03d}"; os.makedirs(outdir, exist_ok=True); os.makedirs(os.path.dirname(done), exist_ok=True)
    pf = pq.ParquetFile(f"{ROOT}/{f}")
    names = pf.schema.names
    cols = ["image", "markdown"] + [c for c in ("query", "image_id", "query_type") if c in names]
    rows, seen = [], {}
    for rg in range(pf.num_row_groups):
        for row in pf.read_row_group(rg, columns=cols).to_pylist():
            md = row.get("markdown"); im = row.get("image")
            if not md or len(md) < 40 or not im:
                continue
            key = row.get("image_id")
            if key in seen:
                rows.append({"image": seen[key], "text": md, "query": row.get("query"), "dup_page": True}); continue
            b = im.get("bytes") if isinstance(im, dict) else im
            if not b:
                continue
            try:
                pil = Image.open(io.BytesIO(b)).convert("RGB"); pil.thumbnail((max_side, max_side))
            except Exception:
                continue
            name = f"{idx:03d}/{len(seen):05d}.jpg"; pil.save(f"{V}/corpora/ko_vdr_pages/{name}", quality=88); seen[key] = name
            rows.append({"image": name, "text": md, "query": row.get("query"), "query_type": row.get("query_type"), "src": f"ko-vdr/{f}"})
    with open(done + ".tmp", "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(done + ".tmp", done); vol.commit()
    return {"idx": idx, "pages": len(seen), "rows": len(rows)}


@app.function(image=image, volumes={V: vol}, cpu=2, memory=8192, timeout=60 * 60 * 8)
def build_all(max_files: int = 0):
    t0 = time.time(); n = len(files()); idxs = list(range(min(n, max_files) if max_files else n))
    res = list(page_file.map(idxs, return_exceptions=True))
    ok = [r for r in res if isinstance(r, dict)]; bad = [str(r)[:120] for r in res if not isinstance(r, dict)]
    vol.reload(); pages = rows = 0
    with open(f"{V}/corpora/ko_vdr_pages.jsonl", "w", encoding="utf-8") as fo:
        for i in idxs:
            p = f"{V}/corpora/ko_vdr_pairs/{i:03d}.jsonl"
            if os.path.exists(p):
                for l in open(p, encoding="utf-8"):
                    fo.write(l); rows += 1; pages += '"dup_page"' not in l
    vol.commit()
    msg = f"[vdr] DONE {len(ok)}/{len(idxs)} files, {pages} unique pages, {rows} rows (with per-query duplicates), {len(bad)} failed, in {time.time()-t0:.0f}s"
    print(msg, flush=True)
    with open(f"{V}/corpora/PROVENANCE.md", "a", encoding="utf-8") as f:
        f.write(time.strftime("%F %T ") + "NomaDamas/ko-vdr-train-public page pairs (CC BY 4.0): " + msg + "\n")
    vol.commit()
    return {"pages": pages, "rows": rows, "failed": bad[:5]}
