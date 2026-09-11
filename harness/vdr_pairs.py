"""Turn ko-vdr parquet rows into (image file, text) pairs for Songgot-V alignment: writes JPEGs under
/vol/corpora/ko_vdr_pages and a pairs.jsonl with the page markdown, same shape as ingest_korean.pages.

    modal deploy harness/vdr_pairs.py
    .venv/bin/python harness/spawn.py songgot-vdr-pairs build limit=60000
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


@app.function(image=image, volumes={V: vol}, cpu=16, memory=65536, timeout=60 * 60 * 6)
def build(limit: int = 60000, max_side: int = 1280):
    import pyarrow.parquet as pq
    from PIL import Image
    t0 = time.time(); root = f"{V}/corpora/ko_vdr/data"; out = f"{V}/corpora/ko_vdr_pages"; os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(root) if f.endswith(".parquet"))
    pf0 = pq.ParquetFile(f"{root}/{files[0]}"); print("[vdr] schema:", pf0.schema.names, flush=True)
    img_col = "image"   # struct {bytes, path} in ko-vdr (probed 2026-09-11)
    txt_col = next((c for c in ("markdown", "text") if c in pf0.schema.names), None)
    cols = [img_col, txt_col] + [c for c in ("query", "image_id", "query_type") if c in pf0.schema.names]
    n = 0; seen = set(); seen_name = {}
    with open(f"{V}/corpora/ko_vdr_pages.jsonl", "w", encoding="utf-8") as fo:
        for f in files:
            pf = pq.ParquetFile(f"{root}/{f}")
            for rg in range(pf.num_row_groups):
                tb = pf.read_row_group(rg, columns=cols).to_pylist()
                for row in tb:
                    md = row.get(txt_col); b = row.get(img_col)
                    if not md or len(md) < 40 or not b:
                        continue
                    if isinstance(b, dict):
                        b = b.get("bytes")
                    key = row.get("image_id") or hash(b[:4096])
                    if key in seen:   # one image per page; the rows repeat a page per query
                        fo.write(json.dumps({"image": seen_name[key], "text": md, "query": row.get("query"), "src": f"ko-vdr/{f}", "dup_page": True}, ensure_ascii=False) + "\n")
                        continue
                    try:
                        pil = Image.open(io.BytesIO(b)).convert("RGB"); pil.thumbnail((max_side, max_side))
                    except Exception:
                        continue
                    name = f"vdr_{n:07d}.jpg"; pil.save(f"{out}/{name}", quality=88); seen.add(key); seen_name[key] = name
                    fo.write(json.dumps({"image": name, "text": md, "query": row.get("query"), "query_type": row.get("query_type"), "src": f"ko-vdr/{f}"}, ensure_ascii=False) + "\n"); n += 1
                    if n % 5000 == 0:
                        print(f"[vdr] {n} pages {time.time()-t0:.0f}s", flush=True); vol.commit()
                    if n >= limit:
                        break
                if n >= limit:
                    break
            if n >= limit:
                break
    vol.commit(); print(f"[vdr] DONE {n} pairs in {time.time()-t0:.0f}s", flush=True)
    return {"pairs": n}
