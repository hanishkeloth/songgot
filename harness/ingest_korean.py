"""Ingest a large private Korean corpus (documents and images) into Modal for training.

Two halves, because the two models want different things from the same 300 GB:
  text  -> Korean pretraining shards for the language model (quality-filtered, deduplicated)
  image -> page images plus their text for the document-understanding VLM

Provenance is recorded for every file (path, sha256, size, type) in corpus_manifest.jsonl before
anything is extracted, so the training set can always be traced back to a source the user owns.

    modal deploy harness/ingest_korean.py
    .venv/bin/python harness/spawn.py songgot-ingest scan     root=/raw/private
    .venv/bin/python harness/spawn.py songgot-ingest extract  root=/raw/private
"""
import hashlib
import json
import os
import re
import time

import modal

app = modal.App("songgot-ingest")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
raw = modal.Volume.from_name("songgot-raw", create_if_missing=True)  # the private 300 GB lands here
V, R = "/vol", "/raw"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("poppler-utils", "libgl1", "libglib2.0-0")
    .pip_install("pypdfium2", "pillow", "python-docx", "openpyxl", "python-pptx", "beautifulsoup4",
                 "lxml", "chardet", "datasketch", "numpy<2.3", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

TEXT_EXT = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".xml"}
DOC_EXT = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".hwp", ".hwpx"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
HANGUL = re.compile(r"[가-힣]")


def sha256(path, cap=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(1 << 20)):
            h.update(b)
            cap -= len(b)
            if cap <= 0:
                break
    return h.hexdigest()


@app.function(image=image, volumes={V: vol, R: raw}, cpu=16, memory=65536, timeout=60 * 60 * 6)
def scan(root: str = "/raw"):
    """Walk the private volume, record every file with its hash and type. Nothing is read for content."""
    t0 = time.time(); rows = 0; by_kind = {}
    os.makedirs(f"{V}/private", exist_ok=True)
    with open(f"{V}/private/corpus_manifest.jsonl", "w", encoding="utf-8") as out:
        for dirpath, _, names in os.walk(root):
            for n in names:
                p = os.path.join(dirpath, n); ext = os.path.splitext(n)[1].lower()
                kind = "text" if ext in TEXT_EXT else "doc" if ext in DOC_EXT else "image" if ext in IMG_EXT else "other"
                try:
                    size = os.path.getsize(p)
                except OSError:
                    continue
                by_kind[kind] = by_kind.get(kind, 0) + 1
                out.write(json.dumps({"path": os.path.relpath(p, root), "kind": kind, "ext": ext, "bytes": size,
                                      "sha256": sha256(p) if size < (64 << 20) else None}, ensure_ascii=False) + "\n")
                rows += 1
                if rows % 20000 == 0:
                    print(f"[scan] {rows} files {time.time()-t0:.0f}s {by_kind}", flush=True)
    vol.commit()
    msg = f"[scan] DONE {rows} files in {time.time()-t0:.0f}s {by_kind}"
    print(msg, flush=True); open(f"{V}/ingest.log", "a").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    return by_kind


def read_text(path, ext):
    """Plain text out of one file. Returns '' for anything we cannot read without OCR."""
    try:
        if ext in TEXT_EXT:
            import chardet
            b = open(path, "rb").read()
            enc = chardet.detect(b[:200000])["encoding"] or "utf-8"
            t = b.decode(enc, "replace")
            if ext in (".html", ".htm", ".xml"):
                from bs4 import BeautifulSoup
                t = BeautifulSoup(t, "lxml").get_text("\n")
            return t
        if ext == ".pdf":
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(path)
            return "\n".join(doc[i].get_textpage().get_text_range() for i in range(min(len(doc), 400)))
        if ext == ".docx":
            import docx
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
        if ext == ".pptx":
            from pptx import Presentation
            return "\n".join(s.text_frame.text for sl in Presentation(path).slides for s in sl.shapes if s.has_text_frame)
        if ext in (".xlsx", ".xls"):
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            return "\n".join("\t".join(str(c) for c in row if c is not None) for ws in wb for row in ws.iter_rows(values_only=True))
    except Exception as e:
        print(f"[extract] skip {os.path.basename(path)}: {type(e).__name__}", flush=True)
    return ""


def keep(t: str) -> bool:
    """Quality gate: long enough, genuinely Korean, not a table of dots or a nav bar."""
    if len(t) < 300:
        return False
    han = len(HANGUL.findall(t))
    if han / max(1, len(t)) < 0.15:
        return False
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if not lines or sum(len(l) for l in lines) / len(lines) < 12:
        return False
    return len(set(lines)) / len(lines) > 0.5


@app.function(image=image, volumes={V: vol, R: raw}, cpu=32, memory=131072, timeout=60 * 60 * 12)
def extract(root: str = "/raw", out_name: str = "private_ko", shard_chars: int = 200_000_000, near_dup: bool = True):
    """Text half: every readable file -> quality-gated, deduplicated Korean text shards under /vol/private."""
    from datasketch import MinHash, MinHashLSH
    man = [json.loads(l) for l in open(f"{V}/private/corpus_manifest.jsonl", encoding="utf-8")]
    todo = [m for m in man if m["kind"] in ("text", "doc")]
    print(f"[extract] {len(todo)} readable files of {len(man)}", flush=True)
    os.makedirs(f"{V}/private/{out_name}", exist_ok=True)
    lsh = MinHashLSH(threshold=0.85, num_perm=64) if near_dup else None
    seen_exact = set(); buf = []; n_chars = 0; k = 0; kept = 0; dropped = 0; t0 = time.time()

    def flush():
        nonlocal buf, k, n_chars
        if not buf:
            return
        open(f"{V}/private/{out_name}/shard_{k:04d}.txt", "w", encoding="utf-8").write("\n\n".join(buf))
        k += 1; buf = []; n_chars = 0; vol.commit()

    for i, m in enumerate(todo):
        t = read_text(os.path.join(root, m["path"]), m["ext"])
        t = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", t)).strip()
        if not keep(t):
            dropped += 1; continue
        h = hashlib.md5(t.encode()).hexdigest()
        if h in seen_exact:
            dropped += 1; continue
        seen_exact.add(h)
        if lsh is not None:
            mh = MinHash(num_perm=64)
            for tok in {t[j:j + 5] for j in range(0, min(len(t), 20000), 3)}:
                mh.update(tok.encode())
            if lsh.query(mh):
                dropped += 1; continue
            lsh.insert(h, mh)
        buf.append(t); n_chars += len(t); kept += 1
        if n_chars >= shard_chars:
            flush()
        if i % 5000 == 0:
            print(f"[extract] {i}/{len(todo)} kept {kept} dropped {dropped} shards {k} {time.time()-t0:.0f}s", flush=True)
    flush()
    total = sum(os.path.getsize(f"{V}/private/{out_name}/{f}") for f in os.listdir(f"{V}/private/{out_name}"))
    msg = f"[extract] DONE kept {kept} dropped {dropped} -> {k} shards, {total/1e9:.2f} GB of Korean text in {time.time()-t0:.0f}s"
    print(msg, flush=True); open(f"{V}/ingest.log", "a").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    return {"kept": kept, "dropped": dropped, "shards": k, "bytes": total}


@app.function(image=image, volumes={V: vol, R: raw}, cpu=32, memory=131072, timeout=60 * 60 * 12)
def pages(root: str = "/raw", out_name: str = "private_pages", dpi: int = 144, max_pages: int = 8, limit: int = 0):
    """Image half: render PDF/office pages to JPEGs with their own text layer, and catalogue existing images.
    Each row of pages.jsonl is a (image, text) pair, which is what the document-understanding VLM trains on."""
    import pypdfium2 as pdfium
    from PIL import Image
    man = [json.loads(l) for l in open(f"{V}/private/corpus_manifest.jsonl", encoding="utf-8")]
    docs = [m for m in man if m["ext"] == ".pdf"]; imgs = [m for m in man if m["kind"] == "image"]
    if limit:
        docs, imgs = docs[:limit], imgs[:limit]
    outdir = f"{V}/private/{out_name}"; os.makedirs(outdir, exist_ok=True)
    n = 0; t0 = time.time()
    with open(f"{V}/private/{out_name}.jsonl", "w", encoding="utf-8") as out:
        for i, m in enumerate(docs):
            try:
                doc = pdfium.PdfDocument(os.path.join(root, m["path"]))
                for pi in range(min(len(doc), max_pages)):
                    page = doc[pi]
                    txt = page.get_textpage().get_text_range().strip()
                    if len(HANGUL.findall(txt)) < 20:
                        continue
                    img = page.render(scale=dpi / 72).to_pil()
                    img.thumbnail((1280, 1280))
                    name = f"{m['sha256'][:16] if m.get('sha256') else i}_{pi}.jpg"
                    img.convert("RGB").save(f"{outdir}/{name}", quality=88)
                    out.write(json.dumps({"image": name, "text": txt, "src": m["path"], "page": pi}, ensure_ascii=False) + "\n"); n += 1
            except Exception as e:
                print(f"[pages] skip {os.path.basename(m['path'])}: {type(e).__name__}", flush=True)
            if i % 500 == 0:
                print(f"[pages] doc {i}/{len(docs)} pages {n} {time.time()-t0:.0f}s", flush=True); vol.commit()
        for i, m in enumerate(imgs):
            out.write(json.dumps({"image_src": m["path"], "text": None, "src": m["path"]}, ensure_ascii=False) + "\n")
    vol.commit()
    msg = f"[pages] DONE {n} rendered pages with text, {len(imgs)} source images catalogued, in {time.time()-t0:.0f}s"
    print(msg, flush=True); open(f"{V}/ingest.log", "a").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    return {"pages": n, "images": len(imgs)}
