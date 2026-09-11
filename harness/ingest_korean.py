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
                 "lxml", "chardet", "datasketch", "numpy<2.3", "huggingface_hub[hf_transfer]", "olefile")
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
                                      "sha256_first8mb": sha256(p)}, ensure_ascii=False) + "\n")
                rows += 1
                if rows % 20000 == 0:
                    print(f"[scan] {rows} files {time.time()-t0:.0f}s {by_kind}", flush=True)
    vol.commit()
    msg = f"[scan] DONE {rows} files in {time.time()-t0:.0f}s {by_kind}"
    print(msg, flush=True); open(f"{V}/ingest.log", "a").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    return by_kind


DOC_SEP = "\n\x1e\n"   # record separator between documents in text shards (documents contain blank lines)

PII = [  # (label, pattern); order matters: most specific first
    ("주민등록번호", re.compile(r"(?<!\d)\d{6}\s?-\s?[1-8]\d{6}(?!\d)")),
    ("카드번호", re.compile(r"(?<!\d)\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}(?!\d)")),
    ("휴대전화", re.compile(r"(?<!\d)01[016789][-. ]?\d{3,4}[-. ]?\d{4}(?!\d)")),
    ("전화번호", re.compile(r"(?<!\d)0(?:2|[3-6][1-5]|70|80)[-. )]\s?\d{3,4}[-. ]\d{4}(?!\d)")),
    ("이메일", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]
ACCOUNT = re.compile(r"(?<!\d)\d{2,6}-\d{2,6}-\d{2,8}(?:-\d{1,4})?(?!\d)")


def mask_pii(t: str):
    """Mask Korean PII patterns with typed placeholders; returns (text, counts). Names and addresses are NOT caught."""
    counts = {}
    for label, rx in PII:
        t, n = rx.subn(f"[{label}]", t)
        if n:
            counts[label] = n
    def acct(m):
        d = re.sub(r"\D", "", m.group(0))
        if len(d) >= 10 and not re.fullmatch(r"(19|20)\d{6}", d[:8]):
            counts["계좌·등록번호"] = counts.get("계좌·등록번호", 0) + 1
            return "[계좌번호]"
        return m.group(0)
    t = ACCOUNT.sub(acct, t)
    return t, counts


def hwp5_text(path):
    """HWP 5.x: OLE container, BodyText/SectionN streams, zlib-deflated when the header says so, records of which
    HWPTAG_PARA_TEXT (67) carry UTF-16LE text with inline/extended control characters that occupy 8 code units."""
    import olefile, struct, zlib
    ole = olefile.OleFileIO(path)
    try:
        flags = struct.unpack("<I", ole.openstream("FileHeader").read()[36:40])[0]
        if flags & 2:                                   # encrypted document
            return ""
        compressed = bool(flags & 1)
        secs = sorted((e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText" and e[1].startswith("Section")), key=lambda e: int(e[1][7:] or 0))
        out = []
        for e in secs:
            data = ole.openstream(e).read()
            if compressed:
                try:
                    data = zlib.decompress(data, -15)
                except zlib.error:
                    continue
            i, n = 0, len(data)
            while i + 4 <= n:
                h = struct.unpack_from("<I", data, i)[0]; i += 4
                tag, size = h & 0x3FF, (h >> 20) & 0xFFF
                if size == 0xFFF:
                    if i + 4 > n:
                        break
                    size = struct.unpack_from("<I", data, i)[0]; i += 4
                if tag == 67:
                    b = data[i:i + size]; w = struct.unpack(f"<{len(b)//2}H", b[:len(b)//2*2]); keep, j = [], 0
                    while j < len(w):
                        c = w[j]
                        if c >= 32:
                            keep.append(c); j += 1
                        elif c in (10, 13):
                            keep.append(10); j += 1
                        elif c in (30, 31):
                            keep.append(32); j += 1
                        elif c == 0 or 24 <= c <= 29:
                            j += 1
                        else:
                            j += 8                      # inline (4-9, 19-20) and extended (1-3, 11-12, 14-18, 21-23) controls
                    out.append(struct.pack(f"<{len(keep)}H", *keep).decode("utf-16le", "ignore"))
                i += size
        return "\n".join(x for x in out if x.strip())
    finally:
        ole.close()


def hwpx_text(path):
    """HWPX: a zip of OWPML XML; paragraph text sits in <hp:t> runs inside <hp:p>."""
    import html, zipfile
    z = zipfile.ZipFile(path); out = []
    names = sorted((n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n)), key=lambda s: int(re.findall(r"\d+", s)[-1]))
    for n in names:
        x = z.read(n).decode("utf-8", "ignore")
        for para in x.split("</hp:p>"):
            t = "".join(re.findall(r"<hp:t(?:\s[^>]*)?>(.*?)</hp:t>", para, re.S))
            t = html.unescape(re.sub(r"<[^>]+>", "", t))
            if t.strip():
                out.append(t)
    return "\n".join(out)


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
        if ext == ".hwp":
            return hwp5_text(path)
        if ext == ".hwpx":
            return hwpx_text(path)
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
    seen_exact = set(); buf = []; n_chars = 0; k = 0; kept = 0; dropped = 0; t0 = time.time(); pii = {}; by_ext = {}

    def flush():
        nonlocal buf, k, n_chars
        if not buf:
            return
        open(f"{V}/private/{out_name}/shard_{k:04d}.txt", "w", encoding="utf-8").write(DOC_SEP.join(buf))
        k += 1; buf = []; n_chars = 0; vol.commit()

    for i, m in enumerate(todo):
        t = read_text(os.path.join(root, m["path"]), m["ext"])
        t = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", t)).strip()
        t, pc = mask_pii(t)
        for k_, v_ in pc.items():
            pii[k_] = pii.get(k_, 0) + v_
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
        buf.append(t); n_chars += len(t); kept += 1; by_ext[m["ext"]] = by_ext.get(m["ext"], 0) + 1
        if n_chars >= shard_chars:
            flush()
        if i % 5000 == 0:
            print(f"[extract] {i}/{len(todo)} kept {kept} dropped {dropped} shards {k} {time.time()-t0:.0f}s", flush=True)
    flush()
    total = sum(os.path.getsize(f"{V}/private/{out_name}/{f}") for f in os.listdir(f"{V}/private/{out_name}"))
    json.dump({"kept": kept, "dropped": dropped, "kept_by_ext": by_ext, "pii_masked": pii, "note": "names and addresses are not masked"},
              open(f"{V}/private/{out_name}_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    msg = f"[extract] DONE kept {kept} dropped {dropped} -> {k} shards, {total/1e9:.2f} GB of Korean text in {time.time()-t0:.0f}s; kept by type {by_ext}; PII masked {pii}"
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
                    name = f"{(m.get('sha256_first8mb') or str(i))[:16]}_{pi}.jpg"
                    img.convert("RGB").save(f"{outdir}/{name}", quality=88)
                    out.write(json.dumps({"image": name, "text": mask_pii(txt)[0], "src": m["path"], "page": pi}, ensure_ascii=False) + "\n"); n += 1
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


@app.function(image=image, volumes={V: vol, R: raw}, cpu=1, memory=2048, timeout=60 * 60 * 24)
def chain(root: str = "/raw/vo-org13-fast", wait_for: str = "[fetch2] DONE"):
    """Server-side orchestration so no laptop process can interrupt it: wait for the archive to finish unpacking,
    then provenance scan, then text extraction and page rendering in parallel, then tokenization (songgot-prep3)."""
    t0 = time.time()
    def say(m):
        print(m, flush=True); open(f"{V}/ingest.log", "a").write(time.strftime("%F %T ") + m + "\n"); vol.commit()
    while True:
        vol.reload()
        if os.path.exists(f"{V}/ingest.log") and wait_for in open(f"{V}/ingest.log", encoding="utf-8").read():
            break
        time.sleep(120)
    say(f"[chain] archive ready after {time.time()-t0:.0f}s; scanning {root}")
    kinds = scan.remote(root)
    say(f"[chain] scan done: {kinds}; extracting text and rendering pages")
    e = extract.spawn(root=root); pg = pages.spawn(root=root)
    ex = e.get(); pp = pg.get()
    say(f"[chain] extract {ex}; pages {pp}; tokenizing")
    tok = modal.Function.from_name("songgot-prep3", "build").remote()
    say(f"[chain] DONE in {time.time()-t0:.0f}s; token buckets {tok}")
    return {"scan": kinds, "extract": ex, "pages": pp, "tokens": tok}
