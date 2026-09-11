"""Pull the two licence-clean Korean document corpora onto the Modal volume for Songgot-V.

  NomaDamas/ko-vdr-train-public   310,226 real Korean document pages with markdown + layout, CC BY 4.0
  nvidia/OCR-Synthetic-Multilingual-v1  ko/ split, 2.27M synthetic Korean OCR samples, CC BY 4.0

Both cards state the licence and commercial use; provenance is recorded in /vol/corpora/PROVENANCE.md.

    modal deploy harness/fetch_hf_corpora.py
    .venv/bin/python harness/spawn.py songgot-corpora fetch_vdr
    .venv/bin/python harness/spawn.py songgot-corpora fetch_nvidia_ko max_files=40
"""
import os
import time

import modal

app = modal.App("songgot-corpora")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]", "h5py", "numpy<2.3", "pyarrow").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})


def note(line: str):
    os.makedirs(f"{V}/corpora", exist_ok=True)
    with open(f"{V}/corpora/PROVENANCE.md", "a", encoding="utf-8") as f:
        f.write(time.strftime("%F %T ") + line + "\n")
    print(line, flush=True); vol.commit()


@app.function(image=image, volumes={V: vol}, cpu=8, memory=32768, timeout=60 * 60 * 6, ephemeral_disk=524288)
def fetch_vdr(repo: str = "NomaDamas/ko-vdr-train-public"):
    from huggingface_hub import snapshot_download
    t0 = time.time(); dst = f"{V}/corpora/ko_vdr"
    snapshot_download(repo, repo_type="dataset", local_dir=dst, allow_patterns=["*.parquet", "README.md", "*.json"])
    n = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(dst) for f in fs)
    note(f"{repo}: CC BY 4.0 (card), {n/2**30:.1f} GiB parquet -> corpora/ko_vdr in {time.time()-t0:.0f}s")
    return {"bytes": n}


@app.function(image=image, volumes={V: vol}, cpu=8, memory=32768, timeout=60 * 60 * 8, ephemeral_disk=524288)
def fetch_nvidia_ko(repo: str = "nvidia/OCR-Synthetic-Multilingual-v1", max_files: int = 40):
    """Only the Korean split (train files first), capped so the first training rounds start today."""
    from huggingface_hub import HfApi, hf_hub_download
    t0 = time.time(); api = HfApi(); dst = f"{V}/corpora/nvidia_ocr_ko"; os.makedirs(dst, exist_ok=True)
    files = sorted(f.path for f in api.list_repo_tree(repo, path_in_repo="ko", repo_type="dataset", recursive=True) if f.path.endswith(".h5"))
    train = [f for f in files if "/train" in f or "train" in os.path.basename(f)]
    pick = (train or files)[:max_files]
    got = 0
    for i, fp in enumerate(pick):
        hf_hub_download(repo, fp, repo_type="dataset", local_dir=dst); got += 1
        if i % 5 == 0:
            print(f"[nvidia_ko] {i+1}/{len(pick)} {fp} {time.time()-t0:.0f}s", flush=True); vol.commit()
    n = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(dst) for f in fs)
    note(f"{repo} ko/: CC BY 4.0, explicitly 'ready for commercial use' (card); {got} of {len(files)} h5 files, {n/2**30:.1f} GiB -> corpora/nvidia_ocr_ko in {time.time()-t0:.0f}s")
    return {"files": got, "bytes": n}
