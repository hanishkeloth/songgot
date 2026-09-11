"""Parallel variant of fetch_archive.py: 16 ranged connections via aria2c (GCS serves byte ranges), same SHA-256
check, unpacks to a separate directory so it cannot collide with the single-connection fetch still running.

    modal deploy harness/fetch_archive_fast.py
    .venv/bin/python harness/spawn.py songgot-fetch2 fetch url=<signed url> sha256_expected=<hex>
"""
import hashlib
import os
import subprocess
import time

import modal

app = modal.App("songgot-fetch2")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
raw = modal.Volume.from_name("songgot-raw", create_if_missing=True)
V, R = "/vol", "/raw"
image = modal.Image.debian_slim(python_version="3.11").apt_install("aria2", "unzip", "p7zip-full")


@app.function(image=image, volumes={V: vol, R: raw}, cpu=16, memory=32768, timeout=60 * 60 * 12, ephemeral_disk=1_048_576)
def fetch(url: str, name: str = "vo-org13-20260911.zip", sha256_expected: str = "", dest: str = "vo-org13-fast", conns: int = 16):
    t0 = time.time(); os.makedirs("/scratch", exist_ok=True)
    log = open(f"{V}/ingest.log", "a")
    def say(m):
        print(m, flush=True); log.write(time.strftime("%F %T ") + m + "\n"); log.flush(); vol.commit()
    say(f"[fetch2] downloading {name} with {conns} connections")
    for attempt in range(6):
        rc = subprocess.run(["aria2c", "-c", "-x", str(conns), "-s", str(conns), "-k", "64M", "--file-allocation=none",
                             "--summary-interval=300", "--console-log-level=warn", "--max-tries=10", "--retry-wait=10",
                             "-d", "/scratch", "-o", name, url]).returncode
        if rc == 0:
            break
        say(f"[fetch2] aria2c rc={rc} attempt {attempt + 1}, resuming"); time.sleep(20)
    tmp = f"/scratch/{name}"; size = os.path.getsize(tmp)
    say(f"[fetch2] downloaded {size / 2**30:.1f} GiB in {time.time() - t0:.0f}s")
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        while (b := f.read(64 << 20)):
            h.update(b)
    digest = h.hexdigest(); say(f"[fetch2] sha256 {digest}")
    if sha256_expected and digest != sha256_expected.lower():
        say(f"[fetch2] SHA-256 MISMATCH: expected {sha256_expected}")
        return {"ok": False, "sha256": digest, "bytes": size}
    out = f"{R}/{dest}"; os.makedirs(out, exist_ok=True); say(f"[fetch2] unpacking to {out}")
    r = subprocess.run(["7z", "x", "-y", f"-o{out}", tmp], capture_output=True, text=True)
    if r.returncode != 0:
        say(f"[fetch2] 7z rc={r.returncode}: {r.stderr[-300:]}")
        r = subprocess.run(["unzip", "-q", "-o", tmp, "-d", out], capture_output=True, text=True)
    n = sum(len(fs) for _, _, fs in os.walk(out)); raw.commit()
    say(f"[fetch2] DONE {n} files under {dest} in {time.time() - t0:.0f}s (sha256 {digest[:16]}, {size} bytes)")
    return {"ok": True, "files": n, "sha256": digest, "bytes": size}
