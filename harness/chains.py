"""Server-side chains that wait on the volume and then call functions of other deployed apps, so no laptop process
is in the loop (local waiters get killed under memory pressure; `modal run` clients cancel their calls when killed).

    modal deploy harness/chains.py
    .venv/bin/python harness/spawn.py songgot-chains docqa_to_v2      # docqa DONE -> songgot-v2 sft -> eval_k
    .venv/bin/python harness/spawn.py songgot-chains trackb_fast      # sft ckpt -> songgot-base2 predict
"""
import os
import time

import modal

app = modal.App("songgot-chains")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11")


def log(m):
    print(m, flush=True)
    with open(f"{V}/chains.log", "a") as f:
        f.write(time.strftime("%F %T ") + m + "\n")
    vol.commit()


def wait_for(path=None, marker=None, logfile=None, every=120, hours=24):
    t0 = time.time()
    while time.time() - t0 < hours * 3600:
        vol.reload()
        if path and os.path.exists(f"{V}/{path}"):
            return True
        if marker and logfile and os.path.exists(f"{V}/{logfile}") and marker in open(f"{V}/{logfile}", encoding="utf-8", errors="ignore").read():
            return True
        time.sleep(every)
    return False


@app.function(image=image, volumes={V: vol}, cpu=1, memory=1024, timeout=60 * 60 * 24)
def docqa_to_v2(steps: int = 6000):
    ok = wait_for(marker="[docqa] DONE", logfile="teacher.log")
    log(f"[chain] docqa ready={ok}; starting Songgot-V stage 2")
    r = modal.Function.from_name("songgot-v2", "sft").remote(steps=steps)
    log(f"[chain] v2 sft {r}; scoring K-DTCBench and K-MMBench")
    s = modal.Function.from_name("songgot-v2", "eval_k").remote(ckpt="ckpt/v2/sft_docqa", tag="songgot_v_docqa", benches="kdtcbench,kmmbench", limit=1000)
    log(f"[chain] SONGGOT-V SCORED {s}")
    return s


@app.function(image=image, volumes={V: vol}, cpu=1, memory=1024, timeout=60 * 60 * 24)
def trackb_fast(ckpt: str = "base_q35_08b_v8fast/final", name: str = "q35_08b_v8fast"):
    ok = wait_for(path=f"ckpt/{ckpt}/model.safetensors")
    log(f"[chain] track B fast ckpt ready={ok}; predicting")
    r = modal.Function.from_name("songgot-base2", "predict").remote(ckpt=ckpt, name=name)
    log(f"[chain] TRACK-B FAST PREDICTED {r} -> preds/{name}.jsonl")
    return r


@app.function(image=image, volumes={V: vol}, cpu=1, memory=1024, timeout=60 * 60 * 24)
def trackb_slow(ckpt: str = "base_q35_08b_v8/final", name: str = "q35_08b_v8"):
    ok = wait_for(path=f"ckpt/{ckpt}/model.safetensors", hours=23)
    log(f"[chain] track B slow ckpt ready={ok}; predicting")
    r = modal.Function.from_name("songgot-base", "predict").remote(ckpt=ckpt, name=name)
    log(f"[chain] TRACK-B SLOW PREDICTED {r} -> preds/{name}.jsonl")
    return r
