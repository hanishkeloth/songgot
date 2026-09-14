"""Data operations that run on Modal (the Mac is memory-starved; nothing large is processed locally).

    modal deploy harness/data_ops.py
    .venv/bin/python harness/spawn.py songgot-data copy_consistent src=sft/train_v8.jsonl dst=sft/train_v8cc.jsonl then_train=True
"""
import json
import random
import re
import time

import modal

app = modal.App("songgot-data")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11")


def verbatim(r):
    """Every argument value appears in the query (whitespace ignored); booleans exempt; nothing else invented."""
    if r["call"]["name"] == "none":
        return True
    q = re.sub(r"\s+", "", r["query"])
    for v in (r["call"].get("arguments") or {}).values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            if str(v) not in q and str(int(v)) not in q:
                return False
        elif isinstance(v, str):
            if re.sub(r"\s+", "", v) not in q:
                return False
        else:
            return False
    return True


@app.function(image=image, volumes={V: vol}, cpu=4, memory=32768, timeout=60 * 60 * 2)
def copy_consistent(src: str = "sft/train_v8.jsonl", dst: str = "sft/train_v8cc.jsonl", seed: int = 9, then_train: bool = False, limit: int = 200000):
    vol.reload(); t0 = time.time(); keep = []; n = 0
    for l in open(f"{V}/{src}", encoding="utf-8"):
        r = json.loads(l); n += 1
        if verbatim(r):
            keep.append(l)
    random.Random(seed).shuffle(keep)
    open(f"{V}/{dst}", "w", encoding="utf-8").writelines(keep); vol.commit()
    msg = f"[data] copy_consistent: kept {len(keep)} of {n} rows -> {dst} in {time.time()-t0:.0f}s"
    print(msg, flush=True); open(f"{V}/chains.log", "a").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    if then_train:
        out = dst.split("/")[-1].replace("train_", "base_q35_08b_").replace(".jsonl", "") + "/final"
        c = modal.Function.from_name("songgot-base2", "sft").spawn(base="Qwen/Qwen3.5-0.8B", out=out, data=dst, epochs=1, limit=limit, batch=8, accum=4)
        modal.Function.from_name("songgot-chains", "trackb_fast").spawn(ckpt=out, name=out.split("/")[0])
        open(f"{V}/chains.log", "a").write(time.strftime("%F %T ") + f"[data] spawned Track B sft on {dst} -> ckpt/{out} ({c.object_id}) and its predict chain\n"); vol.commit()
    return {"kept": len(keep), "total": n}
