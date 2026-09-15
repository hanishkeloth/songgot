"""Uniform weight average ("model soup") of line-B checkpoints that share a base and a recipe, on Modal CPU.
Runs from the same init and the same data recipe with different data samples land in the same basin; averaging
their weights usually damps the per-function run-to-run swings seen between v10 and v11 (generate_random_password
19 -> 10, count_words 0 -> 20) without a new training run. A soup is a CANDIDATE: it must be scored on the benchmark
against its members with the same scorer before it is used.

    modal deploy harness/soup_base.py
    .venv/bin/python harness/spawn.py songgot-soup-base soup members=base_q35_08b_v10/final,base_q35_08b_v11/final out=soup_q35_08b_v10v11/final
"""
import os
import shutil

import modal

app = modal.App("songgot-soup-base")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "safetensors", "numpy<2.3")


@app.function(image=image, volumes={V: vol}, cpu=8, memory=65536, timeout=60 * 60)
def soup(members: str = "base_q35_08b_v10/final,base_q35_08b_v11/final", out: str = "soup_q35_08b_v10v11/final"):
    import torch
    from safetensors.torch import load_file, save_file
    vol.reload()
    dirs = [f"{V}/ckpt/{m.strip()}" for m in members.split(",") if m.strip()]
    acc = None
    for d in dirs:
        sd = load_file(f"{d}/model.safetensors")
        if acc is None:
            acc = {k: v.to(torch.float32).clone() for k, v in sd.items()}
        else:
            assert set(sd) == set(acc), "parameter sets differ: not the same architecture"
            for k, v in sd.items():
                assert v.shape == acc[k].shape, f"shape differs for {k}"
                acc[k] += v.to(torch.float32)
        print(f"[soup] loaded {d} ({len(sd)} tensors)", flush=True)
    n = len(dirs)
    ref_dtype = load_file(f"{dirs[0]}/model.safetensors")[next(iter(acc))].dtype
    avg = {k: (v / n).to(ref_dtype).contiguous() for k, v in acc.items()}
    od = f"{V}/ckpt/{out}"; os.makedirs(od, exist_ok=True)
    save_file(avg, f"{od}/model.safetensors", metadata={"format": "pt", "soup": ",".join(dirs)})
    for f in os.listdir(dirs[0]):
        if f != "model.safetensors" and os.path.isfile(f"{dirs[0]}/{f}"):
            shutil.copy(f"{dirs[0]}/{f}", f"{od}/{f}")
    vol.commit(); print(f"[soup] DONE {n} members -> ckpt/{out}", flush=True)
    return {"members": dirs, "out": out}
