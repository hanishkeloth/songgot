import modal, os
app = modal.App("songgot-volcheck")
vol = modal.Volume.from_name("songgot")
@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/vol": vol}, timeout=120)
def check():
    vol.reload()
    out = {}
    for p in ["/vol/tok/songgot-vocab.gguf", "/vol/tok/spm.model", "/vol/sft/train.jsonl", "/vol/ckpt/pre/final/model.safetensors"]:
        out[p] = (os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else None, os.path.isdir(p))
    out["ls tok"] = sorted(x for x in os.listdir("/vol/tok") if not x.endswith(".bin"))
    return out
@app.local_entrypoint()
def main():
    print(check.remote())
