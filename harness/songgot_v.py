"""Songgot-V: a tiny Korean document-understanding VLM.

Architecture, and what is ours: the language model is Songgot, trained from scratch by us (12 or 16
layers, our own 32k Korean tokenizer). The vision encoder is an open pretrained tower (SigLIP2,
Apache 2.0) which we do NOT claim to have trained; it is frozen for stage 1. Between them sits a
pixel-shuffle + MLP projector trained from random init. This is the nanoVLM / SmolVLM arrangement and
the split is stated on the model card, because "trained from scratch" must mean the part we trained.

Stages:
  1. align   - frozen tower, frozen LM, train the projector only, on (image, text) pairs
  2. sft     - unfreeze the LM, train projector + LM on Korean document questions
Both read pairs written by harness/ingest_korean.py (the user's own documents) and any synthetic
document renders.

    modal deploy harness/songgot_v.py
    .venv/bin/python harness/spawn.py songgot-v align  pairs=private/private_pages.jsonl steps=4000
"""
import json
import os
import random
import time

import modal

app = modal.App("songgot-v")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V, CACHE = "/vol", "/root/.cache/huggingface"
TOWER = "google/siglip2-base-patch16-256"   # Apache 2.0, 86M params, 256px, strong multilingual OCR prior
SPECIAL = ["<|system|>", "<|user|>", "<|call|>", "<|end|>", "<|pad|>"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "transformers==4.51.3", "huggingface_hub[hf_transfer]",
                 "numpy<2.3", "safetensors", "accelerate", "pillow", "sentencepiece")
    .pip_install("llama-cpp-python", extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)


def build(lm_dir: str, tower_id: str = TOWER, pool: int = 2):
    """Songgot LM + frozen SigLIP2 tower + pixel-shuffle MLP projector."""
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, SiglipVisionModel

    class Projector(nn.Module):
        """Pixel shuffle by `pool` (256 patches -> 64 tokens at pool=2) then a 2-layer MLP into LM width."""
        def __init__(self, d_in, d_out, pool):
            super().__init__()
            self.pool = pool
            self.net = nn.Sequential(nn.Linear(d_in * pool * pool, d_out * 2), nn.GELU(), nn.Linear(d_out * 2, d_out))
            self.norm = nn.LayerNorm(d_out)

        def forward(self, x):                       # x: (B, N, D), N a square
            B, N, D = x.shape; s = int(N ** 0.5); p = self.pool
            x = x.view(B, s, s, D)
            x = x.view(B, s // p, p, s // p, p, D).permute(0, 1, 3, 2, 4, 5).reshape(B, (s // p) ** 2, D * p * p)
            return self.norm(self.net(x))

    class SonggotV(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = SiglipVisionModel.from_pretrained(tower_id, torch_dtype=torch.bfloat16)
            self.lm = AutoModelForCausalLM.from_pretrained(lm_dir, torch_dtype=torch.bfloat16)
            self.proj = Projector(self.tower.config.hidden_size, self.lm.config.hidden_size, pool).to(torch.bfloat16)
            self.n_img = (self.tower.config.image_size // self.tower.config.patch_size // pool) ** 2

        def embed(self, pixel_values, input_ids, img_slots):
            v = self.tower(pixel_values=pixel_values).last_hidden_state
            v = self.proj(v)                                        # (B, n_img, d)
            e = self.lm.get_input_embeddings()(input_ids)
            for b in range(e.shape[0]):
                e[b, img_slots[b]: img_slots[b] + v.shape[1]] = v[b]
            return e

        def forward(self, pixel_values, input_ids, labels, attention_mask, img_slots):
            e = self.embed(pixel_values, input_ids, img_slots)
            return self.lm(inputs_embeds=e, attention_mask=attention_mask, labels=labels)

    return SonggotV()


def tok_of():
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    p = f"{V}/tok/songgot-vocab.gguf"
    if not os.path.exists(p):
        p = hf_hub_download("palette-lab/songgot", "songgot-vocab.gguf")
    t = Llama(model_path=p, vocab_only=True, verbose=False)
    return (lambda s: t.tokenize(s.encode("utf-8"), add_bos=False, special=True),
            lambda i: t.detokenize([int(x) for x in i], special=True).decode("utf-8", "replace"),
            t.token_bos(), t.token_eos())


@app.function(image=image, gpu="H100", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 12, memory=131072)
def align(lm: str = "ckpt/sftv8e3e3/final", pairs: str = "private/private_pages.jsonl", img_root: str = "private/private_pages",
          steps: int = 4000, batch: int = 16, lr: float = 1e-3, max_len: int = 768, out: str = "v/align", train_lm: bool = False,
          seed: int = 0, log_every: int = 50):
    """Stage 1: teach the projector to speak the LM's embedding space. Loss on the page text only."""
    import torch
    from PIL import Image
    from transformers import SiglipImageProcessor
    vol.reload()
    enc, dec, bos, eos = tok_of()
    end_id = 3 + SPECIAL.index("<|end|>"); pad = 3 + SPECIAL.index("<|pad|>")
    model = build(f"{V}/{lm}").cuda()
    proc = SiglipImageProcessor.from_pretrained(TOWER)
    for p_ in model.tower.parameters():
        p_.requires_grad_(False)
    for p_ in model.lm.parameters():
        p_.requires_grad_(train_lm)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    n_tr = sum(p_.numel() for p_ in params)
    rows = [json.loads(l) for l in open(f"{V}/{pairs}", encoding="utf-8") if '"text": null' not in l]
    rng = random.Random(seed); rng.shuffle(rows)
    log = open(f"{V}/songgot_v.log", "a")
    def say(m):
        print(m, flush=True); log.write(time.strftime("%F %T ") + m + "\n"); log.flush()
    say(f"[v] align: {len(rows)} pairs, {n_tr/1e6:.1f}M trainable of {sum(p_.numel() for p_ in model.parameters())/1e6:.1f}M, {model.n_img} image tokens")
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.03)
    prompt_ids = [bos] + enc("<|user|>\n")            # image tokens go right after this
    tail_ids = enc("\n<|call|>\n")
    ptr = 0; t0 = time.time(); model.train()
    for step in range(steps):
        px, ids, labs, slots = [], [], [], []
        while len(px) < batch and ptr < len(rows):
            r = rows[ptr]; ptr += 1
            if ptr >= len(rows):
                rng.shuffle(rows); ptr = 0
            try:
                im = Image.open(f"{V}/{img_root}/{r['image']}").convert("RGB")
            except Exception:
                continue
            body = enc(r["text"][:1200]) + [end_id]
            seq = prompt_ids + [pad] * model.n_img + tail_ids + body
            lab = [-100] * (len(seq) - len(body)) + body
            px.append(proc(images=im, return_tensors="pt")["pixel_values"][0])
            slots.append(len(prompt_ids)); ids.append(seq[:max_len]); labs.append(lab[:max_len])
        if not px:
            break
        L = max(len(s) for s in ids)
        inp = torch.full((len(ids), L), pad, dtype=torch.long); lb = torch.full((len(ids), L), -100, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for j, (s, l) in enumerate(zip(ids, labs)):
            inp[j, :len(s)] = torch.tensor(s); lb[j, :len(l)] = torch.tensor(l); att[j, :len(s)] = 1
        loss = model(torch.stack(px).cuda().to(torch.bfloat16), inp.cuda(), lb.cuda(), att.cuda(), slots).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % log_every == 0 or step == steps - 1:
            say(f"[v] step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} elapsed {(time.time()-t0)/60:.1f}m")
        if step and step % 1000 == 0:
            os.makedirs(f"{V}/ckpt/{out}", exist_ok=True)
            torch.save({"proj": model.proj.state_dict(), "step": step}, f"{V}/ckpt/{out}/proj.pt"); vol.commit()
    os.makedirs(f"{V}/ckpt/{out}", exist_ok=True)
    torch.save({"proj": model.proj.state_dict(), "step": steps}, f"{V}/ckpt/{out}/proj.pt")
    if train_lm:
        model.lm.save_pretrained(f"{V}/ckpt/{out}/lm", safe_serialization=True)
    json.dump({"lm": lm, "tower": TOWER, "n_img": model.n_img, "pairs": pairs, "steps": steps}, open(f"{V}/ckpt/{out}/config.json", "w"))
    vol.commit(); say("[v] align DONE")
    return {"steps": steps}
