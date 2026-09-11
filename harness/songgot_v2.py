"""Songgot-V, tiled: a tiny Korean document-understanding VLM that can actually read a page.

v1 fed a whole page as one 256px tile; text on a page is unreadable at that size and the alignment loss
flattened at ~5.4 after 25 steps (2026-09-11). v2 follows SmolVLM: 512px tiles, the page letterboxed onto a
2x3 grid (portrait) or 3x2 (landscape) plus one global thumbnail = 7 tiles, each pixel-shuffled by 4 into 64
tokens, 448 image tokens per page.

What is ours: the language model (Songgot, trained from scratch) and the projector (random init). The vision
tower is google/siglip2-base-patch16-512 (Apache 2.0), pretrained by Google, frozen in stage 1; the model card
says so.

Every log step also measures IMAGE GAIN: the loss on the same batch with the images shuffled across samples,
minus the loss with the right images. If the projector carries information the gain is positive and growing;
if it is ~0 the model is only learning a text prior.

    modal deploy harness/songgot_v2.py
    .venv/bin/python harness/spawn.py songgot-v2 align pairs=corpora/ko_vdr_pages.jsonl img_root=corpora/ko_vdr_pages steps=3000
"""
import json
import os
import random
import time

import modal

app = modal.App("songgot-v2")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V, CACHE = "/vol", "/root/.cache/huggingface"
TOWER = "google/siglip2-base-patch16-512"
SPECIAL = ["<|system|>", "<|user|>", "<|call|>", "<|end|>", "<|pad|>"]
TILE = 512

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "transformers==4.51.3", "huggingface_hub[hf_transfer]",
                 "numpy<2.3", "safetensors", "accelerate", "pillow", "sentencepiece")
    .pip_install("llama-cpp-python", extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)


def tiles_of(img):
    """Letterbox onto a 2x3 (portrait) or 3x2 (landscape) canvas of 512px tiles, plus a global thumbnail."""
    from PIL import Image
    w, h = img.size; cols, rows = (2, 3) if h >= w else (3, 2)
    cw, ch = cols * TILE, rows * TILE
    s = min(cw / w, ch / h); nw, nh = max(1, int(w * s)), max(1, int(h * s))
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255)); canvas.paste(img.resize((nw, nh), Image.BICUBIC), ((cw - nw) // 2, (ch - nh) // 2))
    thumb = Image.new("RGB", (TILE, TILE), (255, 255, 255)); t = img.copy(); t.thumbnail((TILE, TILE), Image.BICUBIC)
    thumb.paste(t, ((TILE - t.size[0]) // 2, (TILE - t.size[1]) // 2))
    return [thumb] + [canvas.crop((c * TILE, r * TILE, (c + 1) * TILE, (r + 1) * TILE)) for r in range(rows) for c in range(cols)]


def build(lm_dir: str, tower_id: str = TOWER, shuffle: int = 4):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, SiglipVisionModel

    class Projector(nn.Module):
        def __init__(self, d_in, d_out, r):
            super().__init__()
            self.r = r
            self.net = nn.Sequential(nn.Linear(d_in * r * r, d_out * 2), nn.GELU(), nn.Linear(d_out * 2, d_out))
            self.norm = nn.LayerNorm(d_out)

        def forward(self, x):                         # (T, N, D) with N a square
            T, N, D = x.shape; s = int(N ** 0.5); r = self.r
            x = x.view(T, s // r, r, s // r, r, D).permute(0, 1, 3, 2, 4, 5).reshape(T, (s // r) ** 2, D * r * r)
            return self.norm(self.net(x))

    class SonggotV(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = SiglipVisionModel.from_pretrained(tower_id, torch_dtype=torch.bfloat16)
            self.lm = AutoModelForCausalLM.from_pretrained(lm_dir, torch_dtype=torch.bfloat16)
            self.proj = Projector(self.tower.config.hidden_size, self.lm.config.hidden_size, shuffle)
            per = (self.tower.config.image_size // self.tower.config.patch_size // shuffle) ** 2
            self.n_img = per * 7

        def image_tokens(self, px):                   # px: (B, 7, 3, 512, 512)
            B = px.shape[0]
            with torch.no_grad():
                v = self.tower(pixel_values=px.flatten(0, 1).to(torch.bfloat16)).last_hidden_state
            v = self.proj(v.float())                   # (B*7, 64, d), projector in fp32
            return v.view(B, -1, v.shape[-1])          # (B, 448, d)

        def forward(self, px, input_ids, labels, attention_mask, slot):
            v = self.image_tokens(px)
            e = self.lm.get_input_embeddings()(input_ids).float()
            e = torch.cat([e[:, :slot], v, e[:, slot + v.shape[1]:]], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.lm(inputs_embeds=e, attention_mask=attention_mask, labels=labels)

    return SonggotV()


class Pairs:
    """Torch dataset over pairs.jsonl rows -> (pixel tiles, ids, labels)."""
    def __init__(self, rows, img_root, proc, enc, bos, end_id, pad, n_img, max_len):
        self.rows, self.img_root, self.proc, self.enc = rows, img_root, proc, enc
        self.bos, self.end_id, self.pad, self.n_img, self.max_len = bos, end_id, pad, n_img, max_len
        self.head = [bos] + enc("<|user|>\n"); self.tail = enc("\n<|call|>\n")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        r = self.rows[i]
        try:
            im = Image.open(f"{self.img_root}/{r['image']}").convert("RGB")
        except Exception:
            im = Image.new("RGB", (TILE, TILE), (255, 255, 255))
        px = self.proc(images=tiles_of(im), return_tensors="pt")["pixel_values"]          # (7,3,512,512)
        room = self.max_len - len(self.head) - self.n_img - len(self.tail) - 1
        body = self.enc(r["text"])[:room] + [self.end_id]
        ids = self.head + [self.pad] * self.n_img + self.tail + body
        lab = [-100] * (len(ids) - len(body)) + body
        return px, torch.tensor(ids), torch.tensor(lab)


def collate(batch, pad):
    import torch
    px = torch.stack([b[0] for b in batch]); L = max(len(b[1]) for b in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long); lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for j, (_, i_, l_) in enumerate(batch):
        ids[j, :len(i_)] = i_; lab[j, :len(l_)] = l_; att[j, :len(i_)] = 1
    return px, ids, lab, att


def tok_of():
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    p = f"{V}/tok/songgot-vocab.gguf"
    if not os.path.exists(p):
        p = hf_hub_download("palette-lab/songgot", "songgot-vocab.gguf")
    t = Llama(model_path=p, vocab_only=True, verbose=False)
    return lambda s: t.tokenize(s.encode("utf-8"), add_bos=False, special=True), t.token_bos()


@app.function(image=image, gpu="H100", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 20, memory=131072, cpu=16)
def align(lm: str = "ckpt/sftv8e3e3/final", pairs: str = "corpora/ko_vdr_pages.jsonl", img_root: str = "corpora/ko_vdr_pages",
          steps: int = 3000, batch: int = 16, lr: float = 1e-3, max_len: int = 1024, out: str = "v2/align", seed: int = 0,
          log_every: int = 50, workers: int = 12):
    import torch
    from functools import partial
    from transformers import SiglipImageProcessor
    vol.reload()
    enc, bos = tok_of(); end_id = 3 + SPECIAL.index("<|end|>"); pad = 3 + SPECIAL.index("<|pad|>")
    model = build(f"{V}/{lm}").cuda()
    for p_ in list(model.tower.parameters()) + list(model.lm.parameters()):
        p_.requires_grad_(False)
    params = list(model.proj.parameters())
    proc = SiglipImageProcessor.from_pretrained(TOWER)
    rows = [json.loads(l) for l in open(f"{V}/{pairs}", encoding="utf-8")]
    rows = [r for r in rows if r.get("text") and not r.get("dup_page")]
    random.Random(seed).shuffle(rows)
    ds = Pairs(rows, f"{V}/{img_root}", proc, enc, bos, end_id, pad, model.n_img, max_len)
    slot = len(ds.head)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, collate_fn=partial(collate, pad=pad), drop_last=True, persistent_workers=True)
    log = open(f"{V}/songgot_v.log", "a")
    def say(m):
        print(m, flush=True); log.write(time.strftime("%F %T ") + m + "\n"); log.flush()
    say(f"[v2] align: {len(rows)} pages, {sum(p.numel() for p in params)/1e6:.1f}M trainable, {model.n_img} image tokens per page, tower {TOWER}")
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.03)
    step = 0; t0 = time.time(); model.train()
    while step < steps:
        for px, ids, lab, att in dl:
            px, ids, lab, att = px.cuda(), ids.cuda(), lab.cuda(), att.cuda()
            loss = model(px, ids, lab, att, slot).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % log_every == 0 or step == steps - 1:
                with torch.no_grad():   # image gain: same batch, images rolled by one sample
                    wrong = model(px.roll(1, 0), ids, lab, att, slot).loss.item()
                say(f"[v2] step {step}/{steps} loss {loss.item():.4f} shuffled {wrong:.4f} image_gain {wrong - loss.item():+.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/60:.1f}m")
            if step and step % 1000 == 0:
                os.makedirs(f"{V}/ckpt/{out}", exist_ok=True); torch.save({"proj": model.proj.state_dict(), "step": step}, f"{V}/ckpt/{out}/proj.pt"); vol.commit()
            step += 1
            if step >= steps:
                break
    os.makedirs(f"{V}/ckpt/{out}", exist_ok=True)
    torch.save({"proj": model.proj.state_dict(), "step": steps}, f"{V}/ckpt/{out}/proj.pt")
    json.dump({"lm": lm, "tower": TOWER, "n_img": model.n_img, "tiles": 7, "pairs": pairs, "steps": steps}, open(f"{V}/ckpt/{out}/config.json", "w"))
    vol.commit(); say("[v2] align DONE")
    return {"steps": steps}
