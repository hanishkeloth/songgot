"""Korean VLM benchmarks for tiny models, one harness, same prompt for every model (each model's own chat
template carries the image): NCSOFT K-MMBench (dev), K-SEED (test), K-MMStar (val), K-DTCBench (test), all
CC BY-NC 4.0, multiple choice scored by exact letter. Greedy decoding, first standalone A-D letter in the
answer counts. Scoring is the next-token likelihood over the option letters (no prose parsing, fair to tiny
models that ramble). Results to /vol/kvlm/<tag>.json.

    modal run --detach harness/kvlm_modal.py --model Qwen/Qwen3.5-0.8B --tag qwen35_08b --limit 1000
"""
import json
import os
import re
import time

import modal

app = modal.App("songgot-kvlm")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V = "/vol"; CACHE = "/root/.cache/huggingface"
TV = os.environ.get("SONGGOT_TRANSFORMERS", "5.17.0")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", f"transformers=={TV}", "huggingface_hub[hf_transfer]", "numpy<2.3", "safetensors", "accelerate", "pillow", "datasets", "num2words")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)
BENCH = {"kmmbench": ("NCSOFT/K-MMBench", "dev"), "kseed": ("NCSOFT/K-SEED", "test"), "kmmstar": ("NCSOFT/K-MMStar", "val"), "kdtcbench": ("NCSOFT/K-DTCBench", "test")}


def rows_of(name, limit):
    from datasets import load_dataset
    repo, split = BENCH[name]; out = []
    for r in load_dataset(repo, split=split, streaming=True):
        if name == "kmmbench":
            opts = [(k, r[k]) for k in "ABCD" if r.get(k) not in (None, "None", "nan", "")]
            q = (r.get("hint") + "\n" if r.get("hint") not in (None, "None", "nan", "") else "") + r["question"]
        elif name in ("kseed", "kdtcbench"):
            opts = [(k.upper(), r[f"choice_{k}"]) for k in "abcd" if r.get(f"choice_{k}") not in (None, "None", "")]
            q = r["question"]
        else:  # kmmstar: options are inside the question text
            opts = []; q = r["question"].replace("<image>", "").strip()
        text = q + ("\n" + "\n".join(f"{k}. {v}" for k, v in opts) if opts else "") + "\n정답 선택지의 알파벳 하나만 답하세요."
        out.append({"id": str(r.get("index", r.get("question_id", len(out)))), "image": r["image"].convert("RGB"), "text": text, "answer": str(r["answer"]).strip().upper()[:1],
                    "category": str(r.get("category", "")), "opts": [k for k, _ in opts] or None})
        if limit and len(out) >= limit:
            break
    return out


def opts_of(r):
    return [(k, 1) for k in r["opts"]] if r.get("opts") else None


def pick_letter(s: str):
    m = re.search(r"\b([ABCD])\b", s.strip().upper())
    if m:
        return m.group(1)
    m = re.match(r"\s*\(?([ABCD])[\).:\s]", s.strip().upper())
    return m.group(1) if m else None


@app.function(image=image, gpu="H100", volumes={V: vol, CACHE: hf_cache}, timeout=60 * 60 * 3, memory=65536)
def run(model: str = "Qwen/Qwen3.5-0.8B", tag: str = "qwen35_08b", limit: int = 1000, benches: str = "kmmbench,kseed,kmmstar,kdtcbench", max_new: int = 8):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    vol.reload(); os.makedirs(f"{V}/kvlm", exist_ok=True)
    path = f"{V}/ckpt/{model}" if os.path.exists(f"{V}/ckpt/{model}") else model
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    tk = getattr(proc, "tokenizer", proc)
    letter_ids = {L: sorted({i for v in (L, " " + L, "(" + L) for i in [tk.encode(v, add_special_tokens=False)[-1]]}) for L in "ABCD"}
    log = open(f"{V}/kvlm.log", "a")
    def say(s):
        print(s, flush=True); log.write(time.strftime("%F %T ") + s + "\n"); log.flush()
    res = {"model": model, "tag": tag, "limit": limit, "date": time.strftime("%F"), "scoring": "next-token likelihood over the option letters", "bench": {}}
    for b in benches.split(","):
        rows = rows_of(b, limit); t0 = time.time(); ok = 0; noletter = 0; per_cat = {}
        for i, r in enumerate(rows):
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["text"]}]}]
            prompt = None
            for kw in ({"enable_thinking": False}, {}):
                for tpl in (proc, getattr(proc, "tokenizer", None)):
                    if tpl is None or prompt is not None:
                        continue
                    try:
                        prompt = tpl.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
                    except Exception:
                        prompt = None
            inputs = proc(text=[prompt], images=[r["image"]], return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = m(**inputs).logits[0, -1].float()
            # letter chosen by next-token likelihood over A-D (both bare and space-prefixed spellings), no parsing of prose
            letters = [k for k, _ in (opts_of(r) or [("A", 1), ("B", 1), ("C", 1), ("D", 1)])]
            score = {L: max(logits[i].item() for i in letter_ids[L]) for L in letters}
            L = max(score, key=score.get); out = L
            hit = L == r["answer"]; ok += hit
            c = per_cat.setdefault(r["category"], [0, 0]); c[0] += hit; c[1] += 1
            if i % 200 == 0:
                say(f"[kvlm] {tag} {b} {i}/{len(rows)} acc {ok/max(1,i+1):.3f} out={out.strip()[:30]!r}")
        res["bench"][b] = {"n": len(rows), "acc": ok / max(1, len(rows)), "no_letter": noletter, "sec": round(time.time() - t0), "by_category": {k: round(v[0] / v[1], 3) for k, v in per_cat.items()}}
        say(f"[kvlm] {tag} {b}: acc {ok/max(1,len(rows)):.3f} n {len(rows)} no_letter {noletter} in {time.time()-t0:.0f}s")
    json.dump(res, open(f"{V}/kvlm/{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); vol.commit()
    say(f"[kvlm] {tag} DONE " + " ".join(f"{b} {v['acc']:.3f}" for b, v in res["bench"].items()))
    return {b: v["acc"] for b, v in res["bench"].items()}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-0.8B", tag: str = "qwen35_08b", limit: int = 1000, benches: str = "kmmbench,kseed,kmmstar,kdtcbench"):
    print(run.remote(model, tag, limit, benches))
