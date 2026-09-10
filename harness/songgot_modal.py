"""Songgot (송곳): Korean-first tiny agentic model, trained from scratch on Modal.

    modal run harness/songgot_modal.py::prep          # corpus + tokenizer + token shards (CPU)
    modal run harness/songgot_modal.py::pretrain      # 8x H100, DDP, ~6B tokens
    modal run harness/songgot_modal.py::sft           # tool-calling post-training
    modal run harness/songgot_modal.py::export_gguf   # HF dir + GGUF f16/q8_0/q4_k_m on the volume

Data (all licence-clean, all disclosed in the paper): fineweb-edu sample-10BT (ODC-By) for
English, Korean Wikipedia 20231101.ko (CC BY-SA 3.0) for Korean, and synthetic Korean agentic
data produced by our own Palette-K-Midm (harness/teacher_gen.py). No AI-Hub bytes, no closed
model outputs. FunctionChat-Bench is test only.
"""
from __future__ import annotations

import json
import os
import time

import modal

APP = "songgot"
app = modal.App(APP)
vol = modal.Volume.from_name("songgot", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
V = os.environ.get("SONGGOT_VOL", "/vol")   # local fallback: SONGGOT_VOL=~/Desktop/SONGGOT/vol python -c "...prep.local()"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
        "torch==2.6.0", "transformers==4.51.3", "sentencepiece>=0.2.0", "datasets==3.6.0",
        "huggingface_hub[hf_transfer]", "numpy<2.3", "safetensors", "tqdm", "gguf",
    )
    .pip_install("llama-cpp-python", extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)

SPECIAL = ["<|system|>", "<|user|>", "<|call|>", "<|end|>", "<|pad|>"]
VOCAB = 32000
SEQ = 1024


def _tok_chunk(args):
    """Tokenize one byte range of a text file to a uint16 shard. Runs in a worker process."""
    import numpy as np
    import sentencepiece as spm
    path, start, end, out, model = args
    sp_ = spm.SentencePieceProcessor(model_file=model)
    ids = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        f.seek(start); buf = f.read(end - start)
    for doc in buf.split("\n\n"):
        doc = doc.strip()
        if not doc:
            continue
        ids.extend(sp_.encode(doc)); ids.append(2)
    arr = np.array(ids, dtype=np.uint16); arr.tofile(out)
    return out, len(arr)


# ----------------------------------------------------------------------------- corpus + tokenizer
@app.function(image=image, volumes={V: vol, "/root/.cache/huggingface": hf_cache},
              cpu=32, memory=131072, timeout=60 * 60 * 6)
def prep(en_tokens_target: float = 3.2e9, ko_repeat: int = 1, tok_sample_mb: int = 300):
    """Stream fineweb-edu and Korean Wikipedia to /vol/raw, train the tokenizer, write uint16 shards."""
    import numpy as np
    import sentencepiece as spm
    from datasets import load_dataset
    from multiprocessing import Pool

    raw = f"{V}/raw"; tokd = f"{V}/tok"; os.makedirs(raw, exist_ok=True); os.makedirs(tokd, exist_ok=True)
    log = open(f"{V}/prep.log", "a")
    def say(m):
        print(m, flush=True); log.write(time.strftime("%F %T ") + m + "\n"); log.flush()

    # 1) Korean Wikipedia
    ko_path = f"{raw}/ko.txt"
    if not os.path.exists(ko_path + ".done"):
        say("kowiki: loading")
        ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
        n = 0
        with open(ko_path, "w", encoding="utf-8") as f:
            for r in ds:
                t = (r.get("text") or "").strip()
                if len(t) < 200:
                    continue
                f.write(t + "\n\n"); n += len(t)
        open(ko_path + ".done", "w").write(str(n))
        say(f"kowiki: {n/1e6:.0f}M chars")
    ko_chars = int(open(ko_path + ".done").read())

    # 2) fineweb-edu, streamed until the char budget (approx 4.3 chars per token for English)
    en_path = f"{raw}/en.txt"
    en_char_target = int(en_tokens_target * 4.3)
    if not os.path.exists(en_path + ".done"):
        say(f"fineweb-edu: streaming to {en_char_target/1e9:.1f}G chars")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        n = 0; t0 = time.time()
        with open(en_path, "w", encoding="utf-8") as f:
            for i, r in enumerate(ds):
                t = r["text"].strip()
                f.write(t + "\n\n"); n += len(t)
                if i % 200000 == 0:
                    say(f"fineweb-edu: {i} docs {n/1e9:.2f}G chars {time.time()-t0:.0f}s")
                if n >= en_char_target:
                    break
        open(en_path + ".done", "w").write(str(n))
        say(f"fineweb-edu: {n/1e9:.2f}G chars")

    # 3) tokenizer sample: Korean-heavy, plus tool-schema JSON so the vocabulary knows braces,
    #    snake_case names and Korean parameter descriptions
    sample = f"{raw}/tok_sample.txt"
    if not os.path.exists(f"{tokd}/spm.model"):
        say("tokenizer: sampling")
        with open(sample, "w", encoding="utf-8") as out:
            for path, mb in ((ko_path, int(tok_sample_mb * 0.5)), (en_path, int(tok_sample_mb * 0.4))):
                with open(path, encoding="utf-8") as f:
                    out.write(f.read(mb * 1_000_000))
            tools = json.load(open(f"{V}/data/tools_ko.json", encoding="utf-8"))
            for _ in range(200):
                for t in tools:
                    out.write(json.dumps({"name": t["name"], "description": t["description"], "parameters": t["parameters"]}, ensure_ascii=False) + "\n")
                    out.write(json.dumps({"name": t["name"], "arguments": {k: "값" for k in t["parameters"].get("properties", {})}}, ensure_ascii=False) + "\n")
        say("tokenizer: training spm (bpe, 32000, byte fallback)")
        spm.SentencePieceTrainer.train(
            input=sample, model_prefix=f"{tokd}/spm", vocab_size=VOCAB, model_type="bpe",
            character_coverage=0.9999, byte_fallback=True, split_digits=True,
            allow_whitespace_only_pieces=True, remove_extra_whitespaces=False,
            user_defined_symbols=SPECIAL, num_threads=32, input_sentence_size=6_000_000,
            shuffle_input_sentence=True, max_sentence_length=8192, pad_id=-1, unk_id=0, bos_id=1, eos_id=2,
        )
        say("tokenizer: done")
    sp = spm.SentencePieceProcessor(model_file=f"{tokd}/spm.model")
    say(f"tokenizer: vocab {sp.get_piece_size()}; ' 안녕하세요, 반갑습니다' -> {sp.encode(' 안녕하세요, 반갑습니다')}")

    # 4) tokenize both corpora to uint16 shards, EOS between documents (module-level worker so
    #    macOS spawn can pickle it)
    for lang, path in (("ko", ko_path), ("en", en_path)):
        if os.path.exists(f"{tokd}/{lang}.done"):
            continue
        size = os.path.getsize(path); chunk = 200_000_000  # 200 MB of text per worker task
        jobs = []
        for k, start in enumerate(range(0, size, chunk)):
            jobs.append((path, start, min(size, start + chunk), f"{tokd}/{lang}_{k:04d}.bin", f"{tokd}/spm.model"))
        say(f"tokenize {lang}: {len(jobs)} chunks")
        total = 0
        with Pool(32) as pool:
            for out, n in pool.imap_unordered(_tok_chunk, jobs):
                total += n
        open(f"{tokd}/{lang}.done", "w").write(str(total))
        say(f"tokenize {lang}: {total/1e9:.2f}B tokens")
    vol.commit()
    say("prep: DONE")
    return {"ko_tokens": int(open(f"{tokd}/ko.done").read()), "en_tokens": int(open(f"{tokd}/en.done").read())}


def _tok_parquet(args):
    """Tokenize a row-group range of a parquet file to a uint16 shard, EOS between documents (worker process)."""
    import numpy as np
    import pyarrow.parquet as pq
    import sentencepiece as spm
    path, rg0, rg1, out, model = args
    sp_ = spm.SentencePieceProcessor(model_file=model)
    pf = pq.ParquetFile(path); n = 0; buf = []
    with open(out, "wb") as f:
        for rg in range(rg0, rg1):
            texts = [t.strip() for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist() if t]
            texts = [t for t in texts if len(t) >= 200]
            for ids in sp_.encode(texts):
                buf.extend(ids); buf.append(2)
            if len(buf) >= 4_000_000:
                np.array(buf, dtype=np.uint16).tofile(f); n += len(buf); buf = []
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f); n += len(buf)
    return out, n


@app.function(image=image, volumes={V: vol}, cpu=48, memory=262144, timeout=60 * 60 * 6, ephemeral_disk=524288)
def prep2(ko_files: int = 4, en_files: int = 17, tokd_name: str = "tok2", workers: int = 48):
    """Corpus v2 for the 12-layer model: FineWeb-2 Korean (kor_Hang, ODC-By) and fresh fineweb-edu sample-100BT
    files (ODC-By), tokenized with the existing tokenizer (tok/spm.model) straight from parquet into /vol/tok2
    shards; the Korean Wikipedia shards from corpus v1 are copied in. Corpus v1 had 0.6B Korean tokens, which
    the 6B-token run sampled about 4.5 times each; this one targets about 12B Korean and 12B English."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import Pool
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    tokd = f"{V}/{tokd_name}"; os.makedirs(tokd, exist_ok=True); model = f"{V}/tok/spm.model"
    log = open(f"{V}/prep2.log", "a")
    def say(m):
        print(m, flush=True); log.write(time.strftime("%F %T ") + m + "\n"); log.flush()
    api = HfApi(); plan = []
    for lang, repo, prefix, k in (("ko", "HuggingFaceFW/fineweb-2", "data/kor_Hang/train", ko_files),
                                  ("en", "HuggingFaceFW/fineweb-edu", "sample/100BT", en_files)):
        files = sorted(f.path for f in api.list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset") if f.path.endswith(".parquet"))[:k]
        plan += [(lang, repo, fp, i) for i, fp in enumerate(files)]
    say(f"prep2: {len(plan)} parquet files ({ko_files} Korean, {en_files} English)")
    dl = ThreadPoolExecutor(1)
    fetch = lambda item: hf_hub_download(item[1], item[2], repo_type="dataset", local_dir="/tmp/pq")
    pending = [it for it in plan if not os.path.exists(f"{tokd}/{it[0]}_{it[3]:03d}.done")]
    fut = dl.submit(fetch, pending[0]) if pending else None
    for j, (lang, repo, fp, i) in enumerate(pending):
        t0 = time.time(); local = fut.result()
        fut = dl.submit(fetch, pending[j + 1]) if j + 1 < len(pending) else None
        nrg = pq.ParquetFile(local).num_row_groups; per = max(1, nrg // (2 * workers))
        jobs = [(local, a, min(nrg, a + per), f"{tokd}/{lang}_{i:03d}_{a:04d}.bin", model) for a in range(0, nrg, per)]
        with Pool(workers) as pool:
            total = sum(n for _, n in pool.imap_unordered(_tok_parquet, jobs))
        open(f"{tokd}/{lang}_{i:03d}.done", "w").write(str(total)); os.remove(local); vol.commit()
        say(f"prep2: {lang} file {i} ({fp.split('/')[-1]}, {nrg} row groups): {total/1e9:.2f}B tokens in {time.time()-t0:.0f}s")
    for f in sorted(os.listdir(f"{V}/tok")):
        if f.startswith("ko_") and f.endswith(".bin") and not os.path.exists(f"{tokd}/ko_wiki_{f[3:]}"):
            shutil.copy(f"{V}/tok/{f}", f"{tokd}/ko_wiki_{f[3:]}")
    vol.commit()
    tot = {}
    for lang in ("ko", "en"):
        tot[lang] = sum(int(open(f"{tokd}/{f}").read()) for f in os.listdir(tokd) if f.startswith(lang + "_") and f.endswith(".done"))
    tot["ko_wiki"] = int(open(f"{V}/tok/ko.done").read())
    say(f"prep2: DONE ko {tot['ko']/1e9:.2f}B + kowiki {tot['ko_wiki']/1e9:.2f}B, en {tot['en']/1e9:.2f}B")
    return tot


@app.function(image=image, volumes={V: vol}, cpu=8, memory=32768, timeout=60 * 60)
def prep_inst(data: str = "sft/train_v6.jsonl", tokd_name: str = "tok2", shard_tokens: int = 50_000_000):
    """Instruction bucket for pretraining: every post-training row rendered exactly as SFT renders it (prompt + call +
    <|end|>), tokenized with the llama.cpp vocab (same ids as SFT, the app and the GGUFs), EOS between rows."""
    import numpy as np
    from llama_cpp import Llama
    vol.reload()
    vocab = f"{V}/tok/songgot-vocab.gguf"
    tok = Llama(model_path=vocab, vocab_only=True, verbose=False)
    enc = lambda t: tok.tokenize(t.encode("utf-8"), add_bos=False, special=True)
    tokd = f"{V}/{tokd_name}"; buf = []; n = 0; k = 0
    for f in os.listdir(tokd):
        if f.startswith("inst_"):
            os.remove(f"{tokd}/{f}")
    def flush():
        nonlocal buf, k
        np.array(buf, dtype=np.uint16).tofile(f"{tokd}/inst_{k:04d}.bin"); k += 1; buf = []
    for l in open(f"{V}/{data}", encoding="utf-8"):
        r = json.loads(l); pr, co = render(r)
        buf.extend(enc(pr + co)); buf.append(2); n += 1
        if len(buf) >= shard_tokens:
            flush()
    if buf:
        flush()
    vol.commit(); total = sum(os.path.getsize(f"{tokd}/{f}") // 2 for f in os.listdir(tokd) if f.startswith("inst_"))
    print(f"[inst] {n} rows -> {k} shards, {total/1e6:.0f}M tokens in {tokd}", flush=True)
    return {"rows": n, "tokens": total}


# ----------------------------------------------------------------------------- model
def make_config(hidden: int = 512, inter: int = 1408, layers: int = 12, heads: int = 8, kv: int = 2):
    from transformers import LlamaConfig
    return LlamaConfig(vocab_size=VOCAB, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                       num_attention_heads=heads, num_key_value_heads=kv, max_position_embeddings=2048,
                       rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=True,
                       bos_token_id=1, eos_token_id=2, pad_token_id=None, attention_bias=False, mlp_bias=False)


def hf_tokenizer(tokd: str):
    """Wrap the SentencePiece model as a HF LlamaTokenizer with our special tokens."""
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer(vocab_file=f"{tokd}/spm.model", legacy=False, add_bos_token=True, add_eos_token=False)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL, "pad_token": "<|pad|>"})
    return tok


# ----------------------------------------------------------------------------- pretraining
def _ddp_worker(rank: int, world: int, cfg: dict):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import LlamaForCausalLM

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511", RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
    torch.manual_seed(1234 + rank); np.random.seed(1234 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokd = f"{V}/{cfg.get('tokd', 'tok')}"  # tok = 6B-token corpus v1, tok2 = FineWeb-2 Korean + fineweb-edu 100BT
    shards = {"ko": sorted(f for f in os.listdir(tokd) if f.startswith("ko_") and f.endswith(".bin")),
              "en": sorted(f for f in os.listdir(tokd) if f.startswith("en_") and f.endswith(".bin")),
              "inst": sorted(f for f in os.listdir(tokd) if f.startswith("inst_") and f.endswith(".bin"))}
    p_inst = cfg.get("p_inst", 0.0) if shards["inst"] else 0.0
    if not shards["inst"]:
        del shards["inst"]
    # each rank owns every world-th shard and reads it into RAM once: random 2 KB reads through a memmap on the
    # network volume ran at 0.01M tok/s on corpus v2 (about 1,000 shards, 49 GB); sequential reads are fast
    mm = {l: [np.fromfile(f"{tokd}/{f}", dtype=np.uint16) for f in fs[rank::world]] for l, fs in shards.items()}
    if rank == 0:
        print(f"[pre] rank 0 holds {sum(len(m) for ms in mm.values() for m in ms)/1e9:.2f}B tokens of {len(shards['ko'])+len(shards['en'])} shards", flush=True)
    wts = {l: np.array([len(m) for m in ms], dtype=np.float64) for l, ms in mm.items()}
    for l in wts:
        wts[l] /= wts[l].sum()
    p_ko = cfg["p_ko"]; B = cfg["batch_per_gpu"]; T = SEQ

    def batch():
        x = np.empty((B, T + 1), dtype=np.int64)
        for i in range(B):
            lang = "inst" if np.random.rand() < p_inst else ("ko" if np.random.rand() < p_ko else "en")
            m = mm[lang][np.random.choice(len(mm[lang]), p=wts[lang])]
            off = np.random.randint(0, len(m) - T - 1)
            x[i] = m[off:off + T + 1].astype(np.int64)
        x = torch.from_numpy(x).to(dev, non_blocking=True)
        return x[:, :-1], x[:, 1:]

    model = LlamaForCausalLM(make_config(cfg.get("hidden", 512), cfg.get("inter", 1408), cfg.get("layers", 12), cfg.get("heads", 8), cfg.get("kv", 2))).to(dev)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"[pre] params {n/1e6:.1f}M, world {world}, tokens/step {world*B*T}", flush=True)
    model = DDP(model, device_ids=[rank])
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, fused=True)
    steps, warm, lr_max, lr_min = cfg["steps"], cfg["warmup"], cfg["lr"], cfg["lr"] * 0.1
    import math
    def lr_at(s):
        if s < warm:
            return lr_max * (s + 1) / warm
        r = (s - warm) / max(1, steps - warm)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * r))

    start = 0
    ck = f"{V}/ckpt/{cfg.get('tag', 'pre')}"
    os.makedirs(ck, exist_ok=True)
    if os.path.exists(f"{ck}/latest.pt"):
        st = torch.load(f"{ck}/latest.pt", map_location=dev)
        model.module.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"] + 1
        if rank == 0:
            print(f"[pre] resumed at step {start}", flush=True)

    model.train(); t0 = time.time(); tok_seen = 0; loss_acc = 0.0; n_acc = 0
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        loss_acc += out.loss.item(); n_acc += 1; tok_seen += world * B * T
        if rank == 0 and (step % 50 == 0 or step == steps - 1):
            el = time.time() - t0
            msg = f"[pre] step {step}/{steps} loss {loss_acc/n_acc:.4f} lr {lr_at(step):.2e} {tok_seen/max(el,1e-6)/1e6:.2f}M tok/s elapsed {el/60:.1f}m"
            print(msg, flush=True); open(f"{V}/pretrain.log", "a").write(time.strftime("%F %T ") + msg + "\n")
            loss_acc = 0.0; n_acc = 0
        if rank == 0 and step > 0 and (step % cfg["ckpt_every"] == 0 or step == steps - 1):
            torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "step": step}, f"{ck}/latest.pt")
            model.module.save_pretrained(f"{ck}/hf_step{step}", safe_serialization=True)
            vol.commit()
        dist.barrier() if step % cfg["ckpt_every"] == 0 else None
    if rank == 0:
        model.module.save_pretrained(f"{ck}/final", safe_serialization=True)
        import shutil; shutil.copy(f"{V}/tok/spm.model", f"{ck}/final/tokenizer.model")
        vol.commit(); print("[pre] DONE", flush=True)
    dist.destroy_process_group()


@app.function(image=image, volumes={V: vol}, gpu="H100:8", timeout=60 * 60 * 8, memory=262144)
def pretrain(tokens: float = 6.0e9, batch_per_gpu: int = 32, lr: float = 2e-3, p_ko: float = 0.45, ckpt_every: int = 2000,
             hidden: int = 512, inter: int = 1408, layers: int = 12, heads: int = 8, kv: int = 2, tag: str = "pre", tokd: str = "tok", p_inst: float = 0.0):
    import torch.multiprocessing as mp
    world = 8
    steps = int(tokens // (world * batch_per_gpu * SEQ))
    cfg = {"steps": steps, "warmup": min(1000, steps // 20), "lr": lr, "p_ko": p_ko, "batch_per_gpu": batch_per_gpu, "ckpt_every": ckpt_every,
           "hidden": hidden, "inter": inter, "layers": layers, "heads": heads, "kv": kv, "tag": tag, "tokd": tokd, "p_inst": p_inst}
    print(f"[pre] launching {world} ranks, {steps} steps", flush=True)
    mp.spawn(_ddp_worker, args=(world, cfg), nprocs=world, join=True)
    return {"steps": steps}


# ----------------------------------------------------------------------------- post-training (tool calling)
def render(example: dict) -> tuple[str, str]:
    """(prompt, completion) in Songgot's text format. Loss is taken on the completion only."""
    tools = json.dumps(example["tools"], ensure_ascii=False, separators=(",", ":"))
    prompt = f"<|system|>\n{tools}\n<|user|>\n{example['query']}\n<|call|>\n"
    completion = json.dumps(example["call"], ensure_ascii=False, separators=(",", ":")) + "<|end|>"
    return prompt, completion


@app.function(image=image, volumes={V: vol}, gpu="H100", timeout=60 * 60 * 3, memory=65536)
def sft(epochs: int = 2, lr: float = 3e-4, batch: int = 32, max_len: int = 1024, init: str = "pre/final", out: str = "sft/final", data: str = "sft/train.jsonl"):
    """Post-training with llama.cpp's tokenizer (vocab-only GGUF on the volume), the same ids the app,
    the evaluator and every GGUF user produce. See harness/songgot_tok.py for why."""
    import random
    import torch
    from transformers import LlamaForCausalLM
    from llama_cpp import Llama
    vol.reload()
    tokd = f"{V}/tok"; vocab_path = f"{tokd}/songgot-vocab.gguf"
    if not os.path.exists(vocab_path):  # volume view can lag a CLI upload; the same file is on the Hub
        from huggingface_hub import hf_hub_download
        vocab_path = hf_hub_download("palette-lab/songgot", "songgot-vocab.gguf")
        print(f"[sft] vocab from the Hub: {vocab_path}", flush=True)
    llm = Llama(model_path=vocab_path, vocab_only=True, verbose=False)

    class _SP:
        def encode(self, s): return llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)
        def bos_id(self): return llm.token_bos()
        def eos_id(self): return llm.token_eos()
    sp = _SP(); pad = 3 + SPECIAL.index("<|pad|>")
    rows = [json.loads(l) for l in open(f"{V}/{data}", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    print(f"[sft] {len(rows)} examples", flush=True)
    dev = torch.device("cuda")
    model = LlamaForCausalLM.from_pretrained(f"{V}/ckpt/{init}", torch_dtype=torch.bfloat16).to(dev)

    def encode(ex):
        p, c = render(ex)
        pi = [sp.bos_id()] + sp.encode(p); ci = sp.encode(c) + [sp.eos_id()]
        ids = (pi + ci)[:max_len]; lab = ([-100] * len(pi) + ci)[:max_len]
        return ids, lab
    enc = [encode(r) for r in rows]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    steps = epochs * (len(enc) // batch); import math; s = 0
    model.train()
    for ep in range(epochs):
        random.Random(ep).shuffle(enc)
        for i in range(0, len(enc) - batch + 1, batch):
            chunk = enc[i:i + batch]; L = max(len(x[0]) for x in chunk)
            ids = torch.full((batch, L), pad); lab = torch.full((batch, L), -100); att = torch.zeros((batch, L), dtype=torch.long)
            for j, (a, b) in enumerate(chunk):
                ids[j, :len(a)] = torch.tensor(a); lab[j, :len(b)] = torch.tensor(b); att[j, :len(a)] = 1
            ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (s + 1) / 100) * (0.5 * (1 + math.cos(math.pi * s / max(1, steps))))
            o = model(input_ids=ids, attention_mask=att, labels=lab)  # not `out`: that is the output directory name
            o.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            if s % 50 == 0:
                msg = f"[sft] ep {ep} step {s}/{steps} loss {o.loss.item():.4f}"
                print(msg, flush=True); open(f"{V}/sft.log", "a").write(time.strftime("%F %T ") + msg + "\n")
            s += 1
    out_dir = f"{V}/ckpt/{out}"; model.save_pretrained(out_dir, safe_serialization=True)
    import shutil; shutil.copy(f"{tokd}/spm.model", f"{out_dir}/tokenizer.model")
    json.dump({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<|pad|>", "unk_token": "<unk>", "additional_special_tokens": SPECIAL,
               "model_max_length": 2048, "tokenizer_class": "LlamaTokenizer", "legacy": False}, open(f"{out_dir}/tokenizer_config.json", "w"), indent=1)
    json.dump({sp_: 3 + i for i, sp_ in enumerate(SPECIAL)}, open(f"{out_dir}/added_tokens.json", "w"), indent=1)
    json.dump({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<|pad|>", "additional_special_tokens": SPECIAL},
              open(f"{out_dir}/special_tokens_map.json", "w"), indent=1)
    vol.commit(); print("[sft] DONE", flush=True)
    return {"steps": steps}


# ----------------------------------------------------------------------------- export
gguf_image = image.run_commands(
    "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
    "cd /opt/llama.cpp && cmake -B build -DGGML_NATIVE=OFF && cmake --build build --target llama-quantize -j 8",
    "pip install -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
)


@app.function(image=gguf_image, volumes={V: vol}, cpu=8, memory=32768, timeout=60 * 30)
def export_gguf(src: str = "sft/final", dst: str = "export", name: str = "songgot"):
    import subprocess
    d = f"{V}/ckpt/{src}"; out = f"{V}/{dst}"; os.makedirs(out, exist_ok=True)
    subprocess.run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", d, "--outfile", f"{out}/{name}-f16.gguf", "--outtype", "f16"], check=True)
    for q in ("Q8_0", "Q4_K_M"):
        subprocess.run(["/opt/llama.cpp/build/bin/llama-quantize", f"{out}/{name}-f16.gguf", f"{out}/{name}-{q.lower()}.gguf", q], check=True)
    vol.commit()
    return {f: os.path.getsize(f"{out}/{f}") for f in os.listdir(out)}
