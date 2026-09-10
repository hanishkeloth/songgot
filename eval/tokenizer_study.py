"""Tokens per Hangul syllable on the 100 FunctionChat SingleCall queries, for Songgot's tokenizer
versus the tokenizers of the comparators. Lower is better: at a 256 to 1024 token budget, every
extra token per syllable is context lost to the tool schemas."""
import json, pathlib, re, sys
sys.path.insert(0, "harness")
HERE = pathlib.Path(__file__).resolve().parent
rows = [json.loads(l) for l in (HERE / "FunctionChat-Singlecall.jsonl").read_text(encoding="utf-8").splitlines()]
queries = [q["content"] for r in rows for q in r["query"]]
syl = sum(len(re.findall(r"[가-힣]", q)) for q in queries); chars = sum(len(q) for q in queries)
out = {}
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="vol/tok/spm.model")
out["Songgot 32k (ours)"] = sum(len(sp.encode(q)) for q in queries)
from transformers import AutoTokenizer
for name, mid in [("Qwen3 (151k)", "Qwen/Qwen3-0.6B"), ("Gemma 3 / FunctionGemma (262k)", "unsloth/functiongemma-270m-it")]:
    try:
        t = AutoTokenizer.from_pretrained(mid); out[name] = sum(len(t(q, add_special_tokens=False)["input_ids"]) for q in queries)
    except Exception as e:
        out[name] = f"ERR {str(e)[:60]}"
try:
    import needle, inspect, os
    pkg = pathlib.Path(needle.__file__).parent
    cands = list(pkg.rglob("*token*")) + list(pkg.rglob("*.model")) + list(pkg.rglob("*.json"))
    out["needle_tokenizer_files"] = [str(c.relative_to(pkg)) for c in cands][:8]
except Exception as e:
    out["needle_tokenizer_files"] = f"ERR {e}"
print(f"queries {len(queries)}, Hangul syllables {syl}, chars {chars}")
for k, v in out.items():
    if isinstance(v, int):
        print(f"{k:34s} tokens {v:5d}  tokens/syllable {v/syl:.2f}  tokens/char {v/chars:.2f}")
    else:
        print(k, v)
