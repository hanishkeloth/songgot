"""Deterministic scoring on Kakao FunctionChat-Bench SingleCall (Apache 2.0, Korean, 2024).

The original benchmark scores with GPT-4 as a judge. We score the "call" task with exact match
instead, so anyone can reproduce the number without an API key:

  * function name must match the ground truth;
  * arguments must equal the ground-truth arguments after normalisation (whitespace trimmed,
    numbers compared as numbers, strings compared exactly), or equal one of the entries in
    `acceptable_arguments` when the benchmark lists alternatives.

500 items = 25 functions x 4 queries x 5 tool conditions (exact, 4_random, 4_close, 8_random,
8_close). We report overall accuracy and per condition, and name-only accuracy as a secondary
number. This is stricter than the judge on paraphrased argument values and more lenient on
nothing; it is stated in the paper as our reading of the benchmark, not the official one.

    python eval/functionchat_exact.py predict --model <hf dir or id> --backend songgot|qwen|gemma|needle --out preds.jsonl
    python eval/functionchat_exact.py score preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "FunctionChat-Singlecall.jsonl"
CONDITIONS = ["exact", "4_random", "4_close", "8_random", "8_close"]


def items():
    """Flatten the benchmark into 500 items."""
    out = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        tools_by = {t["type"]: t["content"] for t in r["tools"]}
        gts = {g["serial_num"]: g["content"] for g in r["ground_truth"]}
        acc = {a["serial_num"]: a["content"] for a in r.get("acceptable_arguments", [])}
        for q in r["query"]:
            for cond in CONDITIONS:
                out.append({
                    "id": f"{r['function_name']}#{q['serial_num']}#{cond}",
                    "function": r["function_name"], "condition": cond, "query": q["content"],
                    "tools": tools_by[cond], "ground_truth": gts[q["serial_num"]], "acceptable": acc.get(q["serial_num"]),
                })
    return out


def _norm(v):
    if isinstance(v, str):
        s = v.strip()
        try:
            f = float(s.replace(",", ""))
            return f if not f.is_integer() else int(f)
        except ValueError:
            return s
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def parse_call(text: str):
    """Find the first {...} JSON object with a name; tolerate wrappers like <tool_call> or code fences."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, s[: s.rfind("}") + 1]):
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                if "name" in d:
                    args = d.get("arguments", d.get("parameters", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            return {"name": d["name"], "arguments": None}
                    return {"name": d["name"], "arguments": args if isinstance(args, dict) else {}}
                if "function" in d and isinstance(d["function"], dict):
                    return parse_call(json.dumps(d["function"], ensure_ascii=False))
        except json.JSONDecodeError:
            # trim trailing junk one brace at a time
            depth = 0
            for i, ch in enumerate(cand):
                depth += ch == "{"; depth -= ch == "}"
                if depth == 0 and i > 0:
                    try:
                        d = json.loads(cand[: i + 1])
                        return parse_call(json.dumps(d, ensure_ascii=False))
                    except json.JSONDecodeError:
                        break
    return None


def gt_options(item):
    g = json.loads(item["ground_truth"])
    name = g["name"]
    args = g.get("arguments", {})
    args = json.loads(args) if isinstance(args, str) and args.strip() else (args or {})
    opts = [_norm(args)]
    acc = item.get("acceptable")
    if acc:
        try:
            a = json.loads(acc) if isinstance(acc, str) else acc
            cands = a if isinstance(a, list) else [a]
            for c in cands:
                if isinstance(c, dict):
                    opts.append(_norm(c.get("arguments", c) if "name" in c else c))
        except json.JSONDecodeError:
            pass
    return name, opts


def score(pred_path: str):
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(pred_path, encoding="utf-8")}
    rows = items()
    tot = {c: [0, 0, 0] for c in CONDITIONS}  # [n, name_ok, full_ok]
    fails = []
    for it in rows:
        p = preds.get(it["id"], {})
        call = parse_call(p.get("output", ""))
        name, opts = gt_options(it)
        n_ok = bool(call) and call["name"] == name
        f_ok = n_ok and call.get("arguments") is not None and _norm(call["arguments"]) in opts
        t = tot[it["condition"]]; t[0] += 1; t[1] += n_ok; t[2] += f_ok
        if not f_ok and len(fails) < 40:
            fails.append({"id": it["id"], "q": it["query"], "gt": it["ground_truth"], "out": (p.get("output") or "")[:160]})
    n = sum(t[0] for t in tot.values()); nm = sum(t[1] for t in tot.values()); fl = sum(t[2] for t in tot.values())
    res = {"n": n, "call_acc": round(fl / n, 4), "name_acc": round(nm / n, 4),
           "by_condition": {c: {"n": t[0], "call_acc": round(t[2] / max(1, t[0]), 4), "name_acc": round(t[1] / max(1, t[0]), 4)} for c, t in tot.items()},
           "fails_sample": fails[:12]}
    return res


# ------------------------------------------------------------------ predictors
def songgot_prompt(item, tools=None):
    tools = tools if tools is not None else [t["function"] for t in item["tools"]]
    return "<|system|>\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":")) + f"\n<|user|>\n{item['query']}\n<|call|>\n"


def predict_songgot(model_dir: str, rows, device="cpu", max_new=160, emit=None):
    """Greedy decoding with the HF weights; tokenization is llama.cpp's (harness/songgot_tok.py), the
    same ids the app and every GGUF user produce, so this number describes the shipped path."""
    import sys, torch
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "harness"))
    from songgot_tok import Tok
    tok = Tok()
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    end_id, pad_id, eos_id = tok.end_id, tok.pad_id, tok.eos_id
    outs = []
    for i, it in enumerate(rows):
        ids = torch.tensor([tok.encode(songgot_prompt(it), bos=True)]).to(device)
        with torch.no_grad():
            g = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False, eos_token_id=[end_id, eos_id], pad_token_id=pad_id)
        text = tok.decode([int(x) for x in g[0][ids.shape[1]:].tolist() if int(x) not in (end_id, eos_id, pad_id)])
        outs.append({"id": it["id"], "output": text.strip()})
        emit(outs[-1]) if emit else None
        if i % 50 == 0:
            print(f"[songgot] {i}/{len(rows)} {text.strip()[:90]}", flush=True)
    return outs


def predict_qwen(model_id: str, rows, device="cpu", max_new=200, emit=None):
    """Qwen3 uses its own chat template with tools; parse <tool_call>...</tool_call>. Thinking disabled."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id); model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device).eval()
    outs = []
    for i, it in enumerate(rows):
        tools = [{"type": "function", "function": t["function"]} for t in it["tools"]]
        text = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        out = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"<tool_call>(.*?)</tool_call>", out, re.S)
        outs.append({"id": it["id"], "output": (m.group(1) if m else out).strip()})
        emit(outs[-1]) if emit else None
        if i % 50 == 0:
            print(f"[qwen] {i}/{len(rows)} {outs[-1]['output'][:90]}", flush=True)
    return outs


def predict_gemma(model_id: str, rows, device="cpu", max_new=200, emit=None):
    """FunctionGemma: its chat template accepts tools; the call comes back as
    <start_function_call>call:NAME{ARGS}<end_function_call> or a JSON block depending on version."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id); model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device).eval()
    outs = []
    for i, it in enumerate(rows):
        tools = [{"type": "function", "function": t["function"]} for t in it["tools"]]
        try:
            text = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "Tools:\n" + json.dumps(tools, ensure_ascii=False) + "\nUser: " + it["query"] + "\nCall:"
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        out = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        m = re.search(r"call:\s*([A-Za-z0-9_\.]+)\s*(\{.*?\})", out, re.S)
        if m:
            try:
                args = json.loads(m.group(2))
            except json.JSONDecodeError:
                args = {}
            out = json.dumps({"name": m.group(1), "arguments": args}, ensure_ascii=False)
        outs.append({"id": it["id"], "output": out.strip()})
        emit(outs[-1]) if emit else None
        if i % 50 == 0:
            print(f"[gemma] {i}/{len(rows)} {outs[-1]['output'][:90]}", flush=True)
    return outs


def predict_needle(rows, emit=None):
    """Needle 2 (cactus-needle) with the benchmark's tools passed as OpenAI-style schemas."""
    import os
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    import needle
    outs = []
    system = "Current date and time: 2026-09-10 09:00 (Thursday), Asia/Seoul."
    for i, it in enumerate(rows):
        tools = [t["function"] for t in it["tools"]]
        try:
            n = needle.Needle(tools=tools, system=system)
            res = n.run(it["query"], strict=False)
            calls = res.get("function_calls") or []
            if calls:
                c = calls[0]
                name = c.get("name") or c.get("function") or ""
                args = c.get("arguments", c.get("args", {}))
                text = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            else:
                text = json.dumps({"name": "none", "arguments": {}, "needle": {k: res.get(k) for k in ("type", "reasoning", "confidence")}}, ensure_ascii=False)
        except Exception as e:
            text = f"ERROR {e}"
        outs.append({"id": it["id"], "output": text})
        emit(outs[-1]) if emit else None
        if i % 50 == 0:
            print(f"[needle] {i}/{len(rows)} {text[:90]}", flush=True)
    return outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "score", "list"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--model"); ap.add_argument("--backend", default="songgot"); ap.add_argument("--out"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = items()
    if a.limit:
        rows = rows[: a.limit]
    if a.cmd == "list":
        print(len(rows), "items;", json.dumps(rows[0], ensure_ascii=False)[:400])
    elif a.cmd == "predict":
        t0 = time.time()
        done = set()
        if pathlib.Path(a.out).exists():
            done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8") if l.strip()}
        todo = [r for r in rows if r["id"] not in done]
        print(f"[predict] {len(done)} done, {len(todo)} to go", flush=True)
        sink = open(a.out, "a", encoding="utf-8")
        def emit(o):
            sink.write(json.dumps(o, ensure_ascii=False) + "\n"); sink.flush()
        fn = {"songgot": lambda: predict_songgot(a.model, todo, a.device, emit=emit), "qwen": lambda: predict_qwen(a.model, todo, a.device, emit=emit),
              "gemma": lambda: predict_gemma(a.model, todo, a.device, emit=emit), "needle": lambda: predict_needle(todo, emit=emit)}[a.backend]
        fn(); sink.close()
        print(f"wrote {a.out} in {time.time()-t0:.0f}s")
        print(json.dumps(score(a.out), ensure_ascii=False, indent=1)[:1500])
    else:
        print(json.dumps(score(a.path), ensure_ascii=False, indent=1))
