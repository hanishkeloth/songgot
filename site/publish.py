"""Publish a scored Songgot-nano checkpoint: fill the model card row from the score JSON, rebuild the
site (tables and figures read the same JSON), print the PDF, commit and push, upload weights to the
Hub and restart the demo Space. Usage: publish.py <ckpt_dir> <score_json> <label>"""
import argparse, json, pathlib, subprocess, sys, datetime
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("score"); ap.add_argument("label")
ap.add_argument("--repo", default="palette-lab/songgot"); ap.add_argument("--name", default="Songgot-nano")
ap.add_argument("--params", default="39M"); ap.add_argument("--gguf", default="songgot-nano"); ap.add_argument("--desc", default="8 layers, 39M parameters, pretrained on an Apple M5 Max with MLX on 320M tokens")
A = ap.parse_args(); ckpt, score_path, label = pathlib.Path(A.ckpt), pathlib.Path(A.score), A.label
r = json.loads(score_path.read_text()); bc = r["by_condition"]; C = ["exact", "4_random", "4_close", "8_random", "8_close"]
row = f"| {A.name} ({label}) | {A.params} | " + " | ".join(f"{bc[c]['call_acc']*100:.1f}" for c in C) + f" | {r['call_acc']*100:.1f} | {r['name_acc']*100:.1f} |"
card = ROOT / "MODEL_CARD.md"; t = card.read_text()
lines = t.splitlines(); out = []; done = False
for ln in lines:
    if ln.startswith(f"| {A.name} ") and not done:
        out.append(row); done = True
    else:
        out.append(ln)
if not done:  # new model row goes right after the header separator of the results table
    for i, ln in enumerate(out):
        if ln.startswith("|---"):
            out.insert(i + 1, row); break
t = "\n".join(out) + "\n"
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
status = (f"## Status ({stamp})\nWeights in this repo are {A.name}, {label}: {A.desc}, post-trained on the v2 tool-calling set. "
          f"Call accuracy on FunctionChat-Bench SingleCall {r['call_acc']*100:.1f} percent (name only {r['name_acc']*100:.1f}). "
          f"GGUF exports (f16, Q8_0, Q4_K_M) are in this repo.\n")
import re
t = re.sub(r"## Status \([^)]*\)\n.*?(?=\n## |\Z)", status.rstrip("\n"), t, count=1, flags=re.S)
card.write_text(t)
subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "site/build.py")], check=True, cwd=ROOT)
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={ROOT/'docs/songgot.pdf'}", "--virtual-time-budget=5000", f"file://{ROOT/'docs/index.html'}"], cwd=ROOT, timeout=120)
subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
subprocess.run(["git", "-c", "user.name=Hanish Keloth", "-c", "user.email=4217831+hanishkeloth@users.noreply.github.com", "commit", "-q", "-m",
                f"{A.name} {label}: {r['call_acc']*100:.1f} percent call accuracy on FunctionChat SingleCall"], cwd=ROOT)
subprocess.run(["git", "push", "-q", "origin", "main"], cwd=ROOT, check=True)
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(A.repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=A.repo, repo_type="model", folder_path=str(ckpt), allow_patterns=["config.json", "model.safetensors", "tokenizer.model", "tokenizer_config.json"],
                  commit_message=f"Songgot-nano weights, {label}")
gg = ROOT / "vol" / f"export_gguf_{A.gguf}"; gg.mkdir(parents=True, exist_ok=True)
subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/llama.cpp/convert_hf_to_gguf.py"), str(ckpt), "--outfile", str(gg / f"{A.gguf}-f16.gguf"), "--outtype", "f16"], check=True, cwd=ROOT)
for q in ("Q8_0", "Q4_K_M"):
    subprocess.run([str(ROOT / "tools/llama.cpp/build/bin/llama-quantize"), str(gg / f"{A.gguf}-f16.gguf"), str(gg / f"{A.gguf}-{q.lower()}.gguf"), q], check=True, cwd=ROOT)
api.upload_folder(repo_id=A.repo, repo_type="model", folder_path=str(gg), allow_patterns=["*.gguf"], commit_message=f"GGUF exports, {label}")
api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=A.repo, repo_type="model", commit_message=f"Model card: {label}")
try:
    api.restart_space("Hanish/songgot")
except Exception as e:
    print("space restart:", e)
print("PUBLISHED", label, row)
