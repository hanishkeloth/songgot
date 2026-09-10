"""Render paper/SONGGOT.md into docs/index.html (GitHub Pages) with the result tables filled
from eval/score_*.json when present. Run after every eval:  .venv/bin/python site/build.py"""
import json, pathlib, re, datetime
import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
md = (ROOT / "paper" / "SONGGOT.md").read_text(encoding="utf-8")

# fill Songgot rows from score files if they exist
COND = ["exact", "4_random", "4_close", "8_random", "8_close"]
def row(label, params, path):
    p = ROOT / "eval" / path
    if not p.exists():
        return None
    r = json.loads(p.read_text()); bc = r["by_condition"]
    return f"| {label} | {params} | " + " | ".join(f"{bc[c]['call_acc']*100:.1f}" for c in COND) + f" | {r['call_acc']*100:.1f} | {r['name_acc']*100:.1f} |"
for label, params, path in (("Songgot-nano (Mac, 320M tokens)", "39M", "score_songgot_nano.json"), ("Songgot (8xH100, 6B tokens)", "50M", "score_songgot.json")):
    r = row(label, params, path)
    if r:
        md = re.sub(r"\| Songgot \| TBD \|[^\n]*\n", r + "\n", md, count=1) if "| Songgot | TBD |" in md else md.replace("| Needle 2 | 45M |", r + "\n| Needle 2 | 45M |", 1)

body = markdown.markdown(md, extensions=["tables", "fenced_code"])
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Songgot (송곳): a Korean-first tiny agentic model</title>
<meta name="description" content="Songgot: a Korean-first tiny agentic model for tool calling on the device. Paper, weights, code, benchmark.">
<link href="https://fonts.googleapis.com/css2?family=Black+Han+Sans&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0e0f11;--panel:#16181c;--line:#262a30;--ink:#f2f0ea;--ink2:#b8b4aa;--ink3:#7f7b72;--accent:#d8ff3d}}
*{{box-sizing:border-box}}html,body{{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans KR","Apple SD Gothic Neo",system-ui,sans-serif;line-height:1.6}}
header{{padding:44px clamp(20px,6vw,90px) 26px;border-bottom:1px solid var(--line)}}
.logo{{font-family:"Black Han Sans",sans-serif;font-size:22px}}.meta{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink3);text-transform:uppercase;letter-spacing:.12em}}
h1.title{{font-family:"Black Han Sans",sans-serif;font-weight:400;font-size:clamp(36px,6vw,72px);line-height:1;margin:12px 0 8px;text-wrap:balance}}h1.title span{{color:var(--accent)}}
.links{{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}}.links a{{border:1px solid var(--line);background:var(--panel);color:var(--ink);text-decoration:none;padding:9px 14px;border-radius:8px;font-size:14px}}.links a.primary{{background:var(--accent);color:#0e0f11;border-color:var(--accent);font-weight:600}}
main{{max-width:78ch;padding:30px clamp(20px,6vw,90px) 90px}}main h1{{display:none}}main h2{{font-size:24px;margin:40px 0 10px;letter-spacing:-.01em}}main h3{{font-size:18px;margin:28px 0 8px}}
main p,main li{{color:var(--ink2);font-size:16px}}main strong{{color:var(--ink)}}main code{{font-family:"IBM Plex Mono",monospace;font-size:13px;background:#0b0c0e;border:1px solid var(--line);padding:1px 5px;border-radius:5px}}
main pre{{background:#0b0c0e;border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto}}main pre code{{border:0;padding:0}}
.tbl{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}th{{color:var(--ink3);font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.08em}}td{{font-variant-numeric:tabular-nums}}
footer{{padding:24px clamp(20px,6vw,90px) 60px;border-top:1px solid var(--line);color:var(--ink3);font-size:13px}}
</style></head><body>
<header><div class="logo">songgot · 송곳</div><div class="meta">paper · palette · {datetime.date.today().isoformat()}</div>
<h1 class="title">A Korean-first tiny agentic model,<br><span>for tool calling on the device.</span></h1>
<div class="links"><a class="primary" href="https://huggingface.co/imcapsule/songgot">Weights on Hugging Face</a><a href="https://github.com/hanishkeloth/songgot">Code on GitHub</a><a href="https://huggingface.co/spaces/imcapsule/songgot">Live demo</a></div></header>
<main>{body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")}</main>
<footer>Songgot is released under Apache 2.0 by Hanish Keloth, Palette. Korean Wikipedia text is CC BY-SA 3.0; FunctionChat-Bench is Apache 2.0 (Kakao). Numbers on this page come from the run logs in the repository; nothing is estimated.</footer>
</body></html>"""
(ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
print("docs/index.html", len(html), "bytes")
