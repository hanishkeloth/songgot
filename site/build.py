"""Render paper/SONGGOT.md into docs/ for GitHub Pages with the result tables filled from
eval/score_*.json, plus the discovery layer: canonical, Open Graph and Twitter tags, JSON-LD
(ScholarlyArticle + SoftwareSourceCode), Google Scholar citation meta, og.png, robots.txt,
sitemap.xml, llms.txt and paper.md for AI search engines, and a Q&A section with direct answers.

    .venv/bin/python site/build.py
"""
import datetime, json, pathlib, re
import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"; DOCS.mkdir(exist_ok=True)
SITE = "https://hanishkeloth.github.io/songgot/"
REPO = "https://github.com/hanishkeloth/songgot"
HF = "https://huggingface.co/imcapsule/songgot"
SPACE = "https://huggingface.co/spaces/imcapsule/songgot"
TODAY = datetime.date.today().isoformat()
TITLE = "Songgot (송곳): a Korean-first tiny agentic model for tool calling on the device"
DESC = ("Songgot is a from-scratch tiny language model (tens of millions of parameters) for Korean tool calling and "
        "structured extraction on phones and small devices, with a Korean-first 32k tokenizer, licence-clean data, "
        "and reproducible exact-match evaluation on Kakao FunctionChat-Bench. Weights, code and paper under Apache 2.0.")

md = (ROOT / "paper" / "SONGGOT.md").read_text(encoding="utf-8")
COND = ["exact", "4_random", "4_close", "8_random", "8_close"]

def row(label, params, path):
    p = ROOT / "eval" / path
    if not p.exists():
        return None
    r = json.loads(p.read_text()); bc = r["by_condition"]
    return f"| {label} | {params} | " + " | ".join(f"{bc[c]['call_acc']*100:.1f}" for c in COND) + f" | {r['call_acc']*100:.1f} | {r['name_acc']*100:.1f} |"

songgot_rows = [r for r in (row("Songgot-nano (Mac, 320M tokens)", "39M", "score_songgot_nano.json"),
                            row("Songgot (8xH100, 6B tokens)", "50M", "score_songgot.json")) if r]
if songgot_rows:
    md = re.sub(r"\| Songgot \| TBD \|[^\n]*\n", "\n".join(songgot_rows) + "\n", md, count=1)
(DOCS / "paper.md").write_text(md, encoding="utf-8")

FAQ = [
    ("What is Songgot?", "Songgot (송곳, Korean for awl) is a Korean-first tiny agentic language model for tool calling and structured extraction on the device, trained from scratch by Hanish Keloth at Palette and released under Apache 2.0."),
    ("How big is it?", "The Mac-trained Songgot-nano has 39M parameters (8 layers, hidden 512, GQA); the GPU-trained Songgot has about 50M parameters (12 layers). Both use a 32k Korean-first SentencePiece tokenizer and export to GGUF for llama.cpp."),
    ("How is it different from Needle 2?", "Needle 2 (Cactus Compute, July 2026) is English only: its 8k tokenizer spends 3.47 tokens per Hangul syllable against Songgot's 0.90, and in our FunctionChat-Bench run it answered no Korean item. Songgot is built for Korean first and evaluated on a Korean benchmark."),
    ("What data was used?", "fineweb-edu sample-10BT (ODC-By), Korean Wikipedia 20231101.ko (CC BY-SA 3.0), glaive-function-calling-v2 (Apache 2.0) and template-generated Korean tool calls released with the code. No AI-Hub data and no closed-model outputs."),
    ("How is it evaluated?", "On Kakao FunctionChat-Bench SingleCall (500 Korean items, 5 tool conditions) with a deterministic exact-match scorer on function name and arguments, so anyone can reproduce the numbers offline. Comparators run under identical prompts: Qwen3-0.6B, FunctionGemma-270M and Needle 2."),
    ("Where are the weights and code?", f"Weights and tokenizer at {HF}; code, data generators, scorer and paper at {REPO}; a CPU demo at {SPACE}."),
]

body = markdown.markdown(md, extensions=["tables", "fenced_code"])
faq_html = "".join(f"<details><summary>{q}</summary><p>{a}</p></details>" for q, a in FAQ)
jsonld = [
    {"@context": "https://schema.org", "@type": "ScholarlyArticle", "headline": TITLE, "name": TITLE, "description": DESC,
     "author": {"@type": "Person", "name": "Hanish Keloth", "url": "https://github.com/hanishkeloth", "affiliation": {"@type": "Organization", "name": "Palette"}},
     "publisher": {"@type": "Organization", "name": "Palette"}, "datePublished": TODAY, "dateModified": TODAY, "inLanguage": ["en", "ko"],
     "url": SITE, "mainEntityOfPage": SITE, "license": "https://www.apache.org/licenses/LICENSE-2.0",
     "keywords": ["Korean", "tool calling", "function calling", "tiny language model", "on-device", "small language model", "Needle 2", "FunctionChat-Bench", "agentic AI", "SentencePiece tokenizer"],
     "isBasedOn": ["https://huggingface.co/Cactus-Compute/needle2", "https://github.com/kakao/FunctionChat-Bench"],
     "citation": ["Cactus Compute, Needle 2 (2026)", "Kakao, FunctionChat-Bench (2024)"]},
    {"@context": "https://schema.org", "@type": "SoftwareSourceCode", "name": "Songgot", "codeRepository": REPO, "programmingLanguage": "Python",
     "license": "https://www.apache.org/licenses/LICENSE-2.0", "author": {"@type": "Person", "name": "Hanish Keloth"}, "runtimePlatform": "PyTorch, MLX, llama.cpp"},
    {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [{"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in FAQ]},
]
head_meta = f"""<link rel="canonical" href="{SITE}">
<meta name="robots" content="index,follow,max-image-preview:large">
<meta name="author" content="Hanish Keloth">
<meta name="keywords" content="Korean tool calling model, tiny language model, on-device LLM, Needle 2 alternative, Korean function calling, FunctionChat-Bench, Songgot, 송곳, 한국어 소형 언어모델, 온디바이스 함수 호출">
<meta property="og:type" content="article"><meta property="og:title" content="{TITLE}"><meta property="og:description" content="{DESC}">
<meta property="og:url" content="{SITE}"><meta property="og:image" content="{SITE}og.png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta property="og:locale" content="en_US"><meta property="og:locale:alternate" content="ko_KR"><meta property="og:site_name" content="Songgot">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="{TITLE}"><meta name="twitter:description" content="{DESC}"><meta name="twitter:image" content="{SITE}og.png">
<meta name="citation_title" content="{TITLE}"><meta name="citation_author" content="Keloth, Hanish"><meta name="citation_publication_date" content="{TODAY.replace('-', '/')}"><meta name="citation_online_date" content="{TODAY.replace('-', '/')}"><meta name="citation_publisher" content="Palette"><meta name="citation_language" content="en"><meta name="citation_abstract_html_url" content="{SITE}"><meta name="citation_public_url" content="{SITE}"><meta name="citation_fulltext_html_url" content="{SITE}">
<link rel="alternate" type="text/markdown" href="{SITE}paper.md" title="Paper as Markdown">
<link rel="alternate" type="text/plain" href="{SITE}llms.txt" title="llms.txt">
<script type="application/ld+json">{json.dumps(jsonld, ensure_ascii=False)}</script>"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{DESC}">
{head_meta}
<link href="https://fonts.googleapis.com/css2?family=Black+Han+Sans&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0e0f11;--panel:#16181c;--line:#262a30;--ink:#f2f0ea;--ink2:#b8b4aa;--ink3:#7f7b72;--accent:#d8ff3d}}
*{{box-sizing:border-box}}html,body{{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans KR","Apple SD Gothic Neo",system-ui,sans-serif;line-height:1.6}}
header{{padding:44px clamp(20px,6vw,90px) 26px;border-bottom:1px solid var(--line)}}
.logo{{font-family:"Black Han Sans",sans-serif;font-size:22px}}.meta{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink3);text-transform:uppercase;letter-spacing:.12em}}
h1.title{{font-family:"Black Han Sans",sans-serif;font-weight:400;font-size:clamp(36px,6vw,72px);line-height:1;margin:12px 0 8px;text-wrap:balance}}h1.title span{{color:var(--accent)}}
.lede{{max-width:70ch;color:var(--ink2);font-size:17px;margin:12px 0 0}}
.links{{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}}.links a{{border:1px solid var(--line);background:var(--panel);color:var(--ink);text-decoration:none;padding:9px 14px;border-radius:8px;font-size:14px}}.links a.primary{{background:var(--accent);color:#0e0f11;border-color:var(--accent);font-weight:600}}
.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;padding:22px clamp(20px,6vw,90px);border-bottom:1px solid var(--line)}}
.fact{{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--panel)}}.fact b{{display:block;font-family:"IBM Plex Mono",monospace;font-size:20px;font-weight:500;color:var(--accent)}}.fact small{{color:var(--ink3);font-size:12px}}
main{{max-width:78ch;padding:30px clamp(20px,6vw,90px) 40px}}main h1{{display:none}}main h2{{font-size:24px;margin:40px 0 10px;letter-spacing:-.01em}}main h3{{font-size:18px;margin:28px 0 8px}}
main p,main li{{color:var(--ink2);font-size:16px}}main strong{{color:var(--ink)}}main code{{font-family:"IBM Plex Mono",monospace;font-size:13px;background:#0b0c0e;border:1px solid var(--line);padding:1px 5px;border-radius:5px}}
main pre{{background:#0b0c0e;border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto}}main pre code{{border:0;padding:0}}
.tbl{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}th{{color:var(--ink3);font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.08em}}td{{font-variant-numeric:tabular-nums}}
section.faq{{max-width:78ch;padding:10px clamp(20px,6vw,90px) 60px}}section.faq h2{{font-size:24px}}details{{border-top:1px solid var(--line);padding:12px 0}}summary{{cursor:pointer;font-weight:600;color:var(--ink)}}details p{{color:var(--ink2);margin:8px 0 0}}
footer{{padding:24px clamp(20px,6vw,90px) 60px;border-top:1px solid var(--line);color:var(--ink3);font-size:13px}}footer a{{color:var(--ink2)}}
</style></head><body>
<header><div class="logo">songgot · 송곳</div><div class="meta">paper · palette · {TODAY}</div>
<h1 class="title">A Korean-first tiny agentic model,<br><span>for tool calling on the device.</span></h1>
<p class="lede">{DESC}</p>
<div class="links"><a class="primary" href="{HF}">Weights on Hugging Face</a><a href="{REPO}">Code on GitHub</a><a href="{SPACE}">Live demo</a><a href="paper.md">Paper as Markdown</a></div></header>
<section class="facts" aria-label="Key facts">
<div class="fact"><b>0.90</b><small>tokens per Hangul syllable (Songgot 32k tokenizer); Needle 2: 3.47</small></div>
<div class="fact"><b>500</b><small>Korean items, FunctionChat-Bench SingleCall, exact-match scorer</small></div>
<div class="fact"><b>Apache 2.0</b><small>weights, tokenizer, code, data generators, paper</small></div>
<div class="fact"><b>0 closed-model labels</b><small>licence-clean data, provenance in the paper</small></div>
</section>
<main>{body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")}</main>
<section class="faq"><h2>Questions and answers</h2>{faq_html}</section>
<footer>Songgot is released under Apache 2.0 by <a href="https://github.com/hanishkeloth">Hanish Keloth</a>, CTO at Palette. Korean Wikipedia text is CC BY-SA 3.0; FunctionChat-Bench is Apache 2.0 (Kakao). Numbers on this page come from the run logs in the repository; nothing is estimated. <a href="llms.txt">llms.txt</a> · <a href="paper.md">paper.md</a> · <a href="sitemap.xml">sitemap</a></footer>
</body></html>"""
(DOCS / "index.html").write_text(html, encoding="utf-8")

(DOCS / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {SITE}sitemap.xml\n", encoding="utf-8")
(DOCS / "sitemap.xml").write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n<url><loc>{SITE}</loc><lastmod>{TODAY}</lastmod><changefreq>weekly</changefreq><priority>1.0</priority></url>\n<url><loc>{SITE}paper.md</loc><lastmod>{TODAY}</lastmod></url>\n<url><loc>{SITE}llms.txt</loc><lastmod>{TODAY}</lastmod></url>\n</urlset>\n', encoding="utf-8")
facts = "\n".join(f"- {q} {a}" for q, a in FAQ)
(DOCS / "llms.txt").write_text(f"""# Songgot (송곳)

> {DESC}

Author: Hanish Keloth (Palette). Published {TODAY}. Licence: Apache 2.0.

## Facts
{facts}
- Tokens per Hangul syllable on the 100 FunctionChat SingleCall queries: Songgot 32k 0.90, Gemma 3 262k 0.98, Qwen3 151k 1.15, Needle 2 8k 3.47.
- Comparator call accuracy (exact match, 500 items): Qwen3-0.6B 43.2 percent, FunctionGemma-270M 2.2 percent, Needle 2 0.0 percent.

## Links
- Paper (HTML): {SITE}
- Paper (Markdown): {SITE}paper.md
- Code: {REPO}
- Weights: {HF}
- Demo: {SPACE}
""", encoding="utf-8")
(DOCS / ".nojekyll").write_text("", encoding="utf-8")

# og.png: title card, text only, own fonts
try:
    from PIL import Image, ImageDraw, ImageFont
    fonts = pathlib.Path.home() / "Desktop" / "PALETTE-K-VIDEO" / "corpus" / "fonts"
    im = Image.new("RGB", (1200, 630), (14, 15, 17)); d = ImageDraw.Draw(im)
    fb = ImageFont.truetype(str(fonts / "BlackHanSans-Regular.ttf"), 76); fs = ImageFont.truetype(str(fonts / "NanumGothic-Bold.ttf"), 28); fm = ImageFont.truetype(str(fonts / "NanumGothic-Bold.ttf"), 22)
    d.text((70, 70), "songgot · 송곳", font=fs, fill=(216, 255, 61))
    d.text((70, 140), "A Korean-first tiny", font=fb, fill=(242, 240, 234)); d.text((70, 225), "agentic model,", font=fb, fill=(242, 240, 234)); d.text((70, 310), "for tool calling on the device.", font=fb, fill=(216, 255, 61))
    d.text((70, 440), "0.90 tokens per Hangul syllable  ·  FunctionChat-Bench, exact match  ·  Apache 2.0", font=fm, fill=(184, 180, 170))
    d.text((70, 540), "Hanish Keloth, Palette  ·  hanishkeloth.github.io/songgot", font=fm, fill=(127, 123, 114))
    im.save(DOCS / "og.png")
except Exception as e:
    print("og.png skipped:", e)
print("built:", sorted(p.name for p in DOCS.iterdir()))
