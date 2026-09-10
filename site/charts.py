"""Figures for the paper, drawn as SVG from measured data only (no illustrative numbers).
fig1 tokens per Hangul syllable, fig2 FunctionChat call accuracy by tool condition, fig3 the
pretraining loss curve read from logs/pretrain_mlx.log, fig4 pretraining corpus in tokens.
Page is single-theme dark, so colours are fixed; the same SVG is rasterised to PNG for paper.md."""
import json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BG, INK, INK2, INK3, LINE, ACC, ACC2 = "#0e0f11", "#f2f0ea", "#b8b4aa", "#7f7b72", "#262a30", "#d8ff3d", "#8fd400"
GREYS = ["#a6a298", "#6b675f", "#45423c"]
FONT = 'font-family="IBM Plex Mono, Menlo, monospace"'

def _open(title, width, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{title}" {FONT} font-size="12">',
            f'<rect width="{width}" height="{height}" fill="{BG}"/>',
            f'<text x="16" y="22" fill="{INK}" font-size="13" font-weight="500">{title}</text>']

def bar_chart(title, rows, unit="", width=720, highlight=0, fmt="{:.2f}", note=""):
    n = len(rows); h = 36; top = 44; height = top + n * h + (34 if note else 16)
    vmax = max(v for _, v in rows) * 1.15; lw = 275
    out = _open(title, width, height)
    for i, (lab, v) in enumerate(rows):
        y = top + i * h; bw = (width - lw - 90) * v / vmax
        col = ACC if i == highlight else GREYS[min(i - 1 if i > highlight else i, 2)]
        out.append(f'<text x="{lw-10}" y="{y+17}" fill="{INK2}" text-anchor="end">{lab}</text>')
        out.append(f'<rect x="{lw}" y="{y+4}" width="{bw:.1f}" height="20" rx="3" fill="{col}"/>')
        out.append(f'<text x="{lw+bw+8:.1f}" y="{y+18}" fill="{INK}">{fmt.format(v)}{unit}</text>')
    if note:
        out.append(f'<text x="16" y="{height-12}" fill="{INK3}" font-size="11">{note}</text>')
    out.append("</svg>"); return "".join(out)

def grouped_bars(title, conds, series, width=720, note=""):
    left, bottom, top = 56, 76, 40; height = 318; plot_h = height - top - bottom; plot_w = width - left - 24
    gw = plot_w / len(conds); bw = min(24, (gw - 12) / max(1, len(series)))
    out = _open(title, width, height)
    for t in (0, 25, 50, 75, 100):
        y = top + plot_h - plot_h * t / 100
        out.append(f'<line x1="{left}" x2="{width-24}" y1="{y:.1f}" y2="{y:.1f}" stroke="{LINE}"/><text x="{left-8}" y="{y+4:.1f}" fill="{INK3}" text-anchor="end" font-size="11">{t}</text>')
    for ci, c in enumerate(conds):
        x0 = left + ci * gw + (gw - bw * len(series)) / 2
        for si, (name, vals, col) in enumerate(series):
            v = vals[ci]; bh = plot_h * v / 100; x = x0 + si * bw
            out.append(f'<rect x="{x:.1f}" y="{top+plot_h-bh:.1f}" width="{bw-2:.1f}" height="{max(bh,1.5):.1f}" rx="2" fill="{col}"/>')
            out.append(f'<text x="{x+(bw-2)/2:.1f}" y="{top+plot_h-bh-4:.1f}" fill="{INK2}" text-anchor="middle" font-size="10">{v:.0f}</text>')
        out.append(f'<text x="{left+ci*gw+gw/2:.1f}" y="{height-54}" fill="{INK2}" text-anchor="middle">{c}</text>')
    lx = left
    for name, _, col in series:
        out.append(f'<rect x="{lx}" y="{height-40}" width="10" height="10" rx="2" fill="{col}"/><text x="{lx+15}" y="{height-31}" fill="{INK2}" font-size="11">{name}</text>'); lx += 15 + 7 * len(name) + 18
    if note:
        out.append(f'<text x="{left}" y="{height-12}" fill="{INK3}" font-size="11">{note}</text>')
    out.append("</svg>"); return "".join(out)

def loss_curve(title, pts, width=720, note=""):
    if len(pts) < 2:
        return ""
    left, bottom, top = 56, 50, 40; height = 300; plot_h = height - top - bottom; plot_w = width - left - 24
    xmax = max(s for s, _ in pts); ymin = min(l for _, l in pts) - 0.2; ymax = max(l for _, l in pts) + 0.2
    X = lambda s: left + plot_w * s / xmax; Y = lambda l: top + plot_h - plot_h * (l - ymin) / (ymax - ymin)
    out = _open(title, width, height)
    for k in range(5):
        l = ymin + (ymax - ymin) * k / 4; y = Y(l)
        out.append(f'<line x1="{left}" x2="{width-24}" y1="{y:.1f}" y2="{y:.1f}" stroke="{LINE}"/><text x="{left-8}" y="{y+4:.1f}" fill="{INK3}" text-anchor="end" font-size="11">{l:.1f}</text>')
    for k in range(5):
        s = xmax * k / 4
        out.append(f'<text x="{X(s):.1f}" y="{height-30}" fill="{INK3}" text-anchor="middle" font-size="11">{int(s):,}</text>')
    out.append(f'<path d="M{" L".join(f"{X(s):.1f},{Y(l):.1f}" for s, l in pts)}" fill="none" stroke="{ACC}" stroke-width="2" stroke-linejoin="round"/>')
    s, l = pts[-1]
    out.append(f'<circle cx="{X(s):.1f}" cy="{Y(l):.1f}" r="4" fill="{ACC}"/><text x="{X(s)-8:.1f}" y="{Y(l)-10:.1f}" fill="{INK}" text-anchor="end">loss {l:.2f} at step {s:,}</text>')
    out.append(f'<text x="{left}" y="{height-12}" fill="{INK3}" font-size="11">optimizer step (32 sequences x 1,024 tokens each)</text>')
    if note:
        out.append(f'<text x="{width-24}" y="{height-12}" fill="{INK3}" text-anchor="end" font-size="11">{note}</text>')
    out.append("</svg>"); return "".join(out)

def loss_points():
    log = ROOT / "logs" / "pretrain_mlx.log"
    if not log.exists():
        return []
    pts = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"\[pre\] step (\d+)/\d+ loss ([0-9.]+)", log.read_text())]
    return [p for p in pts if p[0] > 0]

def figures():
    sys.path.insert(0, str(ROOT / "eval")); import functionchat_exact as F
    figs = {}
    figs["fig1_tokens"] = bar_chart("Figure 1. Tokens per Hangul syllable (lower is better)",
                                    [("Songgot 32k (ours)", 0.90), ("Gemma 3 / FunctionGemma 262k", 0.98), ("Qwen3 151k", 1.15), ("Needle 2 8k", 3.47)],
                                    note="100 FunctionChat SingleCall queries, 2,071 syllables; eval/tokenizer_study.py.")
    conds = ["exact", "4_random", "4_close", "8_random", "8_close"]; series = []
    for name, path, col in [("Songgot-nano (ours)", "score_songgot_nano.json", ACC), ("Songgot (ours)", "score_songgot.json", ACC2),
                            ("Qwen3-0.6B", "preds_qwen3_0.6b.jsonl", GREYS[0]), ("FunctionGemma-270M", "preds_functiongemma_270m.jsonl", GREYS[1]), ("Needle 2", "preds_needle2.jsonl", GREYS[2])]:
        p = ROOT / "eval" / path
        if p.exists():
            r = json.loads(p.read_text()) if path.endswith(".json") else F.score(str(p))
            series.append((name, [r["by_condition"][c]["call_acc"] * 100 for c in conds], col))
    figs["fig2_bench"] = grouped_bars("Figure 2. FunctionChat-Bench SingleCall call accuracy by tool condition (percent)", conds, series,
                                      note="Exact match, 100 items per condition. Needle 2 is English only and answered none.")
    pts = loss_points()
    tail = f"run in progress, {pts[-1][0]:,} of 9,765 steps" if pts and pts[-1][0] < 9765 else "complete"
    figs["fig3_loss"] = loss_curve("Figure 3. Songgot-nano pretraining loss, Apple M5 Max, MLX bf16, from the run log", pts, note=tail)
    figs["fig4_data"] = bar_chart("Figure 4. Pretraining corpus under the Songgot tokenizer (billions of tokens)",
                                  [("Korean Wikipedia 20231101.ko", 0.60), ("fineweb-edu sample-10BT (slice)", 1.63)], unit="B",
                                  note="Korean is upsampled to about half of each batch (p_ko 0.5); no AI-Hub data, no closed-model outputs.")
    return {k: v for k, v in figs.items() if v}

def write_png(figs, out_dir: pathlib.Path):
    for k, svg in figs.items():
        (out_dir / f"{k}.svg").write_text(svg, encoding="utf-8")
        subprocess.run(["rsvg-convert", "-z", "2", "-o", str(out_dir / f"{k}.png"), str(out_dir / f"{k}.svg")], check=True)

if __name__ == "__main__":
    f = figures(); write_png(f, ROOT / "docs"); print({k: len(v) for k, v in f.items()})
