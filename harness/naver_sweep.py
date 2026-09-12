"""Naver prior-art sweep through SerpApi's Naver engine. Naver blocks direct fetches and renders results in JS, so
this is the reliable route. Needs SERPAPI_KEY in the environment (or in ~/Workbench/2026/test/video/.env as
SERPAPI_KEY=...). Writes docs/NAVER_SWEEP_<date>.md with every result: query, title, source, date, link, snippet.

    SERPAPI_KEY=... .venv/bin/python harness/naver_sweep.py
"""
import datetime
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

QUERIES = [
    "온디바이스 한국어 소형 언어모델 공개", "한국어 경량 LLM 오픈소스 공개 2026", "한국어 함수 호출 모델", "한국어 도구 호출 에이전트 모델",
    "한국어 툴 콜링 벤치마크", "FunctionChat-Bench", "카나나 경량 모델 도구 호출", "엑사원 1.2B 온디바이스",
    "한국어 처음부터 학습 소형 모델", "한국어 비전 언어 모델 경량 공개", "한국어 문서 이해 AI 모델 공개", "한국어 OCR 오픈소스 모델 2026",
    "온디바이스 OCR 한국어", "한국어 VLM 1B 미만", "브라우저에서 실행 한국어 AI 모델", "한국어 AI 학습 데이터셋 공개 2026",
]


def key():
    k = os.environ.get("SERPAPI_KEY")
    if not k:
        env = pathlib.Path("~/Workbench/2026/test/video/.env").expanduser()
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("SERPAPI_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        sys.exit("SERPAPI_KEY not set: add SERPAPI_KEY=... to ~/Workbench/2026/test/video/.env")
    return k


def search(q, k, where="news"):
    params = {"engine": "naver", "query": q, "where": where, "api_key": k}
    with urllib.request.urlopen("https://serpapi.com/search.json?" + urllib.parse.urlencode(params), timeout=60) as r:
        return json.load(r)


def main():
    k = key(); today = datetime.date.today().isoformat(); out = []
    for q in QUERIES:
        for where in ("news", "web"):
            try:
                d = search(q, k, where)
            except Exception as e:
                out.append(f"### {q} ({where})\n\n- ERROR {type(e).__name__}: {str(e)[:120]}\n"); continue
            rows = d.get("news_results") or d.get("web_results") or d.get("organic_results") or []
            lines = [f"- **{r.get('title','').strip()}** | {r.get('source') or r.get('displayed_link') or ''} | {r.get('date') or ''} | {r.get('link','')}\n  {str(r.get('snippet') or '').strip()[:200]}" for r in rows[:10]]
            out.append(f"### {q} ({where}, {len(rows)} results)\n\n" + ("\n".join(lines) if lines else "- none") + "\n")
            time.sleep(1)
    p = pathlib.Path("docs") / f"NAVER_SWEEP_{today}.md"
    p.write_text(f"# Naver sweep {today} (SerpApi engine=naver)\n\n" + "\n".join(out), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
