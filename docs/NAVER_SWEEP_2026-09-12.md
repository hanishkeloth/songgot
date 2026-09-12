# Korean-index sweep, 2026-09-12 (partial)

Naver itself was unreachable from every tool available today: the browser automation refuses the domain, WebFetch
refuses it, and a direct fetch returns a 217 KB page whose results are rendered by JavaScript (no headlines in the
HTML; only the dates 2026.09.11 and 2026.09.12 appear). No SerpApi key exists in our environment. Daum's news index
(Kakao) was used as the Korean-language proxy; harness/naver_sweep.py runs the full Naver query set the moment a
SERPAPI_KEY is added. Treat the "first" gate as still OPEN for Naver.

## Daum news, sorted by recency

Query: 온디바이스 한국어 소형 언어모델 공개
- 뉴스1, 2026-08-04: "비용 낮추고 특화 성능 높인 '작은 AI' 등판…SLM 경쟁 본격화" (small-model competition heating up)
- 문화일보, 2026-08-03: Kakao's four lightweight models, "Korean processing efficiency up 30 percent"
- 디지털타임스 / 전자신문, 2026-07-28: Kakao Kanana-2 SLM release (1.3B and 3B) as open source on Hugging Face
- 에너지경제, 2026-07-22: edge AI platform for vision-language-action robotics with Korean specialisation

Query: 한국어 OCR 문서 AI 모델 공개
- 머니투데이 / 아이뉴스24 / 한스경제 / 아시아경제, 2026-08-19: **뉴플로이 (Newploy) launched a free on-device document
  OCR service** built from specialised small models for layout analysis, Korean/English OCR and table structure.
  A product, not open weights; the closest thing to Songgot-V's positioning found in Korean press. Verify whether
  any weights or benchmark numbers were published before claiming anything about on-device Korean document AI.

Query: 한국어 도구 호출 함수 호출 에이전트 경량 모델
- Daum returned **no results at all**. Korean press does not cover tool calling as a category.

Query: 한국어 비전 언어모델 경량 공개 2026
- 더팩트, 2026-06-15: Naver HyperCLOVA X SEED 4B, defence-oriented omnimodal model with custom vision/audio encoders
- 여성경제신문, 2026-05-18: LG EXAONE 4.5 (33B) vision-language
- 아주경제, 2026-05-07: Kakao agent platform using VLMs
- Nothing under 1B.

## What this changes

Nothing in our claim. Newploy is the one name to add to the vision prior-art list (on-device Korean document OCR,
2026-08-19, closed). The Korean tool-calling category remains unoccupied in Korean press below 1B.
