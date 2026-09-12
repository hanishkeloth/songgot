# Paid Korean data: what to buy first, and why

Palette, 2026-09-12. Prices are NOT listed because none of these vendors publish them; each needs a quote request.
Ordered by expected effect on the two models. The free tier (docs/KOREAN_DATA_SOURCES_2026-09-11.md, section A)
is already downloaded; this list is what money adds.

## For the language model (the pretraining gap)

1. **한국언론진흥재단 BigKinds news archive (bigkinds.or.kr)** - decades of Korean newspaper text, clean, edited,
   encyclopedic-adjacent register that matches what our ablation showed helps. Licence for AI training must be
   negotiated with the foundation (research access is free; commercial model training is a separate agreement).
   Effect: the single largest clean Korean prose source that exists. Ask for: bulk text export, AI-training rights,
   commercial-derivative rights for weights.
2. **플리토 (Flitto) language data** - Korean text and parallel corpora sold explicitly for AI training, with
   contracts that name the use. Effect: instruction-style Korean, conversational register our corpus lacks.
3. **국립국어원 모두의 말뭉치 (corpus.korean.go.kr)** - free, but requires an application stating the purpose; the
   licence permits research and, per set, commercial use after approval. Effect: 20+ billion characters of curated
   Korean (newspapers, spoken, web). Worth the paperwork before any purchase.

## For the vision model (Korean documents, OCR)

4. **Nexdata Korean OCR sets** (Nexdata-kr on the Hub shows samples: 5,711 handwriting images, 104,320 natural-scene
   Korean+Hindi OCR, 500,000-image 21-language OCR) - vendor sells the full sets; licence permits training.
   Effect: real Korean handwriting and scene text, which no free set covers.
5. **셀렉트스타 / 크라우드웍스 / 에이모 / 테스트웍스** - Korean labelling companies; buy existing Korean document and
   receipt sets or commission labelling of our own 18,371 pages (layout boxes, key-value, table structure). Ask each
   for their catalogue and a per-image quote for 100k document images.
6. **Getty Images Korea / 포바이포** - rights-cleared Korean imagery; only if Songgot-V needs general Korean scene
   images (signage, storefronts). Lower priority than documents.

## What I would do today

- Send BigKinds and 모두의 말뭉치 the requests now (weeks of lead time; no cost to ask).
- Get Nexdata's quote for the Korean handwriting + scene-text sets (fast, fixed price, fills a real hole).
- Hold the labelling commission until Songgot-V's stage-2 numbers on K-DTCBench show where it fails.

## Do not buy

- Anything derived from AI-Hub (cannot leave Korea, cannot be resold).
- 나무위키 dumps (CC BY-NC-SA).
- Any set whose labels were produced by a closed model (OpenAI, Google, Anthropic terms bar training a competitor).
