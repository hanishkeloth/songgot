# Korean training-data sourcing report

**Date: 2026-09-11. Prepared for Palette / palette-lab (Korean entity; palette-lab is an Indian R&D subsidiary).**

Two models in scope:

- **Model 1 - Songgot-class tiny Korean LM.** 50M-200M parameters, trained from scratch, for tool
  calling and on-device assistant use. Currently pretrained on Korean Wikipedia (0.6B tokens) plus
  English fineweb-edu. Measured in our own run: a 24.4B-token corpus containing 9.0B tokens of
  FineWeb-2 Korean web text scored **8.6 percent** on FunctionChat SingleCall against **11.4 percent**
  for the 6B-token encyclopedic corpus (v5 post-training), and **17.4 against 23.8** on v6. Tool-name
  accuracy rose while whole-call accuracy fell. **General Korean web crawl is not the gap.** What we
  need is clean, educational, instructional, encyclopedic, technical Korean prose.
- **Model 2 - tiny Korean document VLM.** Under 1B parameters, Korean document understanding and OCR:
  scanned documents, forms, receipts, tables, charts, slides, signage, product detail pages.
  Benchmarks: NCSOFT K-DTCBench, K-MMBench, K-SEED, K-MMStar.

Already held: **300 GB of our own Korean documents and images, user-supplied, rights clean.**

---

## Legal frame (verified 2026-09-11, read this before the tables)

Every row in this report was scored against these rules.

### 1. No TDM exception in Korean copyright law

Korea still has **no text-and-data-mining exception**. The UK (CDPA s.29A, 2014), Germany
(UrhG s.60d, 2017) and Japan (Art. 30-4, in force 2019-01-01) all created one; the Korean
National Assembly has not. Bills to add one have been introduced repeatedly and have stalled
against organised creator opposition, and the live legislative proposals are for a
**non-commercial** TDM limitation, which would not cover us. US-style fair use is not
available; Korean 공정이용 (저작권법 제35조의5) is untested for AI pretraining and is not
a reliable basis for a commercial model.

**Operating rule: every corpus needs an actual licence or an explicit AI-training permission.
No exceptions, no "it is on the open web" reasoning.**

One nuance discovered during this research and worth knowing precisely. On **2026-02-26**
문화체육관광부 and 한국저작권위원회 published the **「생성형 AI 저작물 학습 관련 공정이용 안내서」**,
which applies 저작권법 제35조의5 (공정이용) to AI training with a four-factor test. It does *not*
automatically exclude commercial AI training. But the factors it lists as weighing **against** fair
use describe our situation almost exactly: using whole works (which pretraining always does),
highly creative works, paywalled or login-gated content, and robots.txt-blocked content. And one
factor is decisive against us over time:

> the existence of licensing systems and collective management structures for the same purpose

As BECUAI stands up a licensed Korean news market and 대한출판문화협회 builds a book-licensing
scheme, the fair-use argument for scraping those same categories collapses. **Buying gets more
necessary over time, not less.** The guide is reference material, not binding law - courts decide -
so it changes nothing about our operating rule. It only explains why the rule will tighten.

For completeness: multiple TDM 면책 bills (도종환 2021 and 2023, 이용호 2022, 황보승희, 이인영) are
**still pending in the National Assembly and none has passed as of September 2026**. Do not plan
around one passing.

### 2. 공공누리 AI유형 and 제0유형 - the single biggest change in our favour

Confirmed live on kogl.or.kr today. Announced 2026-01-28 by 문화체육관광부 with 과학기술정보통신부
at the 4th Science and Technology Ministers' Conference, as the
**공공저작물 인공지능(AI) 학습 활용 확대 방안**.

- **제0유형: 자유이용** - no attribution required, commercial and non-commercial use permitted,
  derivative works permitted. This is effectively CC0 for Korean public works.
- **AI유형: 인공지능 학습용** - selected *alongside* an existing type 1-4 and displayed with it.
  Where the AI유형 mark is present, the work may be used for AI training **even if the underlying
  type forbids commercial use or modification**.

AI유형 conditions, verbatim from kogl.or.kr:

> 이용자는 공공저작물과 동일하거나 실질적으로 유사한 인공지능 산출물이 생성되지 않도록
> 기술적 조치 등을 하여야 합니다.

and:

> 공공저작물을 이용해 제작한 인공지능 학습용 데이터의 재판매는 금지됩니다.

The three operative consequences for us:

1. **The trained model may be used commercially.** kogl.or.kr states this directly. This is the
   permission we have been missing.
2. **Anti-memorisation is a contractual obligation, not a nicety.** We must be able to show
   technical measures - dedup, memorisation probes, canary extraction tests - and log them.
   Budget engineering time for this; it is a condition of the licence.
3. **We may not resell AI training data built from public works.** Fine for us; we do not sell data.
   But it blocks any plan to publish a derived Korean corpus as a product. Publishing the *model*
   is fine; publishing the *corpus* is not.

Adoption so far: **37 종 공공저작물 across 33 ministries and agencies** have been converted to
공공누리 AI유형, including government policy materials and publications, committee 심결문 and ruling
documents, case collections, and some civil-service examination questions. Contributing bodies named
include 국사편찬위원회, 국방부, 경찰청 and 조세심판원. Separately the **AI/고가치 공공데이터 TOP100**
programme was accelerated by a year: 10 datasets in 2025, 25 more in 2026, the remaining 65 in 2027
(was 2028). 한국저작권위원회 (copyright.or.kr) and kogl.or.kr administer the marks.

**The badge is per item, never per site.** Two institutions on the same portal, and two datasets from
the same institution, routinely carry different types. Every acquisition must record the per-item
badge at download time, with a screenshot or the API's licence field, into a provenance manifest.

### 3. 공공누리 제4유형 is unusable for us

**제4유형: 출처표시 + 상업적 이용금지 + 변경금지.** Non-commercial only *and* no derivative works.
A trained model is a derivative use and we license commercially, so 제4유형 fails on both limbs.
제2유형 (상업적 이용금지) also fails. 제3유형 (변경금지) is a real risk and should be treated as
unusable for training absent an AI유형 mark beside it. **Only 제0유형, 제1유형, and anything carrying
the AI유형 mark are acceptable.**

### 4. Korean statutes and judgments are outside copyright entirely

저작권법 제7조 (보호받지 못하는 저작물):

> 1. 헌법ㆍ법률ㆍ조약ㆍ명령ㆍ조례 및 규칙
> 2. 국가 또는 지방자치단체의 고시ㆍ공고ㆍ훈령 그 밖에 이와 유사한 것
> 3. 법원의 판결ㆍ결정ㆍ명령 및 심판이나 행정심판절차 그 밖에 이와 유사한 절차
> 4. 국가 또는 지방자치단체가 작성한 것으로서 제1호 내지 제3호에 규정된 것의 편집물 또는 번역물

Confirmed: **statutes, treaties, decrees, ordinances, rules, government 고시/공고/훈령, court
judgments and administrative-tribunal decisions carry no copyright at all in Korea**, and neither do
state-made compilations or translations of them. This is stronger than any licence, because it is an
absence of rights rather than a grant. It makes 법제처 law.go.kr, 대법원 판례, and administrative
tribunal decisions our cleanest large source of formal technical Korean prose.

**The one trap: 데이터베이스제작자의 권리.** 저작권법 제93조 gives a database producer a separate
right over "전부 또는 상당한 부분" of the database, and expressly deems repeated or systematic
copying of individual items to be copying of a substantial part where it conflicts with normal
exploitation. 대법원 2022. 5. 12. 선고 2021도1533 applied this to web crawling. So: the *content* of
a statute is free, but mass-scraping a *database* of statutes can still infringe the producer's
right. **Use the official Open API or official bulk distribution, never a crawler.**

### 5. AI-Hub is in-zone only - do not plan on downloading it

AI-Hub (aihub.or.kr) data **may not leave Korea and may not be processed on US infrastructure**. The
안심존 (safe zone) requires Korean nationals working physically in-zone. Only **trained weights** may
be exported, and only after NIA review. Our Modal GPUs are not in Korea, so AI-Hub is out for this
programme, exactly as PLAN.md already recorded ("No AI-Hub bytes").

This extends to laundered copies: **a Hugging Face or GitHub re-upload of AI-Hub data is still
AI-Hub data** and is legally contaminated regardless of what licence the uploader typed. Confirmed
examples are listed in the DO NOT USE section.

The operative clauses, verbatim from aihub.or.kr/intrcn/guid/usagepolicy.do:

> 본 AI데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다

> AI 허브에서 제공하는 AI 데이터셋의 판매 등 상업적 이용을 희망하는 경우 수행기관과 별도 협의가
> 필요합니다

> 국외에 소재하는 법인, 단체 또는 개인이 AI데이터 등을 이용하기 위해서는 수행기관 등 및
> 한국지능정보사회진흥원과 별도로 합의가 필요합니다

> 본 AI데이터 등의 국외 반출을 위해서는 수행기관 등 및 한국지능정보사회진흥원과 별도로 합의가
> 필요합니다

and on many individual datasets: 비상업적인 목적의 연구나 개발에만 사용될 수 있다.

**Application and timeline for the 안심존 route.** Register on aihub.or.kr, submit a 이용 신청 naming
the dataset and the purpose, then a separate 안심존 이용 신청 for the physical facility. The zone
requires Korean nationals working in-zone on NIA-controlled machines; nothing leaves except trained
weights, and only after NIA review of the export request. Realistic elapsed time is measured in
weeks to months per dataset, and it requires staff physically in Korea. **We are not proposing this
route and no AI-Hub bytes should be downloaded.** It is recorded here so the option is understood,
not taken.

**The important corollary, and it is a commercial opportunity rather than a restriction.** Because
commercial use must be negotiated **with the 수행기관** - the private contractor that actually built
the dataset - and not merely with NIA, the contractors are themselves licensable counterparties who
hold the underlying data and the pipelines that made it. Licensing Korean document-image data
*directly from the builder under our own commercial contract* is a completely different legal act
from downloading AI-Hub, and it is the single best-value route in this report. See section C.

### 6. Personal data in document images - PIPA, and a change dated three days ago

Korean document images are full of personal data: names, addresses, resident registration numbers,
card numbers, business registration numbers, signatures. 개인정보보호법 applies independently of
copyright, and 가명처리 (pseudonymisation) of unstructured image data is governed by
**비정형 데이터 가명처리 기준 (2024.2)**, the **AI 개발·서비스를 위한 공개된 개인정보 처리 안내서
(2024.7)**, and the **생성형 인공지능(AI) 개발·활용을 위한 개인정보 처리 안내서 (2025.8.6)** from
개인정보보호위원회.

New and directly relevant: the **개인정보 보호법 AI 특례** amendment passed the National Assembly on
**2026-08-20**, was promulgated as **법률 제21910호 on 2026-09-08**, and takes effect **2027-03-09**.
It adds 제28조의12 to 제28조의15, allowing lawfully collected personal data to be repurposed for AI
development beyond the original collection purpose, **subject to prior PIPC approval**, on three
conditions: (a) de-identification would make the AI development impracticable, (b) technical,
administrative and physical safeguards are in place including cloud-environment protections, and
(c) the development serves a public interest with a markedly low likelihood of harm.

Practical reading for us: **the AI 특례 is an approval route, not a free pass, and it is six months
away.** For anything shipping before 2027-03-09, the safe posture is unchanged - redact or synthesise
personal data in document images, and prefer synthetic and blank-form renders over real filled-in
documents. Our own 300 GB should be run through a redaction pass and the pass should be documented.

### 7. Likeness and NC

- **No real-person likeness (초상권), no celebrity images.** Korean right of publicity is now
  statutory via 부정경쟁방지법 제2조 제1호 (타)목 as well as civil 초상권. Any document-image set
  containing identifiable faces or real ID documents is excluded.
- **CC BY-NC is for benchmarking, never for training.** Every NC set in this report is marked
  BENCHMARK ONLY. This is not a technicality: all four of our target NCSOFT benchmarks are
  CC BY-NC 4.0, which is fine because we only score on them.

### 8. AI 기본법

인공지능 발전과 신뢰 기반 조성 등에 관한 기본법 came into full force **2026-01-22**, making Korea the
first country with comprehensive AI regulation fully in effect. It is a promotion-and-trust framework
rather than a data-rights statute, and its data provisions run in our favour: it obliges government to
supply public data as training data, which is the statutory engine behind the 공공누리 AI유형 rollout
above. Compliance load for a sub-1B open-weight model is transparency and labelling of generative
output, not a licensing barrier.

---

# A) FREE AND DOWNLOADABLE NOW - Korean text corpora for commercial model training

Ranked roughly by value to Model 1 (the tiny tool-calling LM). Verbatim licence names are used
throughout. "Commercial training" means: may we train a model on this and then license the model
commercially.

## A.1 Tier 1 - clean, large, unambiguously usable

| # | Name | URL | Licence (verbatim) | Size | Format | Commercial training | Cost |
|---|---|---|---|---|---|---|---|
| A1 | 국가법령정보 (현행법령, 판례, 행정규칙, 자치법규, 조약, 헌재결정례, 법령해석례, 행정심판례) | https://open.law.go.kr/LSO/openApi/guideList.do | **No copyright at all** - 저작권법 제7조 제1호~제4호 (보호받지 못하는 저작물) | ~191 API endpoints across 8 corpora; the full 현행법령 + 판례 body is the largest clean formal-Korean text in existence outside news | XML via Open API | **YES** - strongest position in this report; absence of rights, not a grant | Free. API key by application, 02-2109-6446 |
| A2 | 한국어 위키백과 dump | https://dumps.wikimedia.org/kowiki/latest/ | **Creative Commons Attribution-ShareAlike 4.0 International** (text; some legacy GFDL) | `kowiki-latest-pages-articles.xml.bz2` = **1,365,272,177 bytes (1.27 GiB compressed)**, dump of 2026-09-01 | XML (bz2) | **YES** with attribution. ShareAlike is a known question for weights; the industry position, and ours, is that weights are not a derivative of the text. We already ship this and disclose it in the model card | Free |
| A3 | 위키문헌 (Korean Wikisource) | https://dumps.wikimedia.org/kowikisource/latest/ | **Creative Commons Attribution-ShareAlike 4.0 International**; underlying source texts largely public domain | `kowikisource-latest-pages-articles.xml.bz2` = **147,611,492 bytes (141 MiB compressed)**, 2026-09-01 | XML (bz2) | **YES** | Free |
| A4 | 위키낱말사전 (kowiktionary) | https://dumps.wikimedia.org/kowiktionary/latest/ | **Creative Commons Attribution-ShareAlike 4.0 International** | 47,392,283 bytes (45 MiB compressed) | XML (bz2) | **YES** - high value per byte for a 32k Korean tokenizer and for morphology | Free |
| A5 | 위키책 (kowikibooks) | https://dumps.wikimedia.org/kowikibooks/latest/ | **Creative Commons Attribution-ShareAlike 4.0 International** | 4,863,554 bytes | XML (bz2) | **YES** - small but it is literally Korean textbook prose, the exact register we want | Free |
| A6 | 공공누리 AI유형 / 제0유형 공공저작물 (37 종, 33 기관) | https://www.kogl.or.kr/info/licenseTypeAi.do | **공공누리 AI유형: 인공지능 학습용** and **공공누리 제0유형: 자유이용** | 37 works so far: 정책자료·간행물, 각종 위원회 심결문·의결서, 판례집, 일부 공무원 시험문제. TOP100 programme adds 25 more in 2026, 65 in 2027 | mixed (PDF, HWP, XML) | **YES** - and the licence text explicitly permits commercial use of the trained model | Free. Condition: anti-memorisation technical measures; may not resell derived training data |

