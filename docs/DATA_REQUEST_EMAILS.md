# Data request drafts (2026-09-12)

Three requests, Korean first with an English version underneath where the counterparty is likely to read English.
Sender: Hanish Keloth, CTO, Palette (hanish@pltt.xyz). Entry points to verify before sending are noted; I did not
confirm exact recipient addresses. Nothing here has been sent.

---

## 1. 한국언론진흥재단 빅카인즈 (BigKinds) - AI 학습 목적 뉴스 데이터 이용 협의

**Where:** bigkinds.or.kr > 고객센터 / 데이터 이용 문의 (전화 02-2001-7114 대표), or the 데이터 제휴 form on the
site. Ask to be routed to 뉴스빅데이터 사업 담당.

**제목:** 한국어 소형 언어모델 사전학습용 뉴스 텍스트 이용 협의 요청 (팔레트)

담당자님께,

팔레트(Palette)의 CTO 하니시 켈로스입니다. 저희는 한국어에 특화된 소형 언어모델을 처음부터(from scratch)
학습하여 오픈 웨이트(Apache 2.0)로 공개하고 있습니다. 현재 공개된 모델은 5천만·1억2천6백만 파라미터 규모이며,
카카오 FunctionChat-Bench 기준 결과와 학습 데이터 출처를 모두 논문과 모델 카드에 명시하고 있습니다
(github.com/hanishkeloth/songgot).

저희 실험에서 한국어 뉴스와 같이 편집을 거친 정제 텍스트가 일반 웹 크롤 텍스트보다 소형 모델 성능에 훨씬 크게
기여함을 확인하였고, 이에 빅카인즈 뉴스 텍스트를 사전학습 데이터로 이용할 수 있는지 협의를 요청드립니다.

문의 사항:
1. AI 모델 학습 목적의 뉴스 본문 텍스트 대량 제공(벌크 익스포트) 가능 여부와 범위(기간, 매체, 분량)
2. 학습된 모델 가중치의 상업적 배포(오픈 웨이트 공개 및 상용 서비스 탑재)를 허용하는 계약 조건
3. 저작권자(각 언론사) 허락 범위와 저작권법상 필요한 절차, 이용 요금 및 견적 절차
4. 학습 시 원문 재현(암기) 방지 조치 등 재단이 요구하는 기술적 조건

학습은 원문을 재배포하지 않으며, 데이터 출처는 모델 카드와 논문에 명시합니다. 원하시면 저희의 데이터 출처 관리
방식(파일 단위 해시·라이선스 기록)을 공유드리겠습니다. 편하신 시간에 통화 또는 미팅 가능합니다.

감사합니다.
하니시 켈로스 / Palette CTO / hanish@pltt.xyz

---

## 2. Nexdata - Korean OCR dataset quote

**Where:** nexdata.ai > "Contact us" / sales form; the Hugging Face org Nexdata-kr links to the same sales team.
Reference the sample repos by name so they know which sets.

**Subject:** Quote request: Korean OCR datasets (handwriting, natural-scene, multilingual) for commercial model training

Hello Nexdata team,

I am the CTO of Palette, a Korean AI company training small open-weight Korean models. We are building a Korean
document-understanding vision-language model and would like a quote and licence terms for the following sets,
whose samples you publish on Hugging Face under Nexdata-kr:

1. 5,711 Images Korean Handwriting OCR Dataset
2. 104,320 Images Korean and Hindi OCR Data in Natural Scenes (Korean portion, or the full set if not separable)
3. 500,000 Images Multilingual OCR Dataset in 21 Languages (Korean portion, or the full set)

Please confirm for each: total image count and annotation format (line or word boxes, transcriptions), price for
a perpetual licence that permits training a model whose weights are released under Apache 2.0 and deployed
commercially, any restriction on derivative datasets, delivery format and lead time, and whether the images contain
identifiable people or personal data (we exclude those).

If you have other Korean document or receipt OCR sets not on Hugging Face, please include them.

Regards,
Hanish Keloth, CTO, Palette (Seoul / Bengaluru), hanish@pltt.xyz

---

## 3. 국립국어원 모두의 말뭉치 - 말뭉치 이용 신청 및 상업적 이용 문의

**Where:** corpus.korean.go.kr > 로그인 후 각 말뭉치별 "이용 신청" (신청서에 이용 목적 기재). 상업적 이용은
말뭉치마다 조건이 달라 별도 문의: 국립국어원 언어정보과 (kcorpus@korea.kr 로 안내되는 경우가 많으니 사이트의
문의처를 확인).

**제목:** 모두의 말뭉치 이용 신청 관련 문의 - 소형 언어모델 학습 및 오픈 웨이트 공개

국립국어원 언어정보과 담당자님께,

팔레트의 CTO 하니시 켈로스입니다. 한국어 소형 언어모델(5천만~3억 파라미터)을 처음부터 학습하여 Apache 2.0
오픈 웨이트로 공개하는 연구·개발을 진행하고 있습니다. 모두의 말뭉치 중 다음 말뭉치의 이용을 신청하고자 하며,
신청 전에 이용 조건을 확인드립니다.

관심 말뭉치: 신문 말뭉치(2020~), 문어 말뭉치, 구어 말뭉치, 일상 대화 말뭉치, 메신저 말뭉치, 웹 말뭉치,
국어 정보 처리용 어휘·의미 관련 말뭉치.

문의 사항:
1. 위 말뭉치를 언어모델 학습 데이터로 이용하는 것이 이용 약관상 허용되는지, 말뭉치별 차이가 있는지
2. 학습된 모델 가중치를 오픈 웨이트로 공개하고 상용 서비스에 탑재하는 것(상업적 이용)의 허용 여부와 절차
3. 이용 신청서에 기재해야 할 이용 목적의 형식과 심사 기간
4. 원문 재배포 금지 외에 요구되는 기술적·계약상 조건(예: 출처 표기 방식)

원문 자체는 재배포하지 않으며, 논문과 모델 카드에 국립국어원 말뭉치 이용 사실을 명시할 예정입니다.

감사합니다.
하니시 켈로스 / Palette CTO / hanish@pltt.xyz
