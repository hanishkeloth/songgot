# Songgot (송곳): a Korean-first tiny agentic model

Overnight build, 2026-09-09 22:20 KST start, deliverable by 2026-09-10 morning. User ask: "one
open source best tiny model like Needle 2, our own, publish research paper, open weights, open
source, unique, trending, under my name and GitHub, proper research as of today, not outdated".

## Positioning (claims only after measurement; comparators named; dated)
- Class: Needle 2 (Cactus Compute, Jul 2026, 45M params, 14 MB, tool calling / device use /
  structured extraction, English only, Apache 2.0). Category is live: 52k downloads in weeks.
- Songgot: Korean-first, bilingual KO/EN, ~50M params, from scratch, own Korean-first tokenizer,
  own Korean agentic post-training data, evaluated on Kakao FunctionChat-Bench SingleCall
  (Apache 2.0, 500 items, deterministic exact-match scoring we implement) plus our own set.
- Prior art on HF as of 2026-09-09: no Korean tool-calling model under 1B (one 2B Gemma
  finetune). Needle 2 itself is English only. Naver sweep still owed before any "first" claim.

## Hard rules
- Licence-clean data only: fineweb-edu (ODC-By) for English, Korean Wikipedia 20231101.ko
  (CC BY-SA 3.0, disclosed), synthetic Korean agentic data from our own Palette-K-Midm (open
  weights, ours). No AI-Hub bytes. No closed-model outputs as labels. FunctionChat-Bench is
  TEST ONLY: disjointness gate on every training example against its queries and tool names.
- No base-model licence issue: from scratch, so any Modal region is fine.
- Single hyphens, dated numbers, comparators in the same table, instrument floors stated.

## Timeline (KST)
- 22:20-23:30 scaffold, tokenizer + corpus prep on Modal (CPU), tool catalog, teacher up.
- 23:30-01:00 synthetic Korean tool data (Midm on vLLM), pretrain launch on 8xH100.
- 01:00-04:00 pretrain ~12B tokens (EN 8B, KO 3B upsampled, tools mixed late).
- 04:00-04:45 post-train (full SFT) on tool calling + extraction, KO + EN.
- 04:45-06:30 eval: Songgot vs Needle 2, FunctionGemma-270M, Qwen3-0.6B on FunctionChat
  SingleCall exact-match; tokens-per-syllable study; on-device speed (GGUF, llama.cpp, Mac +
  phone-class CPU).
- 06:30-08:00 export GGUF/ONNX, HF repo (imcapsule/songgot), GitHub (IMCapsule-ai/songgot),
  paper (paper/SONGGOT.md + PDF), site, memory notes.

## Architecture (decided)
Llama-style decoder (HF LlamaConfig): 12 layers, d_model 512, 8 heads, 2 KV heads (GQA),
intermediate 1408, vocab 32k, context 1024 (train), RoPE. ~50M params. Reason: exports to
GGUF/llama.cpp and ONNX today, runs on phones via llama.cpp and Cactus; attention-only SAN
from the Needle paper is a follow-up ablation, not tonight's risk.

## Tokenizer (the Korean-first decision)
SentencePiece BPE, 32k, trained on kowiki 45% + fineweb-edu 35% + synthetic Korean agentic
10% + JSON/tool schemas 10%, byte fallback, all 2,350 KS X 1001 syllables forced as user
symbols. Metric reported: tokens per Hangul syllable on FunctionChat queries versus Needle 2,
Gemma and Qwen tokenizers.
