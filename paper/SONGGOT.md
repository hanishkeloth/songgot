# Songgot (송곳): a Korean-first tiny agentic model for tool calling on the device

Hanish Keloth, Palette (Seoul and Bengaluru). Draft of 2026-09-10. Numbers marked TBD are filled
from the run logs before release; nothing here is estimated.

## Abstract

Tiny agentic models (Needle 2, 45M parameters, 14 MB, July 2026) made on-device tool calling
practical on phones and microcontrollers, but the released model is English only. Songgot is a
Korean-first model in the same class: a decoder trained from scratch with a Korean-first 32k
tokenizer, bilingual Korean and English pretraining on licence-clean text, and post-training on
Korean tool-calling and structured-extraction data that no closed model touched. We evaluate on
Kakao's FunctionChat-Bench SingleCall (500 Korean items) with a deterministic exact-match scorer
that anyone can reproduce without an API key, against Needle 2, FunctionGemma-270M and Qwen3-0.6B
under identical prompts. Results for the comparators are measured and in Table 2; Songgot rows are filled from the run logs as each training run finishes (Songgot-nano on 2026-09-10). Weights, tokenizer, data generator, scorer and this paper
are released under Apache 2.0.

## 1. Why a Korean-first tiny model

- The class exists and is growing: Needle 2 passed 52,000 downloads within weeks of release.

- On Hugging Face as of 2026-09-09 there is no Korean tool-calling model under 1B parameters;
  the one Korean function-calling finetune found is a 2B Gemma.

- Korean is expensive for English-first tokenizers. On the 100 FunctionChat SingleCall queries,
  Qwen3's 151k vocabulary spends 1.15 tokens per Hangul syllable, Gemma's 262k vocabulary 0.98,
  and Songgot's 32k Korean-first vocabulary 0.90 (Table 1). At a 256 to 1024 token budget shared
  with tool schemas, that is context.

## 2. Model

Llama-style decoder, hidden 512, GQA with 8 query and 2 key-value heads, SwiGLU intermediate
1408, RoPE, tied embeddings, 32k vocabulary, trained from scratch. Two sizes: Songgot-nano, 8
layers, 39M parameters, pretrained on an Apple M5 Max with MLX (320M tokens); Songgot, 12 layers,
about 50M parameters, 8xH100 run on 6B tokens queued behind GPU budget.
Exports to safetensors, GGUF (f16, Q8_0, Q4_K_M) and runs under llama.cpp.

![Figure 3. Songgot-nano pretraining loss on the Mac, read from the run log](fig3_loss.png)

## 3. Tokenizer

SentencePiece BPE, 32,000 pieces, byte fallback, digits split, trained on a 300 MB sample:
Korean Wikipedia 50 percent, fineweb-edu 40 percent, tool-schema JSON 10 percent. Special
tokens `<|system|> <|user|> <|call|> <|end|> <|pad|>`.

Table 1. Tokens per Hangul syllable on the 100 FunctionChat SingleCall queries (2,071 syllables).

| tokenizer | vocab | tokens | tokens per syllable |
|---|---|---|---|
| Songgot (ours) | 32k | 1,873 | 0.90 |
| Gemma 3 / FunctionGemma | 262k | 2,028 | 0.98 |
| Qwen3 | 151k | 2,392 | 1.15 |
| Needle 2 | 8k | 7,192 | 3.47 |

![Figure 1. Tokens per Hangul syllable on the FunctionChat queries, lower is better](fig1_tokens.png)

## 4. Data

Pretraining (all disclosed, all licence-clean):

- fineweb-edu sample-10BT (ODC-By), first 1.63B tokens under our tokenizer.

- Korean Wikipedia, dump 20231101.ko (CC BY-SA 3.0), 0.60B tokens, upsampled to roughly half
  of the training mix.

- No AI-Hub data (its terms forbid leaving Korea and our GPUs do not sit there), no crawled
  Korean web text of unclear licence.

![Figure 4. Pretraining corpus in tokens under the Songgot tokenizer](fig4_data.png)

Post-training (85,408 examples): 64,000 Korean tool calls generated from hand-written frames
for 58 Korean tools (messaging, calendar, transit, home appliances, commerce, public services,
health, work, information), with register variants (반말, 존댓말, 격식, 사투리, typos), Korean date
and time words resolved against a fixed calendar, and 4,000 "no tool applies" negatives;
21,408 English examples from glaive-function-calling-v2 (Apache 2.0). No closed model produced
a label. A disjointness gate aborts the build if any training tool name or query appears in
FunctionChat-Bench; three tool names had to be renamed because of it.

## 5. Evaluation

FunctionChat-Bench SingleCall (Kakao, 2024, Apache 2.0): 25 functions, 4 Korean queries each,
5 tool conditions (exact, 4 random, 4 close, 8 random, 8 close) = 500 items. The official
protocol uses GPT-4 as judge; we score with exact match on function name and arguments (numbers
compared as numbers, acceptable alternatives honoured) so the number is reproducible offline.
Every model receives the same tools and query, rendered in its own documented format.

Table 2. Call accuracy (exact match) on SingleCall, by tool condition, percent. Songgot rows are
written by the build from the run logs; a row reads "training" until its run has finished.

| model | params | exact | 4_random | 4_close | 8_random | 8_close | all | name only |
|---|---|---|---|---|---|---|---|---|
| Songgot-nano | 39M | training | | | | | | |
| Needle 2 | 45M | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| FunctionGemma-270M | 270M | 3.0 | 5.0 | 1.0 | 1.0 | 1.0 | 2.2 | 36.2 |
| Qwen3-0.6B | 600M | 48.0 | 49.0 | 45.0 | 37.0 | 37.0 | 43.2 | 70.8 |

![Figure 2. Call accuracy by tool condition on FunctionChat-Bench SingleCall](fig2_bench.png)

## 6. Honest reading

- Songgot-nano sees 320M pretraining tokens; the Needle 2 recipe used 200B, about 600 times
  more. We expect that gap to show first in argument values (paraphrased places and times) and in
  rare tools, and we report every condition whatever it says. Where Songgot loses, this section
  will say so with the failing items named.

- The template-generated Korean data is narrow by construction; a teacher-generated set from
  our own Palette-K-Midm was planned and is queued behind GPU availability.

- The preview checkpoint used to open the demo on 2026-09-10 is an early one (65M tokens,
  24,000 post-training examples) and is labelled as such in the table.

## 7. Release

Weights and tokenizer: hf.co/imcapsule/songgot. Code, data generators, scorer, paper:
github.com/hanishkeloth/songgot. Licence Apache 2.0. Korean Wikipedia attribution and CC BY-SA
notice in the model card.
