# Songgot: the investor questions, answered with measurements

Palette, 2026-09-12. Every number below is measured on a public benchmark with a public scorer, dated, and
reproducible from github.com/hanishkeloth/songgot. Where we lose, the number is here too.

## "Aren't you just fine-tuning someone else's model?"

No. We run two lines and label them.

**Line A, from scratch.** Songgot is a Korean-first language model whose every weight started at random
initialisation under our training: our own 32k Korean tokenizer, our own pretraining on licence-clean text, our
own post-training data written by our own open teacher and verified by a second pass. Three sizes exist today
(50M, 126M, 303M in training). Nothing in the shipped weights comes from another lab's model.

**Line B, on an open base.** For customers who need the best number today, we post-train a permissively licensed
open model (Apache 2.0) on the same data. Its card names the base. It is a product, not a research claim.

The from-scratch line is the one we grow. It is the only line that can be sovereign (no upstream licence, no
foreign tokenizer, no dependency on another company's release schedule), the only line whose data provenance we
control end to end, and the only line that can be made small enough for a browser tab or a phone without a
network.

## "How good is it?"

FunctionChat-Bench SingleCall (Kakao, 500 Korean items, 5 tool conditions), exact match on function name and
arguments under our own public scorer, measured 2026-09-11:

| model | params | who made it | call accuracy |
|---|---|---|---|
| Kanana-2-1.3B-Instruct | 1.3B | Kakao (also authored the benchmark) | 70.4 |
| EXAONE-4.0-1.2B | 1.28B | LG | 63.0 |
| Qwen3.5-0.8B | 0.8B | Alibaba | 45.2 |
| Qwen3-0.6B | 0.6B | Alibaba | 43.2 |
| **Songgot-M (from scratch)** | **126M** | **Palette** | **33.2** |
| **Songgot (from scratch)** | **50M** | **Palette** | **33.0** |
| DNA3.0-0.8B | 0.8B | Dnotitia | 4.8 |
| FunctionGemma-270M | 270M | Google | 2.2 |
| Needle 2 | 45M | Needle | 0.0 |

Read it plainly: at 50M parameters we are at half of Kakao's 1.3B model and 25 times smaller. Below 200M
parameters no Korean model has published a tool-calling number at all. Above 1B, Kakao and LG lead.

## "Is the claim defensible?"

We swept the Hugging Face Hub, Korean press and Korean search on 2026-09-11 (docs/PRIOR_ART_2026-09-11.md). We
do not claim "first Korean tool-calling model" (Kakao and LG shipped first), "first from-scratch tiny Korean
model" (three exist), or "best" (we are not). We claim: the smallest openly licensed Korean model trained from
scratch for tool calling with a published FunctionChat-Bench number, at 50M and 126M parameters, running offline
in a browser. Every word in that sentence is load-bearing.

## "Why will it keep improving?"

Because the curve has not stopped moving and we know which lever moves it. In 48 hours the same 50M weights went
11.4 -> 23.8 -> 27.6 -> 32.0 -> 33.0 as we replaced hand-collected post-training data with teacher-written,
verified Korean data (now 336,602 pairs over 183,737 tool schemas, released as a dataset). That lever is now flat
on the 6B-token base, and we measured that too. The next lever is pretraining: more parameters and more clean
Korean text. Songgot-L (303M) finishes today; a 200 GB private Korean document corpus, verified by hash, is
ingested and tokenized; 245,791 open Korean document pages and 1 TB of synthetic Korean OCR are on the volume.

## "What about vision?"

Songgot-V is a Korean document-understanding model: our from-scratch language model, an open pretrained vision
tower (SigLIP2, Apache 2.0, stated on the card), and a projector trained from random initialisation. The first
alignment run on 245,791 Korean pages shows the model reads the image (loss with the right page 3.67, with a
shuffled page 5.54). We found no Korean vision-language model under 1B parameters from any Korean lab; the
smallest is 1.7B. Benchmarks: NCSOFT K-DTCBench, K-MMBench, K-SEED, K-MMStar; baselines for five models are in
the repository.

## "What can go wrong?"

Kakao or LG could release a sub-200M tool-calling model tomorrow and take the size claim. A from-scratch model
may never match a 1.3B model trained on trillions of tokens at 50M parameters; the honest target is the best
model per parameter and per byte, not the best model. Korean data law has no text-mining exception, so every
byte we train on needs a licence or ownership, and we keep a provenance file for every corpus. Modal compute is
about USD 300 a day at the current pace.
