# Korean VLM baselines, 2026-09-10

harness/kvlm_modal.py, 1,000 items per benchmark (K-DTCBench all 240), NCSOFT K-* sets (CC BY-NC 4.0), letter-likelihood scoring, H100, greedy.

| model | K-MMBench | K-SEED | K-MMStar | K-DTCBench |
|---|---|---|---|---|
| google/gemma-4-26B-A4B-it | 0.872 | 0.772 | 0.655 | 0.921 |
| Qwen/Qwen3.5-9B | 0.876 | 0.792 | 0.659 | 0.858 |
| google/gemma-4-12B-it | 0.838 | 0.768 | 0.663 | 0.842 |
| Qwen/Qwen3.5-4B | 0.819 | 0.767 | 0.571 | 0.738 |
| google/gemma-4-E4B-it | 0.783 | 0.745 | 0.520 | 0.742 |
| NCSOFT/VARCO-VISION-2.0-1.7B | 0.783 | 0.715 | 0.533 | 0.646 |
| google/gemma-4-E2B-it | 0.675 | 0.660 | 0.434 | 0.517 |
| Qwen/Qwen3.5-0.8B | 0.668 | 0.622 | 0.444 | 0.525 |
| LiquidAI/LFM2.5-VL-450M | 0.610 | 0.624 | 0.432 | 0.329 |
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct | 0.302 | 0.268 | 0.214 | 0.304 |

2026-09-17: two Ornith 1.5 models (Ornith AI, MIT, Qwen3.5 lineage with continued pretraining and a self-improvement RL loop) measured
on the same protocol. Ornith-1.5-35B-A3B (MoE, 3B active): K-MMBench 0.875, K-SEED 0.785, K-MMStar 0.681, K-DTCBench 0.892, mean 0.808,
a tie with Gemma 4 26B-A4B on the mean (0.805), stronger on natural images and weaker on documents; it is the Palette Desktop Pro
tier for 32 GB+ machines (Q4_K_M 20.7 GB + 0.9 GB projector). Ornith-1.5-9B: 0.850 / 0.777 / 0.641 / 0.838, mean 0.776, below
Qwen3.5-9B (0.796) at the same size, not used. Result files eval/kvlm/ornith15_35b_a3b.json and eval/kvlm/ornith15_9b.json.

2026-09-14: five larger candidates added (same protocol, 1000 items per benchmark, 240 for K-DTCBench) to pick the Pro tier of Palette Desktop. Ranking by mean: Gemma 4 26B-A4B 0.805, Qwen3.5-9B 0.796, Gemma 4 12B 0.778, Qwen3.5-4B 0.724, Gemma 4 E4B 0.697. Gemma 4 26B-A4B is strongest on documents and charts (K-DTCBench 0.921); Qwen3.5-9B on natural images (K-MMBench 0.876, K-SEED 0.792).
