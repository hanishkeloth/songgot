# Korean VLM baselines, 2026-09-10

harness/kvlm_modal.py, 1,000 items per benchmark (K-DTCBench all 240), NCSOFT K-* sets (CC BY-NC 4.0), letter-likelihood scoring, H100, greedy.

| model | K-MMBench | K-SEED | K-MMStar | K-DTCBench |
|---|---|---|---|---|
| NCSOFT/VARCO-VISION-2.0-1.7B | 0.783 | 0.715 | 0.533 | 0.646 |
| google/gemma-4-E2B-it | 0.675 | 0.660 | 0.434 | 0.517 |
| Qwen/Qwen3.5-0.8B | 0.668 | 0.622 | 0.444 | 0.525 |
| LiquidAI/LFM2.5-VL-450M | 0.610 | 0.624 | 0.432 | 0.329 |
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct | 0.302 | 0.268 | 0.214 | 0.304 |
