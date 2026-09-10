# Songgot (송곳)

**A Korean-first tiny agentic model for tool calling on the device.** Same class as Needle 2
(tens of millions of parameters, a few tens of MB, phone-speed), trained from scratch with a
Korean-first tokenizer, licence-clean data, and no closed-model labels.

- Weights and tokenizer: `palette-lab/songgot` on Hugging Face (Apache 2.0)
- Paper: `paper/SONGGOT.md`
- Benchmark: Kakao FunctionChat-Bench SingleCall, scored by exact match (`eval/functionchat_exact.py`)

Try it on the device: https://hanishkeloth.github.io/songgot/app/ (runs in the browser, works offline after the first load).

## Figures

![Tokens per Hangul syllable](docs/fig1_tokens.png)

![FunctionChat-Bench call accuracy by condition](docs/fig2_bench.png)

![Songgot-nano pretraining loss](docs/fig3_loss.png)

## Why

Korean is expensive for English-first tokenizers. On the 100 FunctionChat SingleCall queries:

| tokenizer | vocab | tokens per Hangul syllable |
|---|---|---|
| Songgot (ours) | 32k | 0.90 |
| Gemma 3 / FunctionGemma | 262k | 0.98 |
| Qwen3 | 151k | 1.15 |
| Needle 2 | 8k | 3.47 |

Needle 2 spends nearly four times our tokens on the same Korean request and, in our run, could
not answer a single Korean item.

## Results

See the paper for the full table (updated from run logs, never estimated).

## Reproduce

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python harness/gen_templates_ko.py --n 60000 --out data/ko_raw.jsonl     # Korean tool data, no teacher
.venv/bin/python harness/assemble_sft.py --ko data/ko_raw.jsonl --out data/sft_train.jsonl
SONGGOT_VOL=$PWD/vol .venv/bin/python harness/train_mlx.py pretrain --tokens 3.2e8 --layers 8 --bf16   # Apple silicon
modal run harness/songgot_modal.py::pretrain                                                            # 8x H100 path
.venv/bin/python eval/functionchat_exact.py predict --backend songgot --model vol/ckpt/sft_mlx/final --out preds.jsonl
```

## Data provenance

fineweb-edu sample-10BT (ODC-By), Korean Wikipedia 20231101.ko (CC BY-SA 3.0, attribution in the
model card), glaive-function-calling-v2 (Apache 2.0), and template-generated Korean tool calls
(this repo, Apache 2.0). FunctionChat-Bench is used for testing only; a disjointness gate in
`harness/assemble_sft.py` refuses any overlap of tool names or queries.

## Licence

Apache 2.0. Songgot is a Palette project by Hanish Keloth.

## About

Songgot is built and maintained by **Hanish Keloth**, CTO at [Palette](https://www.pltt.xyz) (Seoul and Bengaluru), where he leads Palette OS, a company operating system in which governed AI agents draft the documents Korean companies run on. Palette publishes its Korean models and benchmarks in the open: Palette-K-Midm (first of seven on PALETTE-BENCH-KO v0.2, 2026-08-21), Palette-K-Doc, Palette-K-Speech and Palette Video, all on [hf.co/palette-lab](https://huggingface.co/palette-lab). Songgot continues that line at the smallest possible size: a Korean-first agentic model that runs on the device.

- GitHub: [github.com/hanishkeloth](https://github.com/hanishkeloth)
- LinkedIn: [linkedin.com/in/hanishkeloth](https://www.linkedin.com/in/hanishkeloth/)
- Hugging Face: [huggingface.co/palette-lab](https://huggingface.co/palette-lab)
- Paper: [hanishkeloth.github.io/songgot](https://hanishkeloth.github.io/songgot/)

Questions, issues and pull requests are welcome in this repository.
