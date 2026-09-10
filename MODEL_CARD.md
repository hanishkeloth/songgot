---
license: apache-2.0
language:
- ko
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- tool-calling
- function-calling
- korean
- on-device
- tiny
datasets:
- HuggingFaceFW/fineweb-edu
- wikimedia/wikipedia
- glaiveai/glaive-function-calling-v2
---
# Songgot (송곳)

A Korean-first tiny agentic model for tool calling on the device, trained from scratch by Hanish Keloth (Palette). Apache 2.0.

- Paper: https://hanishkeloth.github.io/songgot-
- Code, data generators, scorer: https://github.com/hanishkeloth/songgot-
- Demo: https://huggingface.co/spaces/imcapsule/songgot

## Numbers
RESULTS_TABLE

## Format
```
<|system|>
[{"name": "set_alarm", "description": "알람을 설정합니다.", "parameters": {...}}]
<|user|>
내일 아침 7시에 알람 맞춰줘
<|call|>
{"name":"set_alarm","arguments":{"time":"07:00"}}<|end|>
```
Tokenizer: SentencePiece BPE, 32k, byte fallback (`tokenizer.model`). Use `sentencepiece` directly; the special tokens live inside the vocabulary.

## Data and provenance
fineweb-edu sample-10BT (ODC-By), Korean Wikipedia 20231101.ko (CC BY-SA 3.0; this model card carries the attribution and share-alike notice for that text), glaive-function-calling-v2 (Apache 2.0), template-generated Korean tool calls (Apache 2.0, in the repo). No closed-model outputs. FunctionChat-Bench was never used for training.

## Limits
Single-call tool selection and argument extraction only. No multi-turn, no tool results, no free chat. Small models are finicky with rare tools and paraphrased values; validate every call in application code.
