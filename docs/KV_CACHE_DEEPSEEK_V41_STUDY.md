# KV cache: what DeepSeek-V4.1-Flash does, and what of it reaches a 0.8B on-device model

Written 2026-09-16 from the DeepSeek-V4.1-Flash technical report (Hub, 2026-09-10), its `config.json`, the
DeepSeek-V4-Flash `config.json`, and the Qwen3.5 / Ornith / Kanana-2 / EXAONE-4.0 / MiniCPM5 configs. Every byte
figure below is recomputed from the configs; where a number is the vendor's own it is marked so.

## 1. The model, in one table

| item | DeepSeek-V4.1-Flash | source |
|---|---|---|
| backbone | 552B MoE (384 routed + 1 shared experts, top-6, expert dim 2304), 196B Engram tables | report §2.1, §4.2.1 |
| active per token | 8B in prefill, 16B in decode | report abstract |
| layers | 40 = 20-layer causal encoder + 20-layer decoder; 3 extra DSpark draft layers | config `compress_ratios` (43 entries) |
| attention per layer | 64 query heads × head_dim 512, **one KV head**, K and V share one 512-d latent, RoPE on 64 of the 512 dims | config `num_key_value_heads: 1`, `head_dim: 512`, `qk_rope_head_dim: 64` |
| local branch | sliding-window attention in every layer, window 128, FP8 cache | config `sliding_window: 128`; report §2.4.4 |
| global branch | CSA2: layers 0-1 SWA only; encoder 2-19 compression ratio 2; decoder 20-39 ratio 1 | config `compress_ratios` = 0,0, 2×18, 1×20 |
| who owns global KV | 4 layers: 2, 8, 14 (encoder, Full) and 20 (decoder, Full). Every other global layer reuses one of these | config `kv_source_layer_ids: [2, 8, 14, 20]` |
| who owns indices | 8 layers: the 4 above plus decoder 24, 28, 32, 36 (Reindex) | config `index_source_layer_ids` |
| indexer | 32 heads × 128 dims, top-512 entries per query; decoder pool = 2,048 blocks × 8 positions = 16,384 candidates | config `index_*`, `candidate_*` |
| context | 1,048,576 (YaRN ×16 from 65,536) | config |
| global KV per token | **890 bytes** | report Fig. 1(b) |
| V4-Flash global KV per token | **3,514 bytes** | report Fig. 1(b) |

## 2. Where 890 bytes comes from (reconstruction; matches the report to the byte)

One global KV entry:

| part | format | bytes |
|---|---|---|
| main KV latent, 512 channels | E2M1 FP4 | 256 |
| its scales, one E4M3 per 16 channels | 32 × 1 B | 32 |
| indexer K, 128 channels | FP4 | 64 |
| its scale | 4 | 4 |
| **per entry** | | **356** |

Entries per token: three encoder Full layers at ratio 2 contribute 0.5 entry each (1.5), the decoder Full layer at
ratio 1 contributes 1.0. Total 2.5 entries × 356 B = **890 B**.

The same accounting on V4-Flash (43 layers; 21 layers at ratio 4 with an indexer, 20 layers at ratio 128 without;
main KV 576 channels FP8 + 8 B of scales = 584 B; indexer K 68 B):
21 × 0.25 × (584 + 68) + 20 × 584 / 128 = 3,423 + 91 = **3,514 B**. Both generations reconcile with one set of
assumptions, so the decomposition can be trusted.

SWA cache is separate and bounded: 40 layers × 128 positions × 512 channels × 1 B (FP8) ≈ 2.6 MB per sequence,
independent of length, never persisted (see §4).

## 3. The four multiplicative levers, and which generation added each

1. **Entry size.** One KV head instead of 8–64; K and V share a single latent (no 2× for V); 512 dims. A GQA layer
   with 2 KV heads × 256 dims in fp16 stores 2,048 B per token per layer; this stores 288 B per entry.
2. **Sequence.** Ratio-2 pooling in the encoder (two tokens → one entry; V4.1 dropped V4's overlapping 2m window and
   the absolute positional embedding in the compressor). V4 used ratios 4 and 128; V4.1 gets more from layer sharing
   than from pooling, and keeps the decoder at ratio 1.
3. **Layer.** Cross-layer reuse (CSA2 modes). *Full* computes main KV, projects indexer K from it, indexes. *Reindex*
   reuses KV and indexer K from the last Full layer but rescores with its own indexer Q. *Reuse* takes KV and the
   last Top-K indices as they are and runs sparse attention (15 kernels in prefill, 11 in decode). Combined with the
   causal encoder-decoder (CED): the decoder's Full layer projects its KV from the encoder's final hidden state, so
   prefill runs 20 layers, not 40 (YoCo lineage). 36 of 40 layers store no global KV at all.
4. **Precision.** FP4 main KV with quantization-aware training added in post-training, quantized after RoPE. The
   report justifies dropping NVFP4's second global scale: the 512-channel latent has L2 norm ≤ √512 ≈ 22.6 after
   RMSNorm and RoPE, observed max ≈ 10, far under E4M3 × E2M1's 2,688. SWA KV stays FP8 ("sensitive to quantization").

Compute follows the same logic: the top-512 selection bounds attention reads per query, and the hierarchical
candidate pool bounds the deeper decoder indexers to a constant number of scored positions, so single-token decode
FLOPs rise only 1/4 from 4K to 1M context (report Fig. 2).

## 4. Deployment: the part that is not architecture

- Persistent KV (SSD/host, 72 h lifetime) now holds **only global KV**. SWA KV lives in a host-DRAM pool (10% of
  each machine's DRAM, minute-scale TTL) and is never written to SSD. That halves persistent storage on top of the
  4× global reduction → the "1/8 persistent" claim.
- **SWA Bounded Replay.** On a cache hit that misses SWA state, replay only the last n_win = 128 tokens of the prefix
  and truncate the window to that segment. Exact reconstruction would need L × n_win = 5,120 tokens. The result is
  approximate and position-dependent; the report says quality loss is negligible and that post-training simulated
  the same replay so the model is trained for it. The decoder's SWA KV is never cached at all; every prefill replays
  128 tokens through the 20 decoder layers.
- Stated risk (report §6): CSA2 selection errors and approximate SWA reconstruction "may still cause capability
  degradation in untested boundary cases".

Runtime support as of 2026-09-16: vLLM and SGLang support is reported by third-party write-ups (not verified here);
llama.cpp conversion is an open PR (ggml-org/llama.cpp #28696) with runtime on a fork branch and sparse attention
unfinished, per the community GGUF card (vcruz305, 2026-09-12; Q2_K alone is 246 GB). Nothing here runs on a laptop.

## 5. Our models, same units

Bytes per token that grow with context (only the full-attention layers store per-token KV; Gated DeltaNet layers
keep a fixed recurrent state). q8_0 = 8.5 bits per element in llama.cpp.

| model | layers with growing KV | KV heads × dim | B/token fp16 | B/token q8_0 | fixed state |
|---|---|---|---|---|---|
| **Songgot-X 0.8B** (Qwen3.5-0.8B) | 6 of 24 | 2 × 256 | **12,288** | 6,528 | GDN 18 × 16×128×128 f32 ≈ 18.9 MB |
| Qwen3.5-2B | 6 of 24 | 2 × 256 | 12,288 | 6,528 | same ≈ 18.9 MB |
| Ornith-1.5-35B-A3B (Desktop Pro) | 10 of 40 | 2 × 256 | 20,480 | 10,880 | GDN 30 × 32×128×128 f32 ≈ 63 MB |
| Kanana-2-1.3B | 8 of 32 full; 24 sliding (w = 1,024) | 8 × 128 | 32,768 | 17,408 | sliding layers cap at ≈ 100 MB |
| EXAONE-4.0-1.2B | 30 of 30 | 8 × 64 | 61,440 | 32,640 | – |
| MiniCPM5-2B | 42 of 42 | 2 × 128 | 43,008 | 22,848 | – |
| Qwen3-0.6B | 28 of 28 | 8 × 128 | 114,688 | 60,928 | – |
| DeepSeek-V4.1-Flash (global) | 4 of 40 | 1 × 512, K = V | 890 (FP4, incl. indexer) | – | SWA ≈ 2.6 MB |

Songgot-X at its shipped contexts: 2,048 (Pocket) = 24 MB fp16; 4,096 (extension) = 48 MB; 32k = 384 MB;
262k = 3.1 GB fp16 / 1.6 GB q8_0. Ornith 35B at 8,192 = 160 MB; at 262k = 5.1 GB fp16 / 2.7 GB q8_0.

Two consequences. Among the Korean on-device tool-callers we benchmark, Songgot-X already has the smallest growing
KV per token (2.7× below Kanana-2, 5× below EXAONE), because Qwen3.5's 3:1 GDN hybrid is itself a KV design; Ornith's
"less KV cache" is this same Qwen3.5 property, not an Ornith technique. And the 2B has exactly the same KV cost as
the 0.8B, so a Songgot-X 2B tier costs nothing extra in context memory.

## 6. What transfers to us, and what does not

**Transfers without training (runtime, measurable this week).**
- Quantized attention KV in llama.cpp / wllama: `-ctk q8_0 -ctv q8_0 -fa on` (server) or `cache_type_k/v: "q8_0",
  flash_attn: true` (wllama). Halves the growing part; GDN state is untouched (llama.cpp only quantizes the
  non-recurrent layers). Community measurements on Qwen-family models put q8_0 KV at ΔPPL ≈ +0.002–0.005 and warn
  that K below q8_0 collapses; we measure it ourselves on FunctionChat with `harness/eval_gguf.py` before shipping.
- Prefix reuse = their "persistent global KV" at our scale. Every Pocket request re-prefills ~700 tokens of tool
  schema; measured 3.5 s per request of which KV memory is 9 MB. Reusing the prefix (llama-server `--cache-reuse`,
  slot save/restore; wllama's server-derived completion path) removes most of that time. This, not bytes, is the KV
  bottleneck we actually have at 2k–8k context.
- Bounded replay is what llama.cpp's context shift already is; nothing to add.

**Needs pre-training or uptraining (not for line B).**
- Cross-layer sharing of the 6 attention layers' KV (CLA / MLKV / YoCo-style): 6 → 2 KV-owning layers would give
  4,096 B/token fp16, 2,176 q8_0. MLKV-style uptraining needs on the order of 10⁹ tokens; a 0.8B post-train cannot
  absorb architecture surgery without a measured loss, and we do not pre-train Qwen3.5.
- One KV head with K = V sharing, and FP4 KV with QAT: same reason.
- For the parked from-scratch line the design brief is now clear: GDN 3:1 hybrid, one KV head, KV shared across the
  full-attention layers, FP8/q8 cache from the start. That would put a 50M–300M model near 1–2 KB/token.

**Does not apply at our context lengths.** Top-k sparse indexer and the hierarchical candidate pool (pay off when
context ≫ 4k; our prompts are ~800 tokens), the encoder-decoder prefill split (needs long prompts and deep stacks),
Engram and DSpark (not KV).

## 7. Actions

1. Measure Q4_K_M weights × {f16, q8_0, q4_0} KV on FunctionChat-Bench with `harness/eval_gguf.py` (same prompt
   rendering and parser as the bf16 predictor). Ship q8_0 KV in Desktop and the extension only if the call rate
   holds within noise; ship q4_0 nowhere unless it also holds.
2. Desktop: add `-fa on --cache-reuse 256` to the llama-server args (exact computation, prefill reuse across turns),
   then `-ctk q8_0 -ctv q8_0` after step 1.
3. Pocket / extension: pass `cache_prompt: true` through wllama's completion request if the wasm honours it
   (llama-server option), and measure prompt-eval time on the second request.
4. Paper: add the "growing KV bytes per token" column to Table 2 with the figures in §5.

## 8. Measured 2026-09-16, 21:50 KST: prefix reuse on the Desktop server

Bundled llama-server (0.4.0-dev build 71) with Songgot-X Q4_K_M, `-c 4096 -ngl 99 -fa on --cache-reuse 256`, an
8-tool schema prefix and three different user queries, `cache_prompt: true` (the server default):

| request | prompt tokens evaluated | prompt time |
|---|---|---|
| 1, cold | 2,925 | 1,224 ms |
| 2, same prefix, new query | 511 | 55 ms |
| 3, same prefix, new query | 514 | 143 ms |

The ~511 tokens re-evaluated on a hit are not the query: llama.cpp keeps the Gated DeltaNet recurrent state only at
checkpoints, restores the nearest one and replays the tokens since it. That is the same shape as DeepSeek's SWA
Bounded Replay (they replay 128 tokens of sliding-window state; we replay up to a checkpoint interval of GDN state).
`--cache-reuse 256` (KV shifting for partial-prefix hits) is reported by the server as unsupported for this hybrid
context and is silently disabled; it still applies to the dense Gemma chat tier. Prompt time per turn therefore drops
about 10–20× on every turn after the first, with no change to the model or its accuracy.

## 9. Measured 2026-09-16, 22:30 KST: quantized KV on FunctionChat-Bench

Songgot-X Q4_K_M GGUF through llama.cpp (llama-cpp-python 0.3.35, CPU, flash attention on), the same prompt
rendering and parser as the bf16 predictor, 500 SingleCall items (standard error about 1.9 points):

| weights | KV cache | call | name | exact / 4_random / 4_close / 8_random / 8_close |
|---|---|---|---|---|
| bf16 (published) | fp16 | 82.2 | 96.4 | 87 / 84 / 80 / 83 / 77 |
| Q4_K_M | f16 | 80.2 | 94.4 | 86 / 80 / 78 / 81 / 76 |
| Q4_K_M | q8_0 | 81.0 | 94.4 | 86 / 80 / 79 / 81 / 79 |
| Q4_K_M | q4_0 | 80.8 | 94.2 | 87 / 80 / 78 / 81 / 78 |

Reading: the 2-point step from bf16 to Q4_K_M weights is the whole cost of the on-device build and sits inside one
standard error; the KV cache type makes no measurable difference at these prompt lengths (about 800 tokens). q8_0 KV
ships in Desktop (halves the growing cache, GDN state untouched). q4_0 is recorded, not shipped: the community
evidence of K-cache collapse below q8_0 comes from long contexts this benchmark does not exercise.
