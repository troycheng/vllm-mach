# MXFP8 serving benchmark

Qwen3.5-2B and Qwen3.5-4B on one RTX 5090 per model, TP1. Each table reports **output tokens/s at an individual request concurrency** and compares that configuration with the historical open-source **block FP8** reference.

Launcher defaults: 4B uses **FP32 SSM / FP8 E4M3 KV** when both dtype flags are omitted. Changing the default GDN backend to ordered does not change either storage dtype. 2B fixes KV to BF16. Stock vLLM defaults to `auto` (model dtype). The 4B baseline explicitly passes `--kv-cache-dtype bfloat16`.

The baseline is **FP32 SSM state and BF16 KV**, with MXFP8 model weights. SSM precision reduction means FP32 to BF16; KV quantization means BF16 to E4M3 FP8. The 4B progression retains BF16 SSM when enabling FP8 KV. The 2B matrix retains BF16 KV.

Current MXFP8 measurements use native revision `ce0cd05` and the same fused SwiGLU/MXFP8 quantization and hybrid greedy-head optimizations. The production graph and PDL selectors remain active. Both SSM dtypes use ordered GDN in the main matrix; BF16 rounds dense state after each pending token. Stock BF16 comparisons remain in the recheck section.

Block FP8 values are retained historical open-source measurements, not fresh paired trials. The 2B reference used another RTX 5090. Gains describe complete configurations; single-run differences have no repeated-trial confidence interval.

[Machine-readable data](data/mxfp8-benchmark.json) retain exact rates, per-point latency metrics, request counts, artifact hashes and runtime receipts.

## Qwen3.5-4B

![Qwen3.5-4B per-concurrency throughput](assets/qwen35-4b-throughput.svg)

### FP32 SSM / BF16 KV

| Concurrency | Open-source block FP8 | MXFP8 | Gain vs block FP8 |
|---:|---:|---:|---:|
| c4 | 683.96 | 935.12 | +36.72% |
| c16 | 2053.55 | 2501.60 | +21.82% |
| c24 | 2559.26 | 3099.74 | +21.12% |
| c32 | 2887.82 | 3459.35 | +19.79% |
| c48 | 3220.55 | 3907.33 | +21.33% |
| c64 | 3402.76 | 4194.31 | +23.26% |

Launch flags: `--ssm-dtype float32 --kv-cache-dtype bfloat16`.

### BF16 ordered SSM / BF16 KV

| Concurrency | Open-source block FP8 | MXFP8 | Gain vs block FP8 |
|---:|---:|---:|---:|
| c4 | 683.96 | 953.88 | +39.46% |
| c16 | 2053.55 | 2619.71 | +27.57% |
| c24 | 2559.26 | 3260.84 | +27.41% |
| c32 | 2887.82 | 3665.11 | +26.92% |
| c48 | 3220.55 | 4175.26 | +29.64% |
| c64 | 3402.76 | 4546.23 | +33.60% |

Launch flags: `--ssm-dtype bfloat16 --ordered-bf16-ssm --kv-cache-dtype bfloat16`.

### BF16 ordered SSM / FP8 KV

| Concurrency | Open-source block FP8 | MXFP8 | Gain vs block FP8 |
|---:|---:|---:|---:|
| c4 | 683.96 | 978.79 | +43.11% |
| c16 | 2053.55 | 2862.04 | +39.37% |
| c24 | 2559.26 | 3630.11 | +41.84% |
| c32 | 2887.82 | 4153.06 | +43.81% |
| c48 | 3220.55 | 4770.53 | +48.13% |
| c64 | 3402.76 | 5367.96 | +57.75% |

Launch flags: `--ssm-dtype bfloat16 --ordered-bf16-ssm --kv-cache-dtype fp8_e4m3`.

## Qwen3.5-2B

![Qwen3.5-2B per-concurrency throughput](assets/qwen35-2b-throughput.svg)

### FP32 SSM / BF16 KV

| Concurrency | Open-source block FP8 | MXFP8 | Gain vs block FP8 |
|---:|---:|---:|---:|
| c4 | 1261.89 | 1682.89 | +33.36% |
| c16 | 3814.33 | 5134.27 | +34.60% |
| c24 | 5069.68 | 6631.15 | +30.80% |
| c32 | 5870.22 | 7610.93 | +29.65% |
| c48 | 6837.60 | 8963.55 | +31.09% |
| c64 | 7578.86 | 9811.23 | +29.46% |

Launch flags: `--ssm-dtype float32`.

### BF16 ordered SSM / BF16 KV

| Concurrency | Open-source block FP8 | MXFP8 | Gain vs block FP8 |
|---:|---:|---:|---:|
| c4 | 1261.89 | 1720.48 | +36.34% |
| c16 | 3814.33 | 5406.61 | +41.74% |
| c24 | 5069.68 | 7086.18 | +39.78% |
| c32 | 5870.22 | 8086.39 | +37.75% |
| c48 | 6837.60 | 9547.91 | +39.64% |
| c64 | 7578.86 | 10463.62 | +38.06% |

Launch flags: `--ssm-dtype bfloat16 --ordered-bf16-ssm`.

## Baseline and BF16 GDN recheck

2B variants run sequentially on GPU 2 and 4B variants sequentially on GPU 3. Each uses a fresh service and the same updated native binary, prompts, graph policy and KV budget. Grouped BF16 GDN and GDN PDL are disabled; production projection PDL remains unchanged. Ordered BF16 is now the default BF16 backend; `--no-ordered-bf16-ssm` selects stock GDN. Measurements predate the default change.

Worker receipts confirm FP32 state and ready ordered GDN for the baseline, BF16 state and ready ordered GDN for the main BF16 matrix. Stock measurements are retained below as an explicit backend comparison. The main five configuration tables above use this recheck. The earlier raw matrix remains archived.

Change from the preceding matrix baseline (same configuration, different GPU):

| Concurrency | 2B FP32 / BF16 KV | 4B FP32 / BF16 KV |
|---:|---:|---:|
| c4 | -1.01% | +0.08% |
| c16 | +2.52% | +0.08% |
| c24 | +2.38% | -0.04% |
| c32 | +2.08% | -0.03% |
| c48 | +2.30% | +0.23% |
| c64 | +2.21% | +0.40% |

The dtype, cache and GDN route checks pass in both runs. Cross-card differences above describe observed variability; the state comparisons below use the fresh baseline on the same GPU.

### 2B route comparison (output tokens/s)

| Concurrency | FP32 / BF16 KV | BF16 stock / BF16 KV | BF16 ordered / BF16 KV |
|---:|---:|---:|---:|
| c4 | 1682.89 | 1696.51 | 1720.48 |
| c16 | 5134.27 | 5318.61 | 5406.61 |
| c24 | 6631.15 | 6929.11 | 7086.18 |
| c32 | 7610.93 | 7884.13 | 8086.39 |
| c48 | 8963.55 | 9378.64 | 9547.91 |
| c64 | 9811.23 | 10182.11 | 10463.62 |

### 4B route comparison (output tokens/s)

| Concurrency | FP32 / BF16 KV | BF16 stock / BF16 KV | BF16 ordered / BF16 KV | BF16 stock / FP8 KV | BF16 ordered / FP8 KV |
|---:|---:|---:|---:|---:|---:|
| c4 | 935.12 | 956.44 | 953.88 | 981.41 | 978.79 |
| c16 | 2501.60 | 2632.95 | 2619.71 | 2876.85 | 2862.04 |
| c24 | 3099.74 | 3302.98 | 3260.84 | 3701.35 | 3630.11 |
| c32 | 3459.35 | 3732.87 | 3665.11 | 4249.73 | 4153.06 |
| c48 | 3907.33 | 4238.09 | 4175.26 | 4854.81 | 4770.53 |
| c64 | 4194.31 | 4584.45 | 4546.23 | 5381.74 | 5367.96 |

### 4B SSM precision with FP8 KV held fixed (GPU 4)

Both fresh services run sequentially on GPU 4. This pair isolates the complete FP32 ordered versus BF16 stock configuration at the same KV dtype, matching the October 12 SSM study. It uses a different GPU from the main matrix.

| Concurrency | FP32 SSM / FP8 KV | BF16 stock SSM / FP8 KV | BF16 vs FP32 | October 12 gain |
|---:|---:|---:|---:|---:|
| c4 | 954.83 | 977.42 | +2.37% | +2.21% |
| c16 | 2704.38 | 2860.95 | +5.79% | +5.00% |
| c24 | 3419.20 | 3676.16 | +7.52% | +8.15% |
| c32 | 3868.28 | 4208.84 | +8.80% | +9.15% |
| c48 | 4361.85 | 4800.04 | +10.05% | +10.42% |
| c64 | 4824.72 | 5318.65 | +10.24% | +10.61% |

### 4B historical configuration alignment

The October 13 previous-ordered run used **BF16 SSM / FP8 E4M3 KV**, confirmed by its archived launcher argv. It was not a BF16 KV measurement. The current BF16 KV row is a separate configuration required by the unquantized-KV matrix. Only the FP8 KV columns below use matching SSM/KV/backend settings; native revision and GPU still differ between dates.

| Concurrency | Historical ordered / FP8 KV | Current ordered / FP8 KV | Change at matching dtype | Current ordered / BF16 KV |
|---:|---:|---:|---:|---:|
| c4 | 936.45 | 978.79 | +4.52% | 953.88 |
| c16 | 2741.93 | 2862.04 | +4.38% | 2619.71 |
| c24 | 3573.93 | 3630.11 | +1.57% | 3260.84 |
| c32 | 4085.16 | 4153.06 | +1.66% | 3665.11 |
| c48 | 4705.02 | 4770.53 | +1.39% | 4175.26 |
| c64 | 5275.68 | 5367.96 | +1.75% | 4546.23 |

To select BF16 SSM and BF16 KV explicitly, use `--ssm-dtype bfloat16 --kv-cache-dtype bfloat16`. To reproduce the historical dtype/backend selection, use `--ssm-dtype bfloat16 --kv-cache-dtype fp8_e4m3 --ordered-bf16-ssm`. Grouped BF16 GDN and GDN PDL remain disabled in this previous-ordered comparison.

At c64, 2B ordered BF16 reaches **10463.62 tokens/s**, versus **10182.11** for current stock BF16 and **10472.36** for the October 13 ordered run (-0.08% versus that historical run). The default backend now selects this ordered configuration; use `--no-ordered-bf16-ssm` to reproduce stock GDN.

The October 13 GDN study used stock/ordered/grouped BF16 routes, BF16 KV for 2B and FP8 KV for 4B, with an earlier native binary (`c4522aff…`). Its runtime source hashes match the current runtime; native binary, GPU and route differences prevent attributing changes to SSM precision alone. Ordered BF16 retains the unresolved correctness limitations described in the [GDN guide](mxfp8-gdn.md).

## Workload and verification

| Concurrency | Scored requests | Input tokens/request | Output tokens/request |
|---:|---:|---:|---:|
| c4 | 40 | 3000 | 1000 |
| c16 | 160 | 3000 | 1000 |
| c24 | 120 | 3000 | 1000 |
| c32 | 160 | 3000 | 1000 |
| c48 | 240 | 3000 | 1000 |
| c64 | 320 | 3000 | 1000 |

Each configuration completes 1,040 scored requests. Request concurrency is a client limit; physical decode rows can differ because of graph padding and scheduler tails. Startup, JIT and both warmup phases are excluded.

Each point performs independent prewarm (`c` requests with 256 output tokens), internal warmup (`c` requests with 128 output tokens), then the scored requests. Uniform prompts use token IDs in `[1000, 240000)`, contract seed `20260924`, and scored request seed base `2026092400`; independent prewarm uses `20261003` and `2026100300`. Requests are greedy and streamed, with `ignore_eos=true`, `top_k=-1`, `top_p=1`, and prefix caching disabled.

Output throughput uses successful completion-token usage divided by the scored duration of that concurrency point. TTFT starts after semaphore admission; TPOT is `(response end - first token time) / (completion_tokens - 1)`. Streaming-event intervals are not independently reconstructed model-token timestamps.

Every current configuration passes exact request/token accounting and worker receipt checks for the native library, SSM dtype and actual KV storage. 4B disables the optional `VLLM_SMALL_BATCH_ATTN` experimental branch and uses standard FlashInfer attention in both KV modes. Both models use a 19 GiB KV budget and FP32 or BF16 SSM as stated in each table.

## Reproduction

Build [Dockerfile.mxfp8](../deploy/Dockerfile.mxfp8), prepare the [4B](mxfp8-model.md) or [2B](mxfp8-2b-model.md) checkpoint, and start the [4B](mxfp8-champion.md) or [2B](mxfp8-2b-champion.md) launcher with the flags shown under the relevant table. 2B retains `--attention-backend TRITON_ATTN --hybrid-head`; 4B enables the hybrid head by default.

Against an already-ready server, use a new output directory:

```bash
python -m vllm_mach.mxfp8.benchmark \
  --base-url http://127.0.0.1:8000 --model q35-mx8-study \
  --outdir ./sixpoint-results
```

Use `q35-2b-study` for 2B. The client sends requests and does not manage the server. `--points 4 16` runs a subset; only the complete six-point suite receives `complete=true`. Failed or incomplete requests produce exit code 2 and preserve the raw evidence.

## Historical fidelity

Earlier fixed-gold, SSM and KV precision records are retained in `historical_validation` in the same JSON. They identify earlier native revisions and are not new numerical qualification of this throughput matrix. The [KV calibration notes](mxfp8-kv.md) and profile guides describe their scope.

## Regeneration

```bash
python tools/render_mxfp8_benchmark.py
python tools/render_mxfp8_benchmark.py --check
```
