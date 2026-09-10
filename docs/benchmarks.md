# Benchmarks

Qwen3.8-27B on two RTX 5090 GPUs, TP2. The comparison includes FP8 on accelerated vLLM 0.28 and official vLLM 0.29, the native MXFP6 Champion, K5/K6 and K4/K5-derived W6 source stacks, and locally calibrated NVFP4.

## Throughput

The [README figure](../README.md#3k1k-reference-comparison) reports generated tokens per second for fixed 3000-input/1000-output requests. Each configuration runs one full c32 conditioning round, followed by scored c32, c4, c16 and c24 points. Every invocation also includes four unscored 32-output-token warmups.

Output tokens/s:

| Configuration | c4 | c16 | c24 | c32 |
| --- | ---: | ---: | ---: | ---: |
| FP8 · accelerated vLLM 0.28 | 277.84 | 837.84 | 1058.08 | 1205.25 |
| FP8 · official vLLM 0.29 | 268.68 | 804.27 | 1015.63 | 1157.38 |
| MXFP6 Champion | 367.11 | 979.90 | 1142.97 | 1410.72 |
| K5/K6 Hybrid + FP16 SSM + BA | 386.01 | 1066.85 | 1329.31 | 1564.36 |
| K4/K5-derived W6 + FP16 SSM | 379.46 | 1027.02 | 1283.61 | 1537.13 |
| NVFP4 · local calibration | 396.24 | 1199.32 | 1497.03 | 1674.97 |

| Concurrency | Scored requests | Contract seed | Request seed base |
| ---: | ---: | ---: | ---: |
| 4 | 192 | 20260904 | 2026090400 |
| 16 | 512 | 20260916 | 2026091600 |
| 24 | 672 | 20260924 | 2026092400 |
| 32 | 768 | 20260932 | 2026093200 |

Requests use fixed-seed token IDs, greedy sampling and `ignore_eos=true`. Shared serving settings are TP2, BF16 attention KV, maximum model length 8192, 32 sequences, 4096 batched tokens, chunked prefill, no prefix caching or speculative decoding, and graph capture sizes `[1,2,4,8,16,24,32]`.

Mean TPOT, in milliseconds:

| Configuration | c4 | c16 | c24 | c32 |
| --- | ---: | ---: | ---: | ---: |
| FP8 · accelerated vLLM 0.28 | 13.77 | 18.11 | 21.59 | 25.30 |
| FP8 · official vLLM 0.29 | 14.22 | 18.86 | 22.48 | 26.35 |
| MXFP6 Champion | 10.34 | 15.56 | 20.13 | 21.71 |
| K5/K6 Hybrid + FP16 SSM + BA | 9.84 | 14.27 | 17.25 | 19.54 |
| K4/K5-derived W6 + FP16 SSM | 10.04 | 14.83 | 17.85 | 19.87 |
| NVFP4 · local calibration | 9.72 | 12.76 | 15.38 | 18.36 |

FP8, NVFP4 and native MXFP6 use an 8,220,835,840-byte KV budget and FP32 recurrent state. The two FP16-state configurations use 8,218,214,400 bytes. FP8/NVFP4 use 320 blocks with 784-token pages; FP16-state runs use 627 blocks with 400-token pages. The older native MXFP6 runtime reports a 187,245-token KV capacity with the same byte budget.

All 12,864 scored requests completed with the expected token counts. Every point ran for at least 300 seconds, with zero preemptions.

### Runtime configuration

| Configuration | Runtime and execution path |
| --- | --- |
| FP8 · accelerated vLLM 0.28 | Existing patched vLLM/FlashInfer image, native FP8 loader, TRT-LLM collectives |
| FP8 · official vLLM 0.29 | Unmodified `vllm/vllm-openai:v0.29.0`, same FP8 checkpoint, native backend defaults |
| MXFP6 Champion | Original vLLM 0.25.1 / PyTorch 2.11 image, native W6A8, FLASHINFER attention, persistent SM120 GDN, TRT-LLM AllReduce/RMSNorm fusion |
| K5/K6 Hybrid + FP16 SSM + BA | vLLM 0.28 / ExLlamaV3 1.4.6 source stack, FP16 SSM and M16/M24 BA overlap |
| K4/K5-derived W6 + FP16 SSM | vLLM 0.28 source stack, K4/K5 MLP weights decoded and requantized into a W6 execution cache |
| NVFP4 · local calibration | Accelerated vLLM 0.28 image; llm-compressor checkpoint calibrated with 20 ShareGPT4V samples |

FP8 and NVFP4 use TRITON_ATTN and FULL_AND_PIECEWISE compilation. Native MXFP6 also uses FULL_AND_PIECEWISE, with `fi_allreduce_fusion_max_size_mb=64`. The EXL3 source stacks use FULL_DECODE_ONLY. Both FP8 images report PyTorch 2.13.0+cu130 and FlashInfer 0.6.18; the accelerated image includes source patches.

K4/K5 and K5/K6 identify the source checkpoints. K4/K5-derived W6 requires about 6.23 GiB of additional W6 cache per GPU and uses the earlier BA runtime without the M16/M24 overlap switch. Its throughput is 1.7–3.7% below the K5/K6 configuration across these four points; this comparison does not isolate the cause.

These are single-lifecycle, cross-runtime measurements taken September 8–10, 2026. GPU/host monitoring was retained; clocks were not locked. The EXL3 curves use the optimization source stack. Mach release measurements are listed separately below.

### Reproduce and verify

The [public data](data/quantization-comparison-3k1k-20260910.json) includes request timings, token counts, contracts, checkpoint metadata, image IDs and source hashes. Throughput is total output tokens divided by request-set duration. Per-request TPOT is `(latency - TTFT) / (output tokens - 1)`; the table averages it across requests. p99 uses linear interpolation. Client-side admission waiting is included in throughput, but excluded from TTFT and TPOT.

Use the [benchmark client](data/benchmark_fixed_token_contract.py) with the selected server. Run this c32 conditioning round first:

```bash
python -m pip install aiohttp
python docs/data/benchmark_fixed_token_contract.py \
  --base-url http://127.0.0.1:8000 --model Qwen3.8-27B \
  --input-tokens 3000 --output-tokens 1000 \
  --max-concurrency 32 --num-prompts 768 \
  --contract-seed 20260932 --request-seed-base 2026093200 \
  --json-out conditioning_c32.json
```

Discard that result. Repeat with a new output file for scored c32, then run c4, c16 and c24 using the counts and seeds above. The client adds the four short warmups automatically.

## Numerical fidelity

The accuracy comparison measures gold-token logprob MAE against the same BF16 reference. For each query, average the absolute logprob difference over its target tokens; then average the 256 query results with equal weight. Lower is better. This measures numerical fidelity, not a task accuracy percentage.

| Configuration | MAE | 95% query-bootstrap interval |
| --- | ---: | ---: |
| FP8 · official vLLM 0.29 | 0.051988 | 0.046530–0.057672 |
| FP8 · accelerated vLLM 0.28 | 0.052131 | 0.046602–0.057911 |
| MXFP6 Champion | 0.089138 | 0.080897–0.097742 |
| K5/K6 Hybrid + FP16 SSM + BA | 0.091515 | 0.082482–0.100855 |
| K4/K5-derived W6 + FP16 SSM | 0.091434 | 0.083507–0.099575 |
| NVFP4 · local calibration | 0.169987 | 0.153031–0.187708 |

All configurations use the same 256 queries and 10,479 target tokens, with 64 queries each from GSM8K, HumanEval, HellaSwag and CMMLU Chinese history. The decoder is teacher-forced through eight physical-M32 cohorts. Cohort zero is repeated as a stability check. The hook changes token selection only; it reads each runtime's raw gold-token logprobs. Short diagnostic admission limits accommodate all 32 requests together.

The diagnostic uses maximum model length 768 and a prefill budget of 8192 tokens. FP8, NVFP4 and the EXL3 source stacks use FULL_DECODE_ONLY; native MXFP6 retains its FULL_AND_PIECEWISE compilation. The BF16 reference uses eager execution. Per-run settings and repeat-stability results are included in the data extract.

K4/K5-derived W6 and K5/K6 measure MAE 0.091434 and 0.091515. Their paired mean difference is -0.000081, with a 95% query-bootstrap interval of [-0.004722, 0.004390]. These data show no clear fidelity advantage for either configuration.

The [accuracy data](data/accuracy-comparison-m32-20260910.json) contains per-query gold-token logprobs, the BF16 reference, dataset revisions and source hashes. Chart intervals use 20,000 query-bootstrap samples with seed 20260910.

### Fidelity and throughput plot

The combined plot uses M32 MAE on the horizontal axis and throughput gain over official vLLM 0.29 FP8 on the vertical axis. For each configuration and concurrency, gain is `100 * (throughput / FP8 throughput - 1)`. Each point shows the arithmetic mean of the four gains at c4/c16/c24/c32, with equal weight. The vertical range is their minimum and maximum, not a confidence interval. The horizontal range is the same 95% MAE interval shown in the fidelity chart.

Check the tables and regenerate all three figures:

```bash
python docs/data/check_quantization_comparison.py
python docs/data/check_accuracy_comparison.py
python -m pip install matplotlib numpy
python docs/data/plot_comparison.py
```

## Mach a8 acceptance

Measured September 9, 2026 with the a8 runtime changes, before finalizing the wheel version metadata. This shorter acceptance workload had four short warmups per point and no full-length conditioning round.

| Input/output tokens | Concurrency | Requests | Elapsed, s | Output tokens/s | Mean TPOT, ms | p99 TTFT, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 / 256 | 32 | 256 | 45.37 | 1444.52 | 18.98 | 2829.26 |
| 3000 / 1000 | 16 | 128 | 124.23 | 1030.31 | 14.61 | 3429.04 |
| 3000 / 1000 | 24 | 128 | 106.29 | 1204.30 | 17.38 | 5228.80 |
| 3000 / 1000 | 32 | 128 | 85.81 | 1491.60 | 19.75 | 7090.95 |

The service used vLLM 0.28, PyTorch 2.13.0+cu130, ExLlamaV3 1.4.8 with BF16 I/O, patched FlashInfer 0.6.18, B12X 1.3.0 and mxfp6-sm120 0.2.1. The [long-prefill profile](../profiles/vllm-0.28.0/qwen38-checkpoint-long.env) enabled Temporal M24, BA32 and lossless prefill collectives. It used FP32 recurrent state, TRITON_ATTN, FULL_DECODE_ONLY and automatic KV allocation of 324,169 tokens. All 640 requests completed in one lifecycle.

Raw timing extracts are in [serving-results-20260909.json](data/serving-results-20260909.json); verify them with `python docs/data/check_serving_results.py`. See [long-prefill validation](long-prefill.md), [FP16 SSM integration](fp16-ssm.md) and [earlier Champion alignment](champion-alignment.md) for the corresponding implementation checks.
