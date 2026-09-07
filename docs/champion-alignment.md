# Checkpoint/Temporal profile alignment

These source-profile additions ship in the a6 tagged source tree. They are not installed by the Python wheel and were not part of a5.

## Changes

| Reference component | Mach integration |
|---|---|
| Original MXFP6 checkpoint loading, merged M32/prefill QKV | Existing a5 checkpoint profile |
| BF16 QKV, Temporal M24 | Existing a5 provider and native extension |
| Fused SiLU/MXFP8 and AR/RMSNorm/MXFP8 | Existing Mach integrations |
| FlashInfer specialized GDN | New version-locked GDN source profile; opt-in, with fallback |
| SM120 TP2 AllReduce capacity limits | New runtime alignment patch |
| Fused QK norm/MRoPE | New 0.28 backport of vLLM #52676 |
| Online K6 embedding, B12X, fused reconstruction | Existing implementations; included in the complete environment file |
| Explicit `trtllm` communication backend | Included in the environment file |
| Sampling metadata and MXFP6 Graph warmup | Existing patches; no private adapter dependency |
| Quark MXFP8 activation spelling and native MXFP6 registration | Existing Mach plugin registration |
| FLA tuning state | No source difference to port; historical rank-specific autotuning results are not hard-coded |

The reference image also contains experimental MoE and unrelated FlashInfer paths. They are not used by this Dense EXL3 configuration and are not copied into Mach.

## Integration check

Qwen3.8-27B EXL3 K5/K6 with paired original MXFP6 weights, two RTX 5090 GPUs, TP2, vLLM 0.28.0, BF16 KV, TRITON_ATTN and FULL_DECODE_ONLY graph sizes 1/2/4/8/16/24/32. Mach used ExLlamaV3 1.4.8 with the BF16 patch, FlashInfer 0.6.18 and B12X 1.3.0. Temporal, GDN and fused collectives were enabled. The reference used its frozen ExLlamaV3 1.4.6-based runtime.

Package tests: 122 passed. The existing 40-task suite passed 40/40 with zero pass/fail regressions and 37/40 exact text matches to its stored reference. Both workers logged the specialized GDN implementation; QKV/QKVZ Temporal routes and CUDA Graph capture were observed. This is a functional regression check, not broad model-quality or bitwise-equivalence validation.

The short benchmark reused the reference's pretokenized fixed-seed workload: 1024 input tokens and 256 generated tokens per request, 8 waves per concurrency, 4 warmup requests. Each runtime completed 608 requests. Per-request prompt contracts matched; throughput uses server-reported completion tokens divided by elapsed request-set duration.

| Concurrency | Earlier a5 configuration, tok/s | Reference, tok/s | Aligned configuration, tok/s | Aligned/reference |
|---|---:|---:|---:|---:|
|4|281.42|382.17|381.42|−0.20%|
|16|793.98|981.29|967.64|−1.39%|
|24|990.32|1197.30|1199.50|+0.18%|
|32|1136.82|1379.34|1381.80|+0.18%|

One independent lifecycle per configuration, serial on the same GPU pair with resource monitoring. These are diagnostic point measurements, not confidence intervals or a statistical equivalence test. They show that the large deployment gap was recovered, not which individual change produced the recovery. Clocks were not locked. This workload does not cover 3k/1k requests.

At c16, the initial run had 0.93% higher mean TPOT and p99 TTFT increased from 1318.59 ms to 1792.27 ms. A bounded follow-up used one full 128-request c16 warmup and two measured repetitions per runtime, serial on the same GPU pair with unchanged configurations and request contracts:

| Runtime | Full warmup, tok/s | Repeat 1, tok/s | Repeat 2, tok/s | Measured p99 TTFT, ms |
|---|---:|---:|---:|---:|
| Reference | 982.80 | 980.65 | 979.21 | 1320.38 / 1325.49 |
| Aligned Mach | 972.45 | 984.09 | 982.83 | 1302.49 / 1307.26 |

The two measured throughput results averaged 979.93 tok/s for the reference and 983.46 tok/s for Mach (+0.36%, not an established speedup). Mach's first full round reproduced the higher p99 TTFT at 1769.33 ms; the extra delay on requests 12–15 did not recur in either warmed repetition. Mean TPOT was 13.784–13.787 ms for Mach and 13.804–13.820 ms for the reference. All 768 requests across these six rounds completed with the expected token counts. The warmed results do not show the earlier c16 deficit, but the cause of the first-round overhead remains untraced. This is one lifecycle per runtime, not statistical equivalence or a claim that every latency metric matches.

Use the [runtime profile instructions](../profiles/vllm-0.28.0/README.md#checkpointtemporal-serving-configuration) to reproduce the software configuration. The paired MXFP6 checkpoint still needs its own verified identity; architecture compatibility alone does not identify equivalent weights.
