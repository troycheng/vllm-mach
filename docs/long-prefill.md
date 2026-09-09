# Observed-shape prefill collective

Integration of `long_prefill_direct_sum_service_v1` for the existing Qwen3.8-27B EXL3 K5/K6 and checkpoint MXFP6 profile. Weight loading, Temporal M24, BA32, sampling and the separate M4096 direct SUM implementation are unchanged. The rejected BA24 and attention-compaction experiments are not included.

The new extension covers 32 observed prefill row counts at H5120, TP2 and BF16. Source arithmetic, signed-zero handling, odd-row ownership, barriers and PDL completion are preserved. The Mach adapter provides explicit opt-in, version/device/workspace checks and an eager-only dispatcher. Unsupported shapes retain the existing route. [Build and configuration](../native/lossless_prefill/README.md#observed-prefill-shapes).

## Source evidence

The source experiment expanded an earlier 11-shape implementation to 32 shapes. On fixed 3000-input/1000-output requests, long c16/c24 geometric throughput gains were +1.0675% in the full eight-point comparison and +1.0023% in a separate reverse-order confirmation. These compare two source-stack versions; they are not gains over released Mach or the original MXFP6 baseline. Small changes at other concurrency points are not significance claims. Do not pool the rounds or add earlier optimization gains.

Source real-model checks covered 32 shapes × 128 norm instances × two ranks with bitwise residual/norm agreement against installed FlashInfer. Source sanitizers passed. Text differences also occurred between runs of the same implementation; the existing batch-dependent EXL3/MXFP6 routing remains. These collective checks do not establish full-model bitwise equivalence or business accuracy. Other models, MoE and random sampling were not validated by this source experiment.

## Mach integration checks

The native and Python wheels built successfully; 133 package tests passed. Installed FlashInfer and the Mach extension matched bitwise for six synthetic cases at each of the 32 shapes on each rank (384 cases total). All 64 changing-input Graph checks and the shared-workspace/PDL sequence passed. The kernel and codec headers differ from the selected source only in namespace/include names and attribution comments.

The diagnostic Mach service checked 32 shapes × 128 distinct norm instances × two ranks (8192 full residual/norm boundaries), all bitwise equal to installed FlashInfer. M4096's separate direct path also verified 128 instances per rank. BA32 Graph replay included integer-view equality assertions against serial QKV, BA and both split outputs at all 48 layers per rank; changed real requests completed without assertion failures. This checks the changed producer boundary, not a separate full-state dump of the entire model. The 40-task suite passed with zero pass/fail regressions and 36/40 exact stored-reference text matches; all 128 short c32 requests completed.

Diagnostic verification adds reference operations and synchronization; its throughput cannot be used as a release performance result. The verifier-off combined service is checked separately.

The separate verifier-off service passed 40/40 tasks with zero pass/fail regressions and 38/40 exact stored-reference text matches. BA32 capture markers covered all 48 layers per rank; no diagnostic verification markers were present. All requests completed with the contracted token counts:

| Input/output tokens | Concurrency | Requests | Output tokens/s |
| --- | ---: | ---: | ---: |
| 1024/256 | 32 | 256 | 1444.52 |
| 3000/1000 | 16 | 128 | 1030.31 |
| 3000/1000 | 24 | 128 | 1204.30 |
| 3000/1000 | 32 | 128 | 1491.60 |

These are single-lifecycle integration observations with four short warmup requests per point, not a paired performance comparison or the source experiment's full conditioning protocol. The normal service automatically allocated 324,169 KV tokens; the source experiment pinned 335,286. Long c32 uses N128 here versus N256 in the source comparison. Do not compare these numbers as an isolated speedup or regression. Both test workers were stopped and the test containers removed; GPUs 6/7 returned to idle.

Use [qwen38-checkpoint-long.env](../profiles/vllm-0.28.0/qwen38-checkpoint-long.env) for the accepted v0.1.0a8 combination. It enables BA32, M4096 direct SUM and the 32-shape prefill extension, with diagnostics disabled. Available in v0.1.0a8.
