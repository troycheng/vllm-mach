# Rank64 and owner-prefill integration

The September 14 source profile integrates `rank64-mlp32-ragged-20260914`: selected NVFP4 gate/up projections with rank64 compensation at physical M32, and owner-local prefill with MLP replication at 32 even-numbered layers. The separate M32 P1 candidate is not included. Both additions are opt-in through the [complete launcher](installation.md#start-the-complete-profile).

## Clean installation

Validation used a fresh image built from official vLLM 0.29.0, without mounting a previous Python environment, native libraries or source adapters. Dependencies were PyTorch 2.13.0+cu130, patched ExLlamaV3 1.5.0, FlashInfer 0.6.18, B12X 1.3.0 and MXFP6 SM120 0.2.1. MXFP6 was rebuilt against the image's PyTorch ABI. The tested Mach package was `0.1.0a11.dev0`.

The build and launcher now cover the native extensions, matching runtime patches, model paths and full serving configuration. Matching MXFP6 and rank64 assets have been packaged and checked locally. They are not distributed by Mach. The [generation commands](quantization.md) assemble the existing quantizers and capture method; that assembled workflow has not been rerun end to end.

## Correctness

| Check | Result |
| --- | --- |
| Initial full package suite | 270 passed; 3 GPU-only cases skipped |
| MXFP6 GPU operator cases | The 3 skipped cases passed on GPU |
| Lifecycle and owner metadata fixes | 52 focused tests passed |
| Diagnostic snapshot guards | 2 tests passed |
| Rank64, two distinct real M32 input batches | 768 comparisons passed across 48 layers and both TP ranks |
| Owner-prefill, two real M3000 forwards per rank | 2,868 bitwise comparisons passed |
| Task retention against the previous validated run | 40/40 passed; no newly failed cases |

Rank64 checks compare packed values, live scales, eager results and changing-input CUDA Graph results against the original quantization and residual-merge operations. Snapshots were taken while all 32 requests were decoding. Owner checks cover intermediate residuals, normalization, quantized activations and MLP outputs. M4096 ran in the serving workload but was not shadow-compared in this acceptance run. These checks establish port correctness; they are not a new whole-model MAE measurement.

## Short serving regression

Qwen3.8-27B, two RTX 5090 GPUs, TP2, 3000 input / 1000 output tokens. The launcher retained 8,218,214,400 attention-KV bytes per GPU (627 blocks), BF16 activations/KV, FP16 recurrent state, context 8192 and prefill chunks of 4096. MLP replicas add 3,342,336,000 weight bytes per GPU.

Each row used a warmup of C requests with 128 output tokens, followed by the request count below. Contract seed was `20260910`; request seed base was `2026091000`.

| Concurrency | Requests | Output tok/s | Mean TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 4 | 16 | 351.90 | 10.90 |
| 16 | 32 | 1,096.30 | 13.25 |
| 24 | 48 | 1,372.14 | 15.68 |
| 32 | 64 | 1,646.62 | 17.17 |

These are short installation regressions, not replacements for the longer README reference curves or a paired speedup measurement against the source Champion. Use the [installation check command](installation.md#check-and-measure) to repeat the C32 row.

The tested image ID was `sha256:097f00a9353feb348d58a86d0ef30b418636129915d803bca894d617d549a7d8`. Runtime build identities are recorded in `/opt/mach-build/installed.json` and `/opt/mach-build/mxfp6-source.json`.
