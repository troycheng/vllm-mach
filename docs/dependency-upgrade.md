# vLLM 0.29 / ExLlamaV3 1.4.9 upgrade

The development profile carries a9's checkpoint-hybrid optimizations onto vLLM 0.29.0 and patched ExLlamaV3 1.4.9. The short TP2 regression retained throughput; the dependency upgrade does not establish a new speedup.

## Dependencies and compatibility

PyTorch stays at 2.13.0, FlashInfer at 0.6.18, B12X at 1.3.0 and MXFP6 SM120 at 0.2.1. ExLlamaV3's BF16 extension, M32 extension and Temporal extension were rebuilt against the 1.4.9 source. The unchanged lossless-prefill extension was reused with the same Torch ABI.

The port includes three startup fixes:

- vLLM 0.29 caches the plain-weight fused embedding decision during construction. Mach clears it when replacing the embedding method with online EXL3 quantization.
- MXFP6 workspace planning runs before memory profiling, which can capture graphs before normal kernel warmup. Capture-stream registration is retained.
- FlashInfer's local TRT-LLM IPC allocation does not request RDMA-capable memory. This patch was present in the previous validated image and is now included in the source installation profile. Other `SymmDeviceMemory` callers keep their original default.

The GDN port preserves 0.29's CPU handling, prefill length conversion, sigmoid output gate, speculative head-ratio support and missing-metadata handling. Mach retains its guarded FP16 recurrent state and M16/M24/M32 BA overlap. QK norm/MRoPE and explicit capture-stream changes already present upstream are not reapplied.

Use the [vLLM 0.29 installation profile](../profiles/vllm-0.29.0/README.md) and [ExLlamaV3 1.4.9 BF16 patch](../profiles/exllamav3-1.4.9/README.md). Upgrading the Python wheel alone does not install runtime patches.

## Short regression

Measured on September 13, 2026: Qwen3.8-27B K5/K6 checkpoint-hybrid profile, two RTX 5090 GPUs on one PIX pair, TP2, BF16 activations/KV and opt-in FP16 recurrent state. Both versions ran sequentially on the same devices with identical pre-tokenized requests and sampling settings: 3000 input / 1000 output tokens, context limit 8192, maximum batch 32, chunked prefill 4096, no prefix cache, and FULL_DECODE_ONLY graph sizes 1/2/4/8/16/24/32. KV allocation was fixed at 8,218,214,400 bytes.

Each version ran two samples per concurrency: 16 requests at c4, 32 at c16, 48 at c24 and 64 at c32. Each sample followed a concurrency-matched 128-output-token warmup. Values below are means of the two samples; throughput counts generated tokens only. Request contracts, input hashes and per-request 3000/1000 token counts matched. The c24/c32 addendum reused the accepted runtime snapshots and the same physical GPU pair.

| Concurrency | a9 / vLLM 0.28 tok/s | Development / vLLM 0.29 tok/s | Change | a9 TPOT ms | Development TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 350.034 | 350.039 | +0.001% | 10.935 | 10.946 |
| 16 | 1073.050 | 1072.525 | −0.049% | 13.377 | 13.419 |
| 24 | 1348.395 | 1346.660 | −0.129% | 15.827 | 15.841 |
| 32 | 1577.483 | 1575.263 | −0.141% | 17.797 | 17.818 |

Average TPOT changed by +0.10% at c4, +0.31% at c16, +0.09% at c24 and +0.12% at c32. All 640 measured requests across both versions completed successfully. This short check found no material throughput regression. It does not replace the longer reference benchmark curves or validate automatic KV sizing.

Runtime logs confirmed both Temporal M24 bundle shapes, fused SiLU/MXFP8, fused AR/RMSNorm/MXFP8, direct lossless prefill, long prefill and BA overlap during Graph capture.

## Correctness

- CPU tests: 206 passed, 3 skipped.
- Native BF16 tests: 11 passed. Four upstream sliced-MGEMM cases passed changing-input Graph and buffer-canary checks.
- Real-checkpoint eager/Graph checks: 52 cases across both TP ranks passed. These compare each selected path with its reference; the separate optional M32 tile kernel was compiled but not exercised in this run.
- Development service: 40/40 task checks passed; serial versus BA-overlapped execution passed 2,592 state/output byte comparisons at M16/M24/M32.

The old service's automated task score was 39/40 on this pair. The remaining answer used a valid `dict.fromkeys()` implementation that the evaluator's method allowlist rejected. Its exact saved answer passed all three original test cases on inspection and execution. The original score and separate adjudication were retained; no answer was regenerated. This is an evaluator limitation, not an accuracy gain attributed to the upgrade.
