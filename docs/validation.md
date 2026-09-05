# Validation

## 0.1.0a3

The M24/M32 and sampling-metadata integration passed 60 package tests with 3 skipped, a native source build against public ExLlamaV3 headers, and changing-input eager/Graph checks on both TP ranks. A complete EXL3/MXFP6 service with the fused FlashInfer collective then captured graph sizes `[1, 2, 4, 8, 16, 24, 32]`, returned a completion, and passed the existing 40-task retention suite with no pass/fail regressions. Exact output agreement with the stored reference was 33/40. The tested Mach runtime source is commit `dd48f2a`; the subsequent candidate packaging changes version metadata and documentation only.

These checks do not establish a new end-to-end speedup or full-model bitwise equivalence. The new switches remain opt-in. See [experimental decode paths](experimental-decode.md) for source identities, build instructions, and numerical limits.

## 0.1.0a2

Version 0.1.0a2 is based on the provider slice validated on 2026-09-03 against `vllm/vllm-openai:v0.28.0`, ExLlamaV3 1.4.6 with the BF16-I/O extension, and B12X 1.3.0. The Dense MXFP6 bridge and fused paths received separate GPU gates on 2026-09-04.

The clean-container checks covered import without CUDA initialization, idempotent EXL3 and MXFP6 registration, the native MXFP6 W6A8 selector contract, fail-closed behavior when the optional wheel is absent, valid and invalid checkpoint metadata, the bundled Hadamard fold, the serialized fallback, wheel contents, and installation without the development sources.

The EXL3 provider gate used Qwen3.8-27B K5/K6 with TP2. It completed model loading, prefill profiling, FULL_DECODE_ONLY graph capture for `[1, 2, 4, 8, 16, 24, 32]`, health and model-list requests, and single- and four-prompt completions. No worker or CUDA error was observed after the requests.

This validation establishes the initial compatibility boundary. It is not a general support claim for other models, tensor-parallel layouts, vLLM versions, or GPU architectures.

The native Dense call was checked against the public `gemm_from_float` reference using a runtime-matched `mxfp6-sm120==0.2.1` build on SM120. The published prebuilt wheel did not load in the tested vLLM image because its PyTorch ABI differed, so the current installation contract requires building the wheel in the target runtime.

The serving gate used the public Qwen3.5-27B-MXFP6 checkpoint, TP2, two SM120 GPUs, a 12,288-token model limit, and graph sizes `[1, 2, 4, 8, 16, 24, 32]`. vLLM selected `Mxfp6Sm120LinearKernel`, completed Stream-K workspace planning, and captured all seven PIECEWISE and all seven FULL decode graphs. vLLM changed the requested `FULL` mode to `FULL_AND_PIECEWISE` because its GDN backend supports only uniform-batch full graphs. Health, model-list, and chat-completion requests returned HTTP 200; a non-thinking request returned `OK`. No worker or CUDA error was observed. The first inference compiled two unrelated vLLM Triton support kernels; the second emitted no new JIT warning.

The selective EXL3/MXFP6 profile received a separate TP2 gate on Qwen3.8-27B K5/K6. Eager mode loaded 14.22 GiB per rank and completed a 1,852-token prompt plus 32 generated tokens. FULL_DECODE_ONLY mode planned the shared MXFP6 workspace, captured all seven configured graph sizes in 6 seconds, and used 0.13 GiB per rank for CUDA Graphs. Four concurrent requests with 1,702 to 1,972 prompt tokens and a subsequent changed-input request with 2,372 prompt tokens all returned HTTP 200. No plugin, worker, CUDA, or graph replay error was observed.

The fused Dense MLP boundary was checked on SM120 against the public `mxfp6-sm120==0.2.1` reference operations. The fused SiLU/MXFP8 values and logical scales matched the reference exactly, and the following MXFP8-by-MXFP6 down projection produced bitwise-identical BF16 output.

The source implementation for FlashInfer AR/GemmaRMSNorm/MXFP8 was checked with changing inputs at rows 1, 2, 4, 8, 16, 24, and 32 on both TP ranks. Packed values, active scales, residual output, and the following MXFP6 gate/up output were bitwise identical to the two-operation reference. A 40-case task-retention run passed 40/40 with no pass/fail changes. The original A-B-B-A serving observation measured `+1.289%` output throughput at c4 and `+0.781%` at c16, but lacked a continuous external resource monitor and is therefore retained as integration evidence rather than advertised as a release benchmark.

The `0.1.0a2` wheel then passed a fresh TP2 service smoke on the same Qwen3.8-27B profile with FlashInfer 0.6.18. Both ranks selected the fused AR/GemmaRMSNorm/MXFP8 path during FULL_DECODE_ONLY graph capture, all seven configured row counts were captured, the server became healthy, and a completion request returned successfully.

This is a functional gate, not a performance comparison. End-to-end A/B measurements at the documented request shapes and broader correctness coverage remain release work, and no MXFP6 throughput number is claimed by this preview.
