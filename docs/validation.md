# Validation

## 0.1.0a5.dev0

The direct-checkpoint profile passed the existing 95 package tests (3 skipped) and a full CPU comparison with the frozen experimental loader: 256 rank-local projections per TP rank, identical packed weights and logical scales, and matching aggregate content digests. This establishes the port's loading contract for the tested checkpoint, not a public checkpoint download identity.

ExLlamaV3 1.4.8 with the PR #330 BF16 patch was built in the official vLLM 0.28.0 image using Python 3.12, PyTorch 2.13.0+cu130, and the CUDA 13.2 development toolkit. The unmodified image lacked `cusparse.h`. Importing the built extension after PyTorch did not initialize CUDA. Its wheel SHA-256 is `90606feea9a28f5423b712bcec3468ea1dc31c18229744dbfb6019e7819f9586`.

The real-weight native check passed 52 cases on two RTX 5090 GPUs, with three changing inputs per case. EXL3 QKV/QKVZ covered M=1/16/24/32 with CPU Hadamard group IDs; the six MXFP6 projection families covered M=24/32/128. Eager and CUDA Graph outputs were bitwise identical to the respective reference call paths. The separate EXL3 M32 extension was disabled; the checkpoint profile uses MXFP6 at QKV M32.

The first TP2 service check used the original development wheel, the patched 1.4.8 dependency, and `mxfp6-sm120==0.2.1`. All seven FULL_DECODE_ONLY graph sizes `[1, 2, 4, 8, 16, 24, 32]` captured. The existing 40-task suite at concurrency 32 passed 40/40 with no pass/fail regressions; 35/40 outputs matched the stored reference exactly. FlashInfer 0.6.16.post3 could not create its AllReduce workspace on the selected pair and used the fallback, so this run does not validate the fused collective path.

A second configuration installed FlashInfer Python/cubin/JIT-cache 0.6.18. Its patched header had SHA-256 `049e8b8c0b9f866d1a49247399a17de0809f3779521753debcd648b7888b1a4e`, which the original guard rejected. A local experiment allowing that header activated the fused path and captured all seven Graph sizes, but passed 0/40 tasks, with repetitive malformed outputs.

The failure was isolated to the prebuilt communication module in `flashinfer-jit-cache`: FlashInfer loaded that binary instead of compiling the patched header. In the existing TP2 M=4 boundary check, both ranks' activation values and residuals matched the reference, but active scales and the following MXFP6 gate/up output differed. Removing only that cache package and compiling the same patched source restored bitwise equality for all four outputs on both ranks, with identical reference digests. Mach now verifies the selected build path as well as the header and rejects a prebuilt communication module. The package tests report 97 passed and 3 skipped. These observations do not indicate a PDL arithmetic regression.

The corrected wheel then passed the same TP2 service suite with the source-built FlashInfer 0.6.18 module: 40/40 tasks, zero regressions, and 38/40 exact reference outputs. Both ranks selected fused AllReduce/GemmaRMSNorm/MXFP8, and all seven Graph sizes captured. The wheel SHA-256 is `74ef82fca512002d8255df49e948fa8a220f4b31d7a34b57a1da385dd1e2658b`; subsequent documentation edits do not change its runtime source. The service used the same patched ExLlamaV3 1.4.8 and direct-checkpoint profile as the failed configuration.

These are functional checks, not full-model bitwise equivalence, business accuracy, or throughput measurements. Temporal M24 and private FLA arithmetic-layout patches were not installed. The experimental recipe's fidelity and performance results must not be attributed to this Mach build.

## 0.1.0a4

On 2026-09-05, the compatibility candidate passed 68 package tests with 3 skipped. ExLlamaV3 was rebuilt from public commit `d0094bc922bcf2d6cf5e948ba35f347adda3a6ca`; the independent M32 extension was built from the Mach `v0.1.0a3` source. Both used the official `vllm/vllm-openai:v0.28.0` image, Python 3.12, PyTorch 2.13.0+cu130, CUDA toolkit 13.2, and target `12.0a`. Both extensions imported after PyTorch without initializing CUDA.

The fixed Mach wheel then served Qwen3.8-27B K5/K6 on two RTX 5090 GPUs with TP2. QKV MGEMM, BF16 I/O, M24/M32, the native M32 module, and the vLLM 0.28 sampling-metadata patch were enabled. B12X and the MXFP6 hybrid profile were not installed/enabled. Native prefill used no persistent reconstructed-weight cache and a 512 MiB reconstruction limit. The service used an 8,192-token model limit, 4,096 batched tokens, and at most 32 sequences.

All seven FULL_DECODE_ONLY graph sizes `[1, 2, 4, 8, 16, 24, 32]` captured successfully, and both workers loaded the M32 module. The existing 40-task suite ran at concurrency 32: 40/40 passed, zero pass/fail regressions, and 39/40 outputs matched the stored reference exactly. This is a functional compatibility check, not full-model bitwise equivalence or a throughput benchmark. Legacy GPU-metadata routing has unit coverage; the older experimental wheel was not rerun on GPUs in this check. The separate public MXFP6/FlashInfer dependency build remains outside this check.

Native wheel SHA-256 values:

- ExLlamaV3: `cc56e5cb4cb43c1b3cda5818ab25b723bca626a9c6cd1c8c8952e4cb181c210f`
- M32 extension: `150abfaf996d1ec4bcc19a222fceb924a976c17e0396717cc027c3794c5530e5`

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
