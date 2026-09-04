# Validation

Version 0.1.0a1 is based on the provider slice validated on 2026-09-03 against `vllm/vllm-openai:v0.28.0`, ExLlamaV3 1.4.6 with the BF16-I/O extension, and B12X 1.3.0. The Dense MXFP6 bridge received a separate GPU serving gate on 2026-09-04.

The clean-container checks covered import without CUDA initialization, idempotent EXL3 and MXFP6 registration, the native MXFP6 W6A8 selector contract, fail-closed behavior when the optional wheel is absent, valid and invalid checkpoint metadata, the bundled Hadamard fold, the serialized fallback, wheel contents, and installation without the development sources.

The physical gate used Qwen3.8-27B K5/K6 with TP2. It completed model loading, prefill profiling, FULL_DECODE_ONLY graph capture for `[1, 2, 4, 8, 16, 24, 32]`, health and model-list requests, and single- and four-prompt completions. No worker or CUDA error was observed after the requests.

This validation establishes the initial compatibility boundary. It is not a general support claim for other models, tensor-parallel layouts, vLLM versions, or GPU architectures.

The native Dense call was checked against the public `gemm_from_float` reference using a runtime-matched `mxfp6-sm120==0.2.1` build on SM120. The published prebuilt wheel did not load in the tested vLLM image because its PyTorch ABI differed, so the current installation contract requires building the wheel in the target runtime.

The serving gate used the public Qwen3.5-27B-MXFP6 checkpoint, TP2, two SM120 GPUs, a 12,288-token model limit, and graph sizes `[1, 2, 4, 8, 16, 24, 32]`. vLLM selected `Mxfp6Sm120LinearKernel`, completed Stream-K workspace planning, and captured all seven PIECEWISE and all seven FULL decode graphs. vLLM changed the requested `FULL` mode to `FULL_AND_PIECEWISE` because its GDN backend supports only uniform-batch full graphs. Health, model-list, and chat-completion requests returned HTTP 200; a non-thinking request returned `OK`. No worker or CUDA error was observed. The first inference compiled two unrelated vLLM Triton support kernels; the second emitted no new JIT warning.

This is a functional gate, not a performance comparison. End-to-end A/B measurements at the documented request shapes and broader correctness coverage remain release work, and no MXFP6 throughput number is claimed by this preview.
