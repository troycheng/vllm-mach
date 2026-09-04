# Validation

Version 0.1.0a1 is based on the provider slice validated on 2026-09-03 against `vllm/vllm-openai:v0.28.0`, ExLlamaV3 1.4.6 with the BF16-I/O extension, and B12X 1.3.0. The MXFP6 additions in this preview extend the static validation surface but do not yet carry a GPU serving result.

The clean-container checks covered import without CUDA initialization, idempotent EXL3 and MXFP6 registration, the native MXFP6 W6A8 selector contract, fail-closed behavior when the optional wheel is absent, valid and invalid checkpoint metadata, the bundled Hadamard fold, the serialized fallback, wheel contents, and installation without the development sources.

The physical gate used Qwen3.8-27B K5/K6 with TP2. It completed model loading, prefill profiling, FULL_DECODE_ONLY graph capture for `[1, 2, 4, 8, 16, 24, 32]`, health and model-list requests, and single- and four-prompt completions. No worker or CUDA error was observed after the requests.

This validation establishes the initial compatibility boundary. It is not a general support claim for other models, tensor-parallel layouts, vLLM versions, or GPU architectures.

The MXFP6 bridge still needs a clean image with `mxfp6-sm120==0.2.1` for the following release gate: model load with a public OCP-MX checkpoint, native-vs-emulation correctness, workspace planning before graph capture, TP2 serving, and an end-to-end comparison at the documented request shapes. No MXFP6 throughput number is claimed by this preview.
