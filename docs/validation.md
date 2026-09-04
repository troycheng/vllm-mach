# Validation

Version 0.1.0 is based on the provider slice validated on 2026-09-03 against `vllm/vllm-openai:v0.28.0`, ExLlamaV3 1.4.6 with the BF16-I/O extension, and B12X 1.3.0.

The clean-container checks covered import without CUDA initialization, idempotent registration, valid and invalid checkpoint metadata, the bundled Hadamard fold, the serialized fallback, wheel contents, and installation without the development sources.

The physical gate used Qwen3.8-27B K5/K6 with TP2. It completed model loading, prefill profiling, FULL_DECODE_ONLY graph capture for `[1, 2, 4, 8, 16, 24, 32]`, health and model-list requests, and single- and four-prompt completions. No worker or CUDA error was observed after the requests.

This validation establishes the initial compatibility boundary. It is not a general support claim for other models, tensor-parallel layouts, vLLM versions, or GPU architectures.
