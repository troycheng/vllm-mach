# MXFP8 integration notes

The integration preserves the qualified model-specific entry points and their
native ABI, source guards, dual-activation dispatch and ordered GDN state
handling. The general adapter remains available through `vllm-mach-serve`.

The selected implementation adds source-qualified SwiGLU/MXFP8 quantization
fusion, expands the 4B BA/QKVZ fork/join range, and supports bounded K128 greedy
sampling with BF16 refinement. Full scoring and unsupported sampling modes
retain their original heads.

The standalone b12x package was already a dependency for the small-row BF16 BA
implementation. Its compiler/intrinsics utilities and FlashInfer's backend of
the same name are separate integration points.

The [single MXFP8 benchmark](mxfp8-benchmark.md) contains the current
per-concurrency SSM/KV matrix and comparisons with historical open-source
block FP8. Its JSON records the measurement scope and runtime identities.
