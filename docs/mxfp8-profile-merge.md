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

Only the [merged-profile validation results](mxfp8-optimization-results-20261012.md)
are retained for this integration. The result report includes the historical
open-source FP8 reference and identifies its measurement scope.
