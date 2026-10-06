# Optional SiLU/block-FP8 quantization wheel

Prebuild with Torch2.13.0+cu130 and NVCC13.0.88 for SM120. This wheel has no CUTLASS,
FlashInfer or MXFP6 dependency. The Mach worker loads the installed shared
library with `torch.ops.load_library`; it never JIT compiles on a request.

```bash
CUDA_HOME=/usr/local/cuda MAX_JOBS=2 python setup.py bdist_wheel
```

See the Mach `docs/fp8-activation.md` for the exact mutation/fallback contract
and required GPU qualification. There is intentionally no Python extension
module import; Torch operator registration loads the binary directly.
