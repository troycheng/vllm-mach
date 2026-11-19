# Optional SiLU/block-FP8 quantization wheel

The [block-FP8 image](../../deploy/Dockerfile.fp8) builds this SM120 wheel with
NVCC 13.0.88 and PyTorch from the
[selected runtime](../../docs/installation.md#runtime-versions). The Mach
worker loads the installed shared library with `torch.ops.load_library`.
To build the wheel separately, run from `native/fp8_activation` inside that
environment.

```bash
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2 \
  python -m pip wheel --no-deps --no-build-isolation . -w dist
```

See the [activation guide](../../docs/fp8-activation.md) for the mutation and
fallback contract and GPU qualification.
