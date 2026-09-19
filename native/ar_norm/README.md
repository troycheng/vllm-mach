# AR norm MXFP8

`vllm-mach-ar-norm` is an optional prebuilt SM120 primitive for the Qwen3.8-27B MXFP6 TP2 decode path. It registers `torch.ops.mach_norm_quant.run` and `torch.ops.mach_norm_quant.abi()`, which returns `ar-norm-mxfp8-v1`.

The runtime integration is intentionally narrow: it may use this wheel only for BF16 TP2 hidden size 5120, pure decode batches `M=2,4,8,16,24,32`, and an available SM120 device. `M=1`, larger batches, prefill and unsupported shapes retain the original path. An eligible model with fusion enabled requires this wheel and a matching ABI; missing or stale binaries raise an installation error. Disable this producer explicitly with `--no-fused-ar-quant` when the wheel is not installed. The extension never JIT-compiles and has no dependency on experimental absolute paths.

## Build

Install the matching native MXFP6 profile first. This wheel requires the same PyTorch and `flashinfer-python==0.6.18` used by serving, a CUDA 13.0 compiler, C++17, `sm_120f`, and the retained fast-math compile flags.

```bash
cd native/ar_norm
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2 \
  python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --force-reinstall --no-deps dist/vllm_mach_ar_norm-*.whl
```

Or build all native wheels through `deploy/install.py --native-only --cuda-home /usr/local/cuda-13.0`.

## Source and license

The fused collective header is derived from FlashInfer's Apache-2.0 all-reduce fusion implementation. It is shipped in source distributions together with the Apache-2.0 license. The retained MXFP8 norm-quantization kernel body is copied unchanged from the validated Mach source; the binding adds only input diagnostics, an SM120 guard, and the package ABI operator. FlashInfer, CUTLASS, and spdlog headers are consumed from the pinned FlashInfer package at build time.
