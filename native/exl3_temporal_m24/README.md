# Temporal M24 K6 extension

Optional EXL3 QKV/QKVZ kernel for SM120. The plugin selects it only at physical M=24, K=5120, and K6: eight matrices with N=1024 use ten K splits; fourteen matrices with N=512 use twelve. All other shapes retain their existing routes. M32 in the direct-checkpoint profile still uses MXFP6.

Build separately against the same patched ExLlamaV3 source, PyTorch and CUDA runtime used by the service:

```bash
cd native/exl3_temporal_m24
CUDA_HOME=/usr/local/cuda-13.2 EXLLAMA_V3_SOURCE=/path/to/checkout/exllamav3/exllamav3_ext \
  TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=4 python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/exllamav3_temporal_m24_ext-*.whl
```

Enable before starting vLLM:

```bash
export EXL3_BF16_IO=1
export EXL3_BF16_IO_M24=1
export EXL3_TEMPORAL_QKV_M24=1
```

The switch defaults to off. A missing extension logs a warning and falls back to the existing BF16 path; a broken installed extension is an error. The base BF16 extension remains required. The native entry point consumes GPU pointer arrays and GPU Hadamard group IDs prepared by Mach's validated bundle loader. It is not a general tensor API.

For CUDA Graph serving, include 24 in the capture sizes. Request concurrency alone does not determine the physical row count after padding. The release build was checked against the patched ExLlamaV3 1.4.8 source profile.

The device implementation preserves the source experiment's K64 tiling, grouped input Hadamard transform, FP32 partial sums and final transform. It does not repack or requantize weights. Its reduction order differs from the existing EXL3 kernel, so it is not claimed to be bitwise equivalent to that kernel. The experimental service had a c24 gain and a c4 regression; those numbers are not measurements of this Mach release. Private FLA tuning is not included.

Only the grouped M24 K6 entry point is exported. K5, compact FP16 partials and other prototype dispatches are not exposed. The source uses MIT-licensed ExLlamaV3 device helpers; see LICENSE. This extension is not included in the vllm-mach Python wheel.
