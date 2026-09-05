# EXL3 grouped M32 extension

Optional SM120 BF16-I/O kernel for exact M32 grouped QKV/QKVZ calls. This module is built separately and does not replace `exllamav3_ext`. It is excluded from the `vllm-mach` Python wheel.

## Build

Use the same Python, PyTorch, and CUDA environment as the serving process. The extension uses shared device headers from a pinned ExLlamaV3 revision:

```bash
git clone https://github.com/troycheng/exllamav3.git /path/to/exllamav3
git -C /path/to/exllamav3 checkout d0094bc922bcf2d6cf5e948ba35f347adda3a6ca
cd /path/to/vllm-mach/native/exl3_m32
EXLLAMA_V3_SOURCE=/path/to/exllamav3/exllamav3/exllamav3_ext \
TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=2 \
python -m pip install --no-build-isolation .
```

Set `CUDA_HOME` if the desired toolkit is not the default. Do not copy an extension built for a different Python/PyTorch ABI into the serving environment.

Enable `EXL3_BF16_IO=1`, `EXL3_BF16_IO_M32=1`, and `EXL3_BF16_IO_TILE_M32=1` in Mach. The extension checks exact M32, supported quantization parameters, shapes, workspace sizes, and cooperative launch residency. Mach selects it only for grouped bundles.

## Source provenance

The implementation was extracted from the independently tested M32 prototype without changing its four native source files:

| File under src/ | SHA256 |
|---|---|
| `m32_bindings.cpp` | `f1d1ae18635d2095d6a6093476e585f456a1105ee6410c4c40530aa85d6d62a8` |
| `quant/exl3_gemm_inner_m32.cuh` | `3deb4aa62a5b8efcc264af5b28475e120cb22c23a66e0975aa4ac70852ce34fa` |
| `quant/exl3_mgemm_bf16_io.cuh` | `7fa93710899902a972a728db02ac22f70e298d6f9f9cf7fa2e2dba7ced30191f` |
| `quant/exl3_mgemm_bf16_io_m32.cu` | `963a3425c8eaaf991ef59ff1f338a7c218e02cfe37ca91f81277a8eb1f0cc4fe` |

The retained `single_had_m32_fp16` experimental binding is not dispatched by Mach and is not a supported public path. The extension derives from ExLlamaV3 and is distributed under [MIT](LICENSE).

The documented public-header build passed the Mach GPU port checks with Python 3.12, PyTorch 2.13.0+cu130, and CUDA 13.2. See [experimental decode paths](../../docs/experimental-decode.md) for the binary identity and validation limits.
