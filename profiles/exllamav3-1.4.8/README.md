# ExLlamaV3 1.4.8 BF16 build candidate

This source upgrade has passed a native build and real-weight changing-input eager/CUDA Graph checks. Service validation is tracked in [validation.md](../../docs/validation.md). The earlier validated installation remains available in [public-install.md](../../docs/public-install.md). The official 1.4.8 wheel does not export the BF16 APIs required by Mach; use the patched build below.

`bf16-io.patch` carries the two commits from [ExLlamaV3 PR #330](https://github.com/turboderp-org/exllamav3/pull/330), ending at `d0094bc922bcf2d6cf5e948ba35f347adda3a6ca`, onto upstream tag `v1.4.8` (`6ff3a17ea7f3d0026b273d43239398d57f71b788`). The original files retain their license notices; ExLlamaV3 is MIT-licensed.

The patch applies cleanly and preserves the 1.4.8 reconstruction bounds checks. The native build used Python 3.12, PyTorch 2.13.0+cu130, the CUDA 13.2 toolkit, and target `12.0a` in the official vLLM 0.28.0 image. On two SM120 GPUs, QKV/QKVZ BF16 execution at M=1/16/24/32 matched the row-chunked reference bitwise in eager mode and CUDA Graph replay with three changing inputs per case. The separate M32 extension was disabled. These checks do not establish an end-to-end performance gain.

In the intended vLLM/PyTorch/CUDA build environment:

```bash
git clone --branch v1.4.8 --depth 1 https://github.com/turboderp-org/exllamav3.git
cd exllamav3
git apply --check /path/to/vllm-mach/profiles/exllamav3-1.4.8/bf16-io.patch
git apply /path/to/vllm-mach/profiles/exllamav3-1.4.8/bf16-io.patch
CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=4 \
  python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/exllamav3-1.4.8-*.whl
python -c 'import torch; import exllamav3_ext as e; assert callable(e.exl3_mgemm_bf16_io_grouped_had)'
```

Use the serving environment's Python and PyTorch with a complete CUDA development toolkit. Adjust `CUDA_HOME` to its location; the unmodified vLLM image lacked `cusparse.h`, so the validated build mounted the host CUDA 13.2 toolkit. Build parallelism can be raised when memory permits. This installs the native dependency for Mach; it does not install ExLlamaV3's standalone generator dependencies.

The public BF16 API takes CPU Hadamard group IDs. Leave `EXL3_BF16_IO_LEGACY_CUDA_GROUP_IDS` unset. If the separate EXL3 M32 module is needed, rebuild it using [its build instructions](../../native/exl3_m32/README.md) against this patched source. Its GPU group-ID contract is unchanged.

Native wheel SHA-256: `90606feea9a28f5423b712bcec3468ea1dc31c18229744dbfb6019e7819f9586`. The separate M32 extension and a full fidelity/performance comparison against the earlier dependency are not covered by this check.
