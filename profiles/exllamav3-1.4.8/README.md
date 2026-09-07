# ExLlamaV3 1.4.8 BF16 build candidate

This source upgrade is prepared but has not passed a native build or Mach service validation. The existing validated installation remains available in [public-install.md](../../docs/public-install.md). Do not replace the serving dependency with the official 1.4.8 wheel: it does not export the BF16 APIs required by Mach.

`bf16-io.patch` carries the two commits from [ExLlamaV3 PR #330](https://github.com/turboderp-org/exllamav3/pull/330), ending at `d0094bc922bcf2d6cf5e948ba35f347adda3a6ca`, onto upstream tag `v1.4.8` (`6ff3a17ea7f3d0026b273d43239398d57f71b788`). The original files retain their license notices; ExLlamaV3 is MIT-licensed.

The patch applies cleanly. Static checks confirm BF16 binding exports and recursive CUDA source discovery. The 1.4.8 reconstruction bounds checks are unchanged. This does not establish a performance gain, binary compatibility, or CUDA Graph correctness.

In the intended vLLM/PyTorch/CUDA build environment:

```bash
git clone --branch v1.4.8 --depth 1 https://github.com/turboderp-org/exllamav3.git
cd exllamav3
git apply --check /path/to/vllm-mach/profiles/exllamav3-1.4.8/bf16-io.patch
git apply /path/to/vllm-mach/profiles/exllamav3-1.4.8/bf16-io.patch
TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=4 \
  python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/exllamav3-1.4.8-*.whl
python -c 'import torch; import exllamav3_ext as e; assert callable(e.exl3_mgemm_bf16_io_grouped_had)'
```

Use the serving environment's Python, PyTorch, and CUDA toolkit. Build parallelism can be raised when memory permits. This installs the native dependency for Mach; it does not install ExLlamaV3's standalone generator dependencies.

The public BF16 API takes CPU Hadamard group IDs. Leave `EXL3_BF16_IO_LEGACY_CUDA_GROUP_IDS` unset. If the separate EXL3 M32 module is needed, rebuild it using [its build instructions](../../native/exl3_m32/README.md) against this patched source. Its GPU group-ID contract is unchanged.

Before promoting this candidate, complete the native build, public CPU-metadata dispatch tests, changing-input Graph checks, and a Mach TP2 service smoke. Do not change the validated README install pin on the strength of a clean patch application.
