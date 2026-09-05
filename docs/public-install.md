# Public EXL3 installation

The `0.1.0a4` compatibility fix is not yet published. Do not combine `0.1.0a3` BF16 grouped execution with ExLlamaV3 commit `d0094bc`; it fails with `had_group_ids must be a CPU tensor`.

## Fixed runtime contract

The public BF16 API receives CPU Hadamard group IDs. Mach prepares those IDs during weight loading and passes them through graph priming and execution. The independent M32 module still receives its original GPU IDs. Neither path copies the metadata back from the GPU during inference.

Only when using an older experimental ExLlamaV3 wheel with the GPU-metadata API, set:

```bash
export EXL3_BF16_IO_LEGACY_CUDA_GROUP_IDS=1
```

Leave this unset for the public PR #330 revision.

## Build the public dependency

Use the serving environment's Python, PyTorch, and CUDA toolkit. The source build was checked with the official `vllm/vllm-openai:v0.28.0` image, Python 3.12, PyTorch 2.13.0+cu130, and the CUDA 13.2 toolkit:

```bash
git clone https://github.com/troycheng/exllamav3.git
git -C exllamav3 checkout d0094bc922bcf2d6cf5e948ba35f347adda3a6ca
cd exllamav3
TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=4 \
  python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/exllamav3-1.4.6-*.whl
```

Set `CUDA_HOME` to the toolkit directory if needed. Increase `MAX_JOBS` when CPU and RAM allow; this changes build parallelism, not the target architecture.

This minimal installation is for Mach's direct native-extension use inside the existing vLLM environment. It deliberately does not install ExLlamaV3's standalone model/generator dependencies. Install those separately if you need the ExLlamaV3 Python API or CLI. Import PyTorch before testing the extension directly:

```bash
python -c 'import torch; import exllamav3_ext as e; assert callable(e.exl3_mgemm_bf16_io_grouped_had)'
```

The optional M32 module has [separate build instructions](../native/exl3_m32/README.md). Install the fixed Mach wheel in this same environment before using the public BF16 API.

## Without B12X

The fixed provider uses native EXL3 when B12X is absent and no B12X route was explicitly selected. Setting `VLLM_EXL3_B12X_MIN_M` to a positive value without installing B12X produces an explicit error.

For the small-memory native-prefill profile used in the service check, disable persistent reconstructed-weight caching and bound the largest temporary reconstruction:

```bash
export VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE=0
export VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB=512
```

This page covers the EXL3 path. It does not establish a clean public rebuild of the separate MXFP6/FlashInfer hybrid dependencies.
