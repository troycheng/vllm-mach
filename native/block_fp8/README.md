# Selected SM120 n64 block-FP8 extension

This optional wheel contains one swapped CUTLASS Pingpong instantiation: tile 64×32×128 and FP32 block scales with granularity 64/1/128. Its CUDA operator accepts only N2560, K4096/9216, M16–128 divisible by 8, E4M3 input/weight and BF16 output. The adapter supplies exact duplicated N64 weight scales. All other behavior belongs to the Python adapter's stock fallback.

The template, caller, selected tile and allocation policy preserve the frozen prototype, whose source SHA256 was `b97cec7230d85982915e70069de8b3c706d1123475c4d93413cbfd543a656e7f`. The shared template derives from vLLM's Apache-2.0 SM120 blockwise FP8 implementation. Experimental variants were removed; the exported namespace and narrow entry guards are production interfaces.

Build with the validated Torch 2.13.0+cu130, FlashInfer Python 0.6.18 CUTLASS headers, and NVCC 13.0.88:

```bash
MAX_JOBS=2 uv pip install --no-build-isolation --no-deps .
```

The installed `.so` has no Python initializer; `vllm_mach_block_fp8.library_path()` discovers it for `torch.ops.load_library`. Importing the discovery package does not import Torch or initialize CUDA. `build_metadata.json` records compiler and source/header identities in the wheel.
