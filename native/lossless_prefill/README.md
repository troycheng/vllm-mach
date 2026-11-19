# Lossless BF16 prefill collective

The native MXFP6 profile can compress peer transfers in the fused AllReduce,
residual and GemmaRMSNorm calls for Qwen3.8-27B Dense prefill. The codec
reconstructs BF16 values before the original arithmetic and stores blocks that
cannot be encoded exactly as raw BF16.

The vLLM 0.34 runtime overlay selects direct SUM at physical `M=4096` and the
observed-shape codec at the 32 row counts in
[`_LONG_ROWS`](../../src/vllm_mach/mxfp6/sm120_lossless_prefill.py). It requires
contiguous BF16 tensors at hidden size 5120, SM120, TP2 and a compatible
FlashInfer TRT-LLM IPC workspace. Other shapes or layouts follow the normal
AR/Norm dispatcher.

[Setup and usage](../../docs/installation.md) covers the current image and
source-profile installation. Both build the `vllm-mach-lossless-prefill` wheel
against the selected runtime stack with CUDA 13.0. To rebuild just this wheel
in the same environment, use `python deploy/install.py --native-only --native-part lossless_prefill`
from the repository root with `--cuda-home` pointing to that toolkit.

For Qwen3.8-27B Dense MXFP6 TP2, `vllm-mach-serve` enables lossless prefill
and leaves it off during CUDA Graph capture. Use `--no-lossless-prefill`
to disable it. `--verify-prefill` compares the first eligible eager call per
norm instance and row count against FlashInfer, checking both outputs
bitwise. Run throughput measurements with verification off.

For the earlier EXL3/vLLM 0.28 callers, input-only and input+SUM modes, and
standalone diagnostics, see the [v0.4.0 archive](https://github.com/troycheng/vllm-mach/blob/v0.4.0/native/lossless_prefill/README.md).
