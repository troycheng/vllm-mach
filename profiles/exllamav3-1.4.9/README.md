# ExLlamaV3 1.4.9 BF16 build candidate

This is the downstream BF16 I/O patch for upstream `v1.4.9` (`5be886578ec80324c2c715269387be2058724b6e`). Native compilation, 11 BF16 tests, four upstream sliced-MGEMM cases and 52 real-checkpoint eager/Graph cases have passed on SM120 with PyTorch 2.13.0. The vLLM 0.29 TP2 service also passed 40/40 task checks and 2,592 BA state/output comparisons. The published a9 release remains on 1.4.8.

The patch carries the [1.4.8 BF16 implementation](../exllamav3-1.4.8/README.md) forward without changing its public Python API. The upstream GEMM mainloop now takes `size_n_stride` for sliced MGEMM. That argument stays in its upstream position; BF16 output arguments follow it. Mach's BF16 callers pass zero for the new stride because their packed matrices are full-width, independently allocated bundles. The BF16 output stride remains separate.

ExLlamaV3's native GEMM and sliced MGEMM callers are unchanged. Mach does not yet use the new sliced scheduling through its BF16 API. The official 1.4.9 wheel does not contain these downstream BF16 entry points.

Apply in a separate checkout:

```bash
git clone --branch v1.4.9 --depth 1 https://github.com/turboderp-org/exllamav3.git
git -C exllamav3 apply --check /path/to/vllm-mach/profiles/exllamav3-1.4.9/bf16-io.patch
git -C exllamav3 apply /path/to/vllm-mach/profiles/exllamav3-1.4.9/bf16-io.patch
```

Build with the target serving environment's PyTorch and CUDA toolkit. Rebuild the separate M32 and Temporal extensions against the same source. Before use, run the patch's BF16 tests, upstream sliced MGEMM checks, real-weight changing-input Graph checks, and TP2 service validation.

The BF16 implementation originated in [PR #330](https://github.com/turboderp-org/exllamav3/pull/330), which is closed. It remains a downstream integration, not an upstream-supported interface. ExLlamaV3 is MIT-licensed; source notices are retained.
