# ExLlamaV3 1.5.0

The [1.5.0 build profile](../profiles/exllamav3-1.5.0/README.md) carries Mach's BF16 I/O patch onto upstream `0740edc2da569fb99174023c1d2988b1e98cb41e`. vLLM 0.29.0, PyTorch 2.13.0, FlashInfer 0.6.18, B12X 1.3.0 and MXFP6 SM120 0.2.1 stay unchanged. Rebuild ExLlamaV3 and the separate M32/Temporal extensions together.

## Relevant upstream changes

| Change | Effect on Mach |
| --- | --- |
| Multi-row GEMM and two-stage cooperative MoE | Useful building blocks for a future MoE provider. Current Mach checkpoint-hybrid Dense execution retains its own projection routes. |
| SM120 reconstruction HGEMM | Adds `hgemm_recon` with FP16 partial sums folded into FP32 every 32 K elements. Mach keeps `hgemm`, so its existing arithmetic stays unchanged. Evaluating the new path requires routed-shape performance and model-fidelity checks. |
| Batched reconstruction and Hadamard | Can reduce launches across independent same-shape matrices, particularly MoE experts. Mach already groups QKV/QKVZ and routes checkpoint prefill projections through MXFP6. |
| Faster quantizer | Benefits future checkpoint conversion. The initial-state correction concerns K=2; existing K5/K6 weights are unchanged. |
| Deterministic GDN reduction | Applies to ExLlamaV3's native GDN implementation. Mach uses the vLLM GDN overlay. |
| Unconditional Triton imports | Compatible with the current vLLM environment, which already provides Triton. |

The MoE work includes contributions from [vcruz305 (#356)](https://github.com/turboderp-org/exllamav3/pull/356) and [creslinux (#357)](https://github.com/turboderp-org/exllamav3/pull/357), subsequently reworked upstream. See the [complete version diff](https://github.com/turboderp-org/exllamav3/compare/v1.4.9...v1.5.0).

## Validation

The port retains upstream's new multi-row implementation and adds only the existing BF16 output contract to the shared GEMM inner function. ExLlamaV3, M32 and Temporal wheels were rebuilt against the patched 1.5.0 source with Python 3.12, PyTorch 2.13.0+cu130 and CUDA 13.2.

- 206 Mach CPU tests passed; 3 skipped.
- 34 native tests passed: 11 downstream BF16 tests and 23 upstream HGEMM tests.
- Four mixed-width sliced-MGEMM cases passed with changed Graph inputs and output canaries. Maximum relative RMS was 0.001110, matching 1.4.9 and below the 0.002 limit.
- 52 real-checkpoint eager/Graph cases passed across both TP ranks. This harness checks the base BF16 and MXFP6 routes with Temporal and the optional true-M32 tile disabled; the full service separately runs with Temporal enabled.
- TP2 service passed 40/40 task checks, retaining all 40 passing cases from the saved EXL3 Champion reference (`v028-prefill-reconstruct-champion`); exact text agreed on 37/40. This checks task retention, not full-precision BF16 equivalence.
- Serial versus overlapped BA execution passed 2,592 state/output byte comparisons. Both Temporal M24 bundle shapes were activated in the service.

## Serving regression

Qwen3.8-27B K5/K6 checkpoint-hybrid, two RTX 5090 GPUs on the same PIX pair, TP2, 3000 input / 1000 output tokens. The runtime retains BF16 activations/KV, opt-in FP16 recurrent state, Temporal M24, BA overlap and lossless prefill. Maximum context is 8192, maximum sequences 32, chunked prefill 4096; prefix caching is off. FULL_DECODE_ONLY captures sizes 1/2/4/8/16/24/32.

On September 13, 2026, each version ran one sample per concurrency, with 16/32/48/64 measured requests at c4/c16/c24/c32 after per-concurrency warmup. Both variants completed all 160 requests. Input hashes, seeds and token counts matched. All other runtime dependencies and serving arguments were fixed.

Both environments used the same a10 development build (`0.1.0a10.dev0`), whose runtime source matches the released a10 apart from version metadata. Only ExLlamaV3 and its M32/Temporal builds changed.

| Concurrency | 1.4.9 output tok/s | 1.5.0 output tok/s | Change | 1.4.9 TPOT ms | 1.5.0 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 350.34 | 350.37 | +0.01% | 10.937 | 10.938 |
| 16 | 1074.78 | 1074.70 | -0.01% | 13.401 | 13.405 |
| 24 | 1342.87 | 1344.14 | +0.09% | 15.874 | 15.862 |
| 32 | 1576.45 | 1578.58 | +0.14% | 17.809 | 17.786 |

The short regression found no material throughput change. It supports compatibility with the current profile; it does not establish a speedup. The useful follow-up is to evaluate `hgemm_recon` on remaining EXL3 prefill projections and reuse the MoE work when that provider is implemented.
