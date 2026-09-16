# FlashInfer / B12X LM-head audit

Measured on 2026-09-16, RTX 5090 (physical GPU 4), vLLM 0.29.0, FlashInfer 0.6.18, standalone B12X 1.3.0. The subsequent dependency cleanup removes the standalone package requirement without changing production backend selection.

## Measurement method

[Benchmark script](../tools/benchmark_nvfp4_lm_head.py) directly imports `bench_kineto` from `/data/lxy/gemm_bench.py`. Each trial uses 200 measured calls, with the reference helper's 8,000,000,000-byte buffer zeroing before each call to flush cache. Reported values are the median of three trial means.

The actual Qwen3.8-27B-MXFP6 BF16 LM-head rank-0 weight shard is quantized once. TP=2 local dimensions are N=124160, K=5120; M varies below. Activations are random BF16; every backend receives identical operands within each M. FlashInfer backends are autotuned; standalone B12X uses its default `expected_m` heuristic, not an exhaustive independently tuned optimum.

Times are CUDA **kernel-only microseconds**, not wall-clock latency. The multi-kernel metric sums each kernel's mean time multiplied by its per-call launch count. Cache flush, internal Memset/Memcpy, CPU overhead, launch gaps, top-k, BF16 refinement and TP communication are excluded. M=32 was retried separately because its internal reduction-workspace memset shares the profiler label of the cache flush; the first attempt was rejected rather than reporting contaminated timing. The two invocations reset the same random seed independently.

[Raw cold-cache results](data/nvfp4-lm-head-gemm-kineto.json) retain kernel names, launch multiplicities and all trial samples. The earlier [warm CUDA Graph results](data/nvfp4-lm-head-gemm.json) use a different protocol and are not mixed into these results.

## Pure GEMM

| M | Current vLLM → FI B12X | Direct FI B12X | Standalone B12X | FI CUTLASS |
|---|---:|---:|---:|---:|
| 1 | 235.762 | 235.956 | 237.553 | 235.549 |
| 2 | 239.172 | 239.109 | 290.201 | 236.897 |
| 4 | 240.426 | 240.856 | 344.233 | 241.552 |
| 8 | 244.258 | 243.969 | 252.720 | 242.479 |
| 16 | 245.090 | 245.316 | 288.466 | 243.713 |
| 24 | 246.933 | 247.157 | 266.916 | 246.153 |
| 32 | 247.531 | 247.687 | 250.888 | 247.722 |

## Quantization + GEMM kernel-time sum

Includes dynamic activation scale calculation, activation quantization, alpha calculation and GEMM, with the exclusions above.

| M | Current vLLM → FI B12X | Direct FI B12X | Standalone B12X | FI CUTLASS |
|---|---:|---:|---:|---:|
| 1 | 249.721 | 249.604 | 253.413 | 248.846 |
| 2 | 254.623 | 254.874 | 315.077 | 251.471 |
| 4 | 258.688 | 256.211 | 363.797 | 257.776 |
| 8 | 259.579 | 259.695 | 333.985 | 259.626 |
| 16 | 266.346 | 266.323 | 312.405 | 264.747 |
| 24 | 270.971 | 270.887 | 289.219 | 268.728 |
| 32 | 262.632 | 262.270 | 264.732 | 261.913 |

All 28 pure-GEMM comparisons have max/mean absolute difference zero and top-1 agreement 100% against the current implementation on these test inputs. This is backend equivalence on tested operands, not a new full-model fidelity claim.

The current wrapper and direct FlashInfer calls have essentially the same measured performance. CUTLASS is not uniformly faster under this protocol: M=4 is slower and M=32 is effectively tied for pure GEMM. These measurements do not establish a reliable reason to switch the production backend, and cannot be translated directly into serving throughput.

## Dependency and call-site audit

- `src/vllm_mach/mxfp6/hybrid_nvfp4_lm_head.py` already quantizes through FlashInfer and calls vLLM's `flashinfer_scaled_fp4_mm(..., backend="b12x")`. That wrapper invokes `flashinfer.mm_fp4`.
- In installed FlashInfer 0.6.18, `gemm/gemm_base.py` selects `kernels/dense_blockscaled_gemm_sm120_b12x.py`. That kernel is vendored inside FlashInfer and does not import the standalone B12X package. **Keep the backend name `b12x`**: it identifies a FlashInfer implementation, not an external package requirement; `flashinfer` is not a replacement backend name.
- Candidate top-k already uses FlashInfer, with a Torch fallback. BF16 refinement and TP argmax use local Triton kernels. MXFP6 dense GEMM uses `mxfp6-sm120`; owner/lossless prefill use custom extensions. These are not external B12X calls and are not candidates for a blanket B12X-to-FlashInfer substitution.
- There are no direct external B12X imports in current production `src` or `native` code. The new comparison benchmark intentionally imports standalone B12X as a baseline.
- Plain vLLM 0.29.0 does not require B12X; its metadata lists B12X only for the optional `b12x` extra. FlashInfer, mxfp6-sm120 and the owner/lossless-prefill packages have no B12X requirement.

The following **redundant external-package requirements have been removed**, without replacing the head's math path. Locations and descriptions below identify the pre-cleanup requirements:

| Location | Current requirement |
|---|---|
| `src/vllm_mach/mxfp6/serve.py:106` | Rejects NVFP4-head startup unless B12X 1.3.0 is installed |
| `deploy/requirements-runtime.txt:4` | Pins standalone B12X 1.3.0 |
| README support row; `docs/installation.md` | Describe/install standalone B12X for the head |
| `NOTICE` | Describes B12X as an external runtime dependency; wording needs licensing-aware correction, not removal of upstream attribution |

Backend-selection strings, their tests and the runtime patch should stay. Historical experiment receipts should retain the versions actually installed. `tools/service_probe.py` is a separate obsolete EXL3/rank64 probe importing removed code; its B12X version query is not an active GEMM call and requires separate cleanup or migration.

## No-standalone-B12X smoke test

[Reproducible audit](../tools/audit_flashinfer_b12x_dependency.py) ran successfully on physical GPU 5 in a fresh process, blocking all `b12x` imports and hiding its version lookup. Result: **PASS**, blocked import attempts: **[]**.

Verified NVFP4 preparation, FlashInfer B12X GEMM, FlashInfer top-k, BF16 candidate refinement, changing-input CUDA Graph replay, MXFP6 library loading and owner-prefill library loading. Refined candidate logits were checked against BF16 linear output.

This is a small-shape API smoke test, not a full-model serving run with the package physically uninstalled. Before cleanup, a separate read-only launcher probe with B12X metadata hidden exited with code 2 and `Install b12x==1.3.0 before enabling this option`. That check has now been removed; launcher regression tests cover both an absent and a mismatched standalone package while preserving the FlashInfer backend name. No system package was uninstalled.

## Reproduction

Use the validated environment and idle GPUs; preserve the toolkit path spelling to avoid unnecessary recompilation:

```bash
CUDA_VISIBLE_DEVICES=4 CUDA_HOME=/usr/local/cuda-13.0 PYTHONPATH=src \
  /tmp/mach-native-validation-WDlsU1/venv/bin/python tools/benchmark_nvfp4_lm_head.py \
  --model /data1/models/Qwen3.8-27B-MXFP6 --timer kineto \
  --gemm-bench /data/lxy/gemm_bench.py --repeats 200 --trials 3 \
  --rows 1 2 4 8 16 24 --output /tmp/lmhead-kineto-m1-m24.json

CUDA_VISIBLE_DEVICES=4 CUDA_HOME=/usr/local/cuda-13.0 PYTHONPATH=src \
  /tmp/mach-native-validation-WDlsU1/venv/bin/python tools/benchmark_nvfp4_lm_head.py \
  --model /data1/models/Qwen3.8-27B-MXFP6 --timer kineto \
  --gemm-bench /data/lxy/gemm_bench.py --repeats 200 --trials 3 \
  --rows 32 --output /tmp/lmhead-kineto-m32.json

CUDA_VISIBLE_DEVICES=5 CUDA_HOME=/usr/local/cuda-13.0 PYTHONPATH=src \
  /tmp/mach-native-validation-WDlsU1/venv/bin/python tools/audit_flashinfer_b12x_dependency.py
```
