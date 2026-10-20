# Shared GDN and Mamba runtime optimizations

The shared vLLM 0.29.0 runtime profile includes the GDN optimizations ported
from revision `a569aecd64`: preserve the head dimension through gated
normalization, read row-strided mixed-decode inputs directly, reuse the output
buffer, share FlashInfer prefill offsets across layers, and fuse aligned state
block selection. Eligible SM120 prefill also reads and updates the BF16 state
pool directly. Model weights, cache dtypes and sampling settings are unchanged.

These optimizations are independent of weight quantization: none of their
admission predicates requires NVFP4. MXFP8, FP8 and unquantized BF16 models
can use them when they execute the same GDN/Mamba code and satisfy its
backend, tensor-layout and state-dtype requirements. NVFP4 is the checkpoint
format used for the serving and numerical qualification below, not a feature
gate.

These are backbone/runtime changes. The existing Mach hybrid-head and sampling
implementations remain separate. The native MXFP6 profile and block-FP8 profile
are not replaced by this runtime overlay.

## Installation and compatibility

After installing this version of Mach into the pinned vLLM environment, preview
and apply the shared runtime profile:

```bash
python -m vllm_mach.mxfp8.install
python -m vllm_mach.mxfp8.install --apply
```

The installer applies the complete `runtime.patch` to the pinned upstream
source, or makes no changes when the current profile is already installed.
It checks hashes and stages all changes before publication. Older profiles,
unknown sources and partial installations are rejected. To move from an older
profile, start from a clean pinned vLLM installation and apply the full patch. Both 2B and 4B ordered
GDN source guards are updated without disabling drift detection.

Use the normal vLLM launcher for an NVFP4 checkpoint. The model-specific MXFP8
launchers still require MXFP8 weights. Do not apply this source profile on top
of the separate native MXFP6 or block-FP8 overlays.

| Optimization | Admission |
|---|---|
| Normalization layout | CUDA GDN with three-dimensional output and gate tensors |
| Mixed decode IO | SM120, TP1, BF16 inputs/state, 16 key/value heads of dimension 128, supported strides/output layout |
| Prefill metadata reuse | FlashInfer GDN prefill; other backends retain their metadata |
| Fused block selection | CUDA Mamba/GDN with aligned state caching; CPU and `all`/`none` retain existing behavior |
| Indexed BF16 prefill state | Non-speculative SM120 FlashInfer prefill, matching 16-head state layout, supported indexed-state API and SM120 kernel |

FlashInfer 0.6.18 builds differ in SM120 indexed-state support. The public
`state_indices` argument alone is insufficient: some builds only implement it
on SM100/103. Mach checks the SM120 kernel too and preserves the packed FP32
intermediate-state path when support is absent. It does not modify FlashInfer
or silently lower SSM precision. Other optimizations remain available.

The Qwen3.5-4B model's 32 value heads do not satisfy the TP1 mixed-decode and
indexed state-pool guards. Normalization layout, FlashInfer prefill metadata
reuse and aligned state indexing do not require the 2B geometry.

| Model/path | Applicability |
|---|---|
| Qwen3.5-2B NVFP4, MXFP8, FP8 or BF16 | All five can apply under their respective guards; BF16 state is required for the two state/IO specializations |
| Qwen3.5-4B dense | Normalization, prefill metadata and aligned state indexing can apply; the two 16-head specializations retain fallback |
| Other dense or MoE models with GDN/Mamba | Shared normalization/metadata/indexing can apply where those functions are used; specialized IO depends on actual head geometry, dtype and layout |
| Models without the relevant GDN/Mamba operations | These changes do not optimize their attention or dense/expert GEMMs |

Weight format alone neither enables nor disables these paths. FP8 KV is also
not an admission requirement. Aligned state indexing requires `align` cache
mode; `all` and `none` bypass it. A model-specific ordered/persistent GDN
adapter may replace part of the shared computation, so applicability must be
checked on the executed path rather than inferred from the model name.
The separate native MXFP6 and block-FP8 source overlays have not been updated
by this port; mathematical applicability does not imply those installations
already contain these changes.

## Source conclusions and validation scope

The port follows the quantization-independent guards in `vllm-shpgy`:

- `d8a3bb8394`: CUDA normalization layout and eligible indexed BF16 prefill.
- `5165ce2630`: SM120 BF16 mixed-decode input/output reuse.
- `dad0668ce6`: FlashInfer GDN prefill metadata reused across layers.
- `a569aecd64`: aligned-state indexing shared by GDN and other Mamba metadata
  builders, including CPU fallback and speculative-state slots.

The source repository's `39d1e9fb78` compatibility fix additionally records
Qwen3.5-2B MXFP8 startup and two completed generation requests with BF16 SSM/KV
and prefix caching disabled. That establishes non-NVFP4 compatibility at that
revision; it does not establish performance or numerical equivalence for every
later optimization or every MXFP8 profile.

This port checks both 2B and 4B ordered-GDN registration. Its full paired
throughput and frozen256 numerical measurements use the 2B NVFP4 checkpoint
below. Those numerical results and speedups must not be relabeled as MXFP8,
BF16, 4B or MoE measurements.

## NVFP4 validation protocol

Qualification uses a Qwen3.5-based 2B NVFP4 checkpoint on one RTX 5090, TP1,
BF16 SSM and FP8 KV. Baseline and candidate use the same checkpoint, FlashInfer
build and full-vocabulary head. The indexed-state qualification uses a
FlashInfer 0.6.18 build whose SM120 kernels implement indexed BF16 state IO;
the standard dependency fallback is tested separately.

Performance uses the existing six-point streaming client: 3000 input and
1000 output tokens, greedy sampling with EOS ignored, concurrency
4/16/24/32/48/64, 1040 scored requests per complete curve, separate prewarm
and warmup phases, and prefix caching disabled. Report output tokens/s for
each concurrency rather than business requests/s.

The serving comparison uses the V2 runner, FULL_AND_PIECEWISE CUDA graphs,
160 sequence slots, a 2048-token scheduling budget, and a fixed 19 GiB KV
budget. Run each version in its own process with identical dependencies:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS="" VLLM_USE_V2_MODEL_RUNNER=1 \
OMP_NUM_THREADS=4 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name nvfp4-study \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 \
  --max-num-seqs 160 --max-model-len 16384 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 20401094656 \
  --mamba-ssm-cache-dtype bfloat16 --kv-cache-dtype fp8 \
  --attention-backend FLASHINFER --no-enable-prefix-caching \
  --cudagraph-capture-sizes 4 8 16 24 32 48 64 96 128 160 2048
```

```bash
python -m vllm_mach.mxfp8.benchmark \
  --base-url http://127.0.0.1:8000 --model nvfp4-study \
  --outdir ./sixpoint-results
```

Numerical validation uses the repository's frozen256 manifest and unchanged
teacher-forcing protocol at physical M4 and M32. Both arms explicitly use BF16
SSM, FP8 KV and prefix caching to exercise aligned state indexing. Each arm
scores 10479 target-token raw logprobs and repeats its first cohort. Only token
selection is forced; logits and logprob computation remain unchanged.

```bash
python tools/fidelity_native_mxfp6.py \
  --arm nvfp4 --model "$MODEL" --tokenizer "$MODEL" \
  --tensor-parallel-size 1 --attention-backend FLASHINFER \
  --manifest docs/data/fidelity-samples.json --output ./fidelity-m32 \
  --physical-rows 32 --skip-head-probe \
  --ssm-dtype bfloat16 --kv-cache-dtype fp8 --enable-prefix-caching
```

Use `--physical-rows 4` for M4 and independent processes/output directories for
the reference and candidate. This comparison measures fidelity to the same
NVFP4 model before the runtime changes; it does not measure NVFP4 quantization
error against BF16 weights or labeled task accuracy.

## NVFP4 validation results

Against the shared runtime at revision `afcb4cf6bb`, two complete serving
comparisons ran in opposite lifecycle order: candidate/reference, then
reference/candidate. Both arms used the same 64-entry FlashInfer autotune
table for final graph capture. All 4160 scored requests completed successfully.
The following values are arithmetic means of the two per-concurrency output
throughputs, in tokens/s:

| Concurrency | Reference | Optimized | Mean gain | Gains in the two runs |
|---|---:|---:|---:|---:|
| 4 | 1472.44 | 1492.69 | +1.38% | +1.50% / +1.25% |
| 16 | 4618.18 | 4771.69 | +3.32% | +3.33% / +3.32% |
| 24 | 6569.53 | 6870.95 | +4.59% | +4.80% / +4.37% |
| 32 | 7787.41 | 8177.19 | +5.01% | +5.02% / +4.99% |
| 48 | 9618.28 | 10181.02 | +5.85% | +6.24% / +5.47% |
| 64 | 10732.69 | 11473.56 | +6.90% | +7.19% / +6.62% |

These results qualify the configuration and indexed-state-capable dependency
build above. The small C4 gain should not be generalized to every workload.
This is a runtime comparison with the same NVFP4 weights, not a comparison
between quantization formats or a new head configuration.

The frozen256 comparison against the previous shared runtime produced:

| Physical batch | Queries | Scored target tokens | Mean absolute logprob error | Maximum absolute error | Changed target logprobs |
|---|---:|---:|---:|---:|---:|
| M4 | 256 | 10479 | 0 | 0 | 0 |
| M32 | 256 | 10479 | 0 | 0 | 0 |

The query-bootstrap 95% interval is `[0, 0]` in both cases. Repeated first
cohorts also match exactly in each arm. This is exact agreement for the tested
target logprobs, not a guarantee for every model, input or decoding setting.

Focused regression checks cover GPU output/state equivalence across repeated
updates, strided inputs and padded state pools, CUDA graph replay of aligned
indices, unsupported FlashInfer capability fallback, exact installation from
upstream, repeat installation, rejection of older profiles without changing
sources, and both ordered-GDN source guards.

A separate C4 serving smoke test with the original pinned FlashInfer dependency
also passed all 40 scored requests. Its real SM120 capability check returned
false, exercising the packed-state fallback. This smoke test establishes
compatibility; it is not a full performance qualification of that dependency.

## 2026-10-20: prefix caching disabled: same-GPU deployment and isolated GDN comparison

All three arms use the same Qwen3.5-based 2B NVFP4 checkpoint on one RTX 5090.

All six lifecycles succeeded: 5760/5760 scored business requests and 6240/6240 fixed-length requests. Across concurrency points, fixed-length throughput changes 9.42% to 48.45% versus official and 1.94% to 6.88% for GDN off/on. Business QPS changes 3.49% to 26.64% versus official and 2.46% to 9.25% for GDN off/on. Prefix caching is disabled throughout. See the fidelity section for numerical results and limits.

### Compared configurations

| Arm | Runtime | Cache precision and scope |
|---|---|---|
| Official | Unmodified official vLLM 0.29.0 | V2 runner; model-default FP32 SSM and BF16 KV; default attention backend |
| GDN off | Optimized runtime with the GDN changes disabled | BF16 SSM, FP8 KV; other runtime and head optimizations retained |
| Optimized | Optimized runtime with the GDN changes enabled | BF16 SSM, FP8 KV; same dependencies, head and flags as GDN off |

Both deployments report Torch 2.13.0, FlashInfer 0.6.18, Transformers 5.16.1
and Triton 3.7.1. Matching version strings do not establish identical builds.
Official-versus-optimized measures the complete deployment, including cache
precision, backends and runtime differences. GDN off/on isolates the GDN
changes within the same environment and shares the FlashInfer autotune cache.
These deployment results are separate from the Mach runtime qualification above.

### Protocol

All scored runs use TP1, the V2 runner, maximum sequence length 16384,
maximum sequences 160, a 2048-token scheduling budget and 19 GiB KV allocation.
Prefix caching is explicitly disabled with `--no-enable-prefix-caching`.
FULL_AND_PIECEWISE CUDA Graph sizes are
`4,8,16,24,32,48,64,96,128,160,2048`.

Each arm runs twice with a fresh engine, in this order: official → GDN off →
optimized → optimized → GDN off → official. Performance values are arithmetic
means of the two runs. Failed setup attempts are excluded from scoring.

Business testing uses the first 80 records of each Chinese/math/English text/image group, concurrency 4 and 16, temperature 0, seed 20261020, thinking disabled, max output 2048. Each group/concurrency warms its first five records before scoring; multimodal encoder cache follows the same ordered lifecycle. Payloads/images are prepared before timing. QPS includes client dispatch and streaming processing; per-request TTFT is measured at the first received token ID or nonempty text event, and latency through the final stream event. Each arm has 1920 scored business requests across two runs. The five-request warmup does not guarantee coverage of every tensor shape. Any inference-time JIT latency remains included; these are end-to-end business measurements.

Mach synthetic uses the unchanged `vllm_mach.mxfp8.benchmark` contract: exactly 3000 input and 1000 output tokens, ignore EOS, fixed seeds, concurrency 4/16/24/32/48/64, 1040 scored requests per run. Prewarm/internal warmup are excluded from throughput. Prefix caching is disabled for all requests, including suite warmup traffic. Each scored request has validated usage of exactly 3000 prompt and 1000 completion tokens.

### Fixed-length output throughput

| Concurrency | Official tokens/s | GDN off tokens/s | Optimized tokens/s | Deployment gain | GDN code gain |
|---|---:|---:|---:|---:|---:|
| 4 | 1359.67 | 1459.47 | 1487.79 | +9.42% | +1.94% |
| 16 | 3979.49 | 4680.37 | 4829.67 | +21.36% | +3.19% |
| 24 | 5347.35 | 6503.85 | 6778.92 | +26.77% | +4.23% |
| 32 | 6128.88 | 7690.79 | 8040.96 | +31.20% | +4.55% |
| 48 | 7154.66 | 9594.25 | 10150.96 | +41.88% | +5.80% |
| 64 | 7730.77 | 10738.02 | 11476.35 | +48.45% | +6.88% |

### Business: official deployment → optimized

Token counts are mean total completion tokens per 80-request group. TTFT and latency are means in milliseconds.

| Group | C | QPS reference → optimized | Gain | TTFT ms | Latency ms | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| ch_text | 4 | 16.21 → 19.71 | +21.60% | 112.8 → 90.3 | 227.2 → 184.7 | 3068.5 → 3074.0 |
| ch_text | 16 | 25.74 → 29.06 | +12.88% | 291.7 → 316.9 | 492.9 → 457.4 | 3070.5 → 3065.0 |
| ch_image | 4 | 13.00 → 14.85 | +14.21% | 174.5 → 152.5 | 287.1 → 250.6 | 3051.0 → 3052.5 |
| ch_image | 16 | 22.54 → 23.32 | +3.49% | 404.9 → 433.0 | 604.6 → 588.1 | 3072.0 → 3056.5 |
| math_text | 4 | 25.04 → 31.62 | +26.26% | 99.9 → 79.1 | 158.3 → 125.0 | 1642.0 → 1647.5 |
| math_text | 16 | 41.86 → 50.17 | +19.84% | 232.5 → 208.4 | 360.6 → 304.9 | 1649.5 → 1643.0 |
| math_image | 4 | 18.18 → 22.10 | +21.56% | 159.9 → 129.6 | 216.4 → 178.9 | 1629.0 → 1645.5 |
| math_image | 16 | 36.30 → 40.27 | +10.95% | 277.5 → 280.2 | 421.7 → 376.8 | 1649.0 → 1651.0 |
| en_text | 4 | 26.28 → 33.28 | +26.64% | 117.1 → 92.2 | 150.9 → 119.3 | 1248.5 → 1249.5 |
| en_text | 16 | 43.54 → 48.51 | +11.41% | 240.4 → 225.9 | 355.9 → 314.2 | 1250.5 → 1250.0 |
| en_image | 4 | 18.81 → 22.15 | +17.73% | 171.3 → 144.3 | 209.6 → 176.7 | 1247.0 → 1250.0 |
| en_image | 16 | 34.63 → 37.47 | +8.19% | 318.3 → 314.8 | 443.9 → 400.7 | 1255.0 → 1252.0 |

### Business: isolated GDN off → on

Token counts are mean total completion tokens per 80-request group. TTFT and latency are means in milliseconds.

| Group | C | QPS reference → optimized | Gain | TTFT ms | Latency ms | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| ch_text | 4 | 18.45 → 19.71 | +6.85% | 100.6 → 90.3 | 198.0 → 184.7 | 3053.0 → 3074.0 |
| ch_text | 16 | 28.36 → 29.06 | +2.46% | 293.9 → 316.9 | 467.3 → 457.4 | 3069.0 → 3065.0 |
| ch_image | 4 | 14.30 → 14.85 | +3.84% | 158.7 → 152.5 | 262.3 → 250.6 | 3070.0 → 3052.5 |
| ch_image | 16 | 22.53 → 23.32 | +3.52% | 434.4 → 433.0 | 589.8 → 588.1 | 3061.0 → 3056.5 |
| math_text | 4 | 29.53 → 31.62 | +7.06% | 85.9 → 79.1 | 134.8 → 125.0 | 1643.0 → 1647.5 |
| math_text | 16 | 47.71 → 50.17 | +5.16% | 211.7 → 208.4 | 319.9 → 304.9 | 1632.0 → 1643.0 |
| math_image | 4 | 20.63 → 22.10 | +7.16% | 138.4 → 129.6 | 190.2 → 178.9 | 1657.0 → 1645.5 |
| math_image | 16 | 38.87 → 40.27 | +3.60% | 276.7 → 280.2 | 393.0 → 376.8 | 1638.0 → 1651.0 |
| en_text | 4 | 30.46 → 33.28 | +9.25% | 98.7 → 92.2 | 129.4 → 119.3 | 1251.0 → 1249.5 |
| en_text | 16 | 45.73 → 48.51 | +6.07% | 240.8 → 225.9 | 337.9 → 314.2 | 1252.0 → 1250.0 |
| en_image | 4 | 21.25 → 22.15 | +4.26% | 151.6 → 144.3 | 186.5 → 176.7 | 1245.5 → 1250.0 |
| en_image | 16 | 36.35 → 37.47 | +3.08% | 310.2 → 314.8 | 417.2 → 400.7 | 1244.5 → 1252.0 |

### Cache and executed backends

Prefix caching is explicitly disabled in all six launches and confirmed by runtime configuration. Prefix counter deltas are retained below; zero query counters mean a hit rate is not applicable.

| Arm | Business prefix hits / queries | Synthetic prefix hits / queries |
|---|---:|---:|
| official | 0 / 0 | 0 / 0 |
| off | 0 / 0 | 0 / 0 |
| on | 0 / 0 | 0 / 0 |

All arms execute the V2 runner and NVFP4 FlashInfer CUTLASS GEMM. Official defaults select Triton/FLA GDN prefill, FlashAttention attention with LBNHC KV layout and FlashInfer sampling. Off/on select FlashInfer GDN prefill and the SM120 legacy CUDA decode path; they use FlashInfer attention with LBHNC KV layout and the existing hybrid NVFP4 head (128 configured candidates, exact unsorted FlashInfer top-k). Generic FlashInfer sampling is disabled by that runtime.

### Numerical fidelity and output consistency

Each lifecycle scores all 256 queries / 10479 target tokens at HTTP client concurrency 4 and 32 using echoed prompt logprobs. These are teacher-forced prefill conditional logprobs, not controlled physical M4/M32 decode batches. They measure fidelity to this same NVFP4 checkpoint, not labeled business accuracy or error against BF16 model weights.

| Comparison | Client C | MAE, two runs | Max absolute error, two runs | Changed target tokens, two runs |
|---|---:|---:|---:|---:|
| official_to_on | 4 | 0.60191192 / 0.60520989 | 14.749989 / 14.749989 | 10138 / 10138 |
| official_to_on | 32 | 0.60485622 / 0.61705585 | 16.487346 / 16.487346 | 10128 / 10130 |
| off_to_on | 4 | 0.23808155 / 0.14456388 | 10.657141 / 19.169251 | 4155 / 2379 |
| off_to_on | 32 | 0.10923211 / 0.086945469 | 9.2546177 / 7.71875 | 2201 / 2239 |
| official_repeat | 4 | 0.039237676 | 6.8748903 | 558 |
| official_repeat | 32 | 0.048239926 | 7.34375 | 996 |
| off_repeat | 4 | 0.15066953 | 19.169251 | 2911 |
| off_repeat | 32 | 0.092427813 | 7.71875 | 2038 |
| on_repeat | 4 | 0.074356739 | 9.3631358 | 1091 |
| on_repeat | 32 | 0.1265299 | 9.2546177 | 2682 |

Client concurrency itself changes prefill packing. Its within-arm C4-versus-C32 differences provide context for interpreting the cross-arm errors:

| Arm | MAE, two runs | Max absolute error, two runs | Changed tokens, two runs |
|---|---:|---:|---:|
| official | 0.093844487 / 0.095006511 | 7.34375 / 6.8748903 | 1598 / 1693 |
| off | 0.60311782 / 0.60814613 | 16.500017 / 16.919268 | 9939 / 9957 |
| on | 0.59324041 / 0.58501076 | 16.500017 / 16.500017 | 9810 / 9573 |

Business generated-token exact matches (same input/concurrency/seed; scheduling can still affect outputs):

| Comparison | Matching token hashes | Match rate |
|---|---:|---:|
| official_to_on | 1688/1920 | 87.92% |
| off_to_on | 1756/1920 | 91.46% |
| official_repeat | 902/960 | 93.96% |
| off_repeat | 863/960 | 89.90% |
| on_repeat | 883/960 | 91.98% |

The observed nonzero cross-arm errors and nonzero same-arm repeat errors mean this HTTP collection does **not** establish numerical equivalence or isolate a regression caused by GDN code. A controlled physical-batch decode rerun remains necessary before claiming accuracy acceptance for this environment. The earlier historical exact-agreement table is not evidence for this new configuration. The business package provides no reviewed ground-truth labels; output mismatch alone is not evidence of task accuracy loss, and exact agreement alone does not establish general accuracy. Output-length differences must be considered when interpreting its QPS; the fixed-length synthetic comparison removes that particular confound.
