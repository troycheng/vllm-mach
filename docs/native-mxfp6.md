# Native MXFP6 integration

This profile ports the optimization series based on vLLM `v0.29.0`
(`98dff2a81d747d1dba01a47f939f48c3526d4206`), ending at
`8983c0369eb5dfae42194261d06d8fbbdaab15e6`.
The optimization checkout is not a runtime dependency.

## Imported boundaries

| Source commits | Mach integration |
|---|---|
| `b4ac4e3d7a`, `e77a2e5dc3` | Native dense kernel, Quark MXFP8 metadata, accurate backend reporting, both runners' graph/workspace lifecycle |
| `6b57b3ba2d` | Qwen TP2 AllReduce/residual/GemmaRMSNorm, including final norm |
| Mach `8c5021a` | Persistent small-batch GDN and BA overlap port, packaged with null-slot-zero compatibility |
| `9fb75fd71c` | Narrow FP16 SSM admission for native packed GDN decode |
| `9f1c8fec69`, `7c688885a7`, `b032cc235b` | Lossless prefill dispatch, GDN graph break, warmed opt-in codec graphs |
| `b7b4fb089e`, `f7688f8907`, `a2ce755fde` | Owner row transport, replicated MLP, shape-dependent dispatch down to 512 rows |
| `ff180d2371`, `412d11fb26` | NVFP4 LM head candidate search/refinement, compact greedy sampler and request fallbacks |

Mach reuses its existing native dense kernel and warmup implementation. The
general plugin registers the kernel; the source patch does not import Mach from
vLLM's linear registry, avoiding an import cycle. The framework modules
live under `vllm_mach.mxfp6`. GEMM CUDA implementations remain in `mxfp6-sm120`, with prefill code in
the `native/lossless_prefill` / `native/owner_prefill` wheels, now required
for Dense default (unless both prefill options are explicitly disabled).
The persistent GDN source is packaged under `vllm_mach.mxfp6.gdn` and JIT-built
through FlashInfer during warmup.

The wheel includes a 13-file dense source patch, a separate Quark MoE patch,
a version/file manifest and
FlashInfer local IPC patch. `vllm-mach-install` checks the official dependency
versions, stages both patches and checks patch applicability before writing.
Incompatible or partially patched installations are rejected. Repeat installation
is a no-op. The source patch and plugin must both
be installed; the launcher checks the profile before starting workers.

The EXL3 implementation, old runtime overlays and offline asset tools are removed
and remain available in Git history. The default pytest suite targets
the current native MXFP6 package, warmup and deployment paths.

## Numerical and execution contracts

The base path retains packed checkpoint weights and dynamically quantizes
activations to MXFP8. Manual AR/Norm fusion changes reduction/rounding order.
The launcher uses BF16 activations/KV, TP2/PP1, V2 runner, non-speculative
text-only dense inference and full decode graphs at 1/2/4/8/16/24/32 rows.

The default launcher also enables persistent GDN at physical M1/2/4/8 and
BA overlap at M16/24/32. Persistent changes arithmetic and supports FP32 or FP16 SSM storage
with FP32 accumulation. Both routes have independent opt-outs and
retain native execution for unsupported calls. See [GDN validation](gdn-decode.md)
and the [current measurements](native-fidelity.md).

FP16 SSM is opt-in because it changes state precision. Owner prefill retains
the source series' geometry checks and numerical verification mode; it chooses
communication and MLP paths by row count. Lossless codec graph capture is
implemented but disabled in the default serving profile.

The optional NVFP4 head retains the checkpoint BF16 head, creates an NVFP4 copy,
selects 128 candidates, refines with BF16 weights/FP32 accumulation and exchanges
TP argmax pairs. This approximate search can miss the BF16 winner and rounding
can change near ties. It accelerates eligible greedy decode only. Full logits,
logprobs, stochastic sampling, processors, structured outputs and mixed/prefill
batches retain the normal path. The default compact BF16 argmax uses the original
head projection without NVFP4 search.

## Dense producer fusion

The optional Dense TP2 producer routes prepare during MXFP6 warmup. They use
the extension's GDN norm/output and SwiGLU/down-projection operations only for
the supported BF16 CUDA decode geometry; other rows, unsupported models, and
MoE retain the existing path. `VLLM_MACH_FUSED_GDN_QUANT` and
`VLLM_MACH_FUSED_SWIGLU_QUANT` accept `auto` (default), `0`, and `1`. `auto`
selects the operation for supported geometry. Missing required native
operations are an installation/startup error; there is no silent old-library
fallback. `0` is an explicit diagnostic opt-out. See [Dense producer
fusion](dense-producer-fusion.md) for the narrow scope and current evidence.

## Validation

Use a clean runtime after [installation](installation.md):

```bash
CUDA_VISIBLE_DEVICES=0,1 MXFP6_AUTOTUNE=off python -m pytest -q
```

The tests cover staged installation, idempotency, rejection without writes,
EXL3-free plugin registration, checkpoint scale packing, changing-input graph
replay, request eligibility, full-logit preservation, FP16 admission guards,
prefill shape/workspace contracts, and optional native TP2 transport/codec
comparisons. GPU and optional-extension tests skip when prerequisites are absent.

The September 16 GDN port passed all **168 current tests**, checked in groups,
including CPU routing/data checks and GPU graph/prefill checks. The standalone
persistent harness additionally passed 16 FP32/FP16 × SD/DS × M1/2/4/8 cases with
changing inputs, slot reuse, null-slot protection and bitwise graph/eager
agreement. The rebuilt wheel includes the GDN CUDA source and matches the
package source. [Measured serving/fidelity results](native-fidelity.md) and
[individual GDN ablations](gdn-decode.md) are recorded separately.

The final September 15 rerun passed **113/113 tests**, with no skips, including
the published four-profile serving counts/throughput, fidelity bootstrap and
head-recall checks. It ran on SM120 GPUs 4/5 with the optional extensions
installed. Dependency deprecation warnings were reported; there were no test
failures. The final wheel contains only the native provider, matches the source
package files, and passes the installed-profile preflight without changes.

Performance measurements from the original optimization checkout are not
automatically claims about this packaged integration. Use equal model, topology,
request shape, graph configuration and KV allocation when comparing runs.

### Integration acceptance (2026-09-15)

The initial native test suite passed **102/102**, including real SM120 graph replay,
TP2 lossless codec comparisons, owner norm/transport comparisons and NVFP4
full-logit fallbacks. The installer was also applied to newly downloaded
official vLLM 0.29.0 and FlashInfer 0.6.18 wheels: the first application passed,
and the second changed no files. The serving environment's Python, CUDA source,
headers and shared libraries were compared against those patched official
wheels with no differences. No optimization checkout was on its import path.

The acceptance service completed 64/64 requests at 3000 input / 128 output
tokens with all optional switches and prefill verification enabled. Four
non-thinking Chinese chat checks returned the expected arithmetic, capital,
translation and chemical-formula answers. Nine completion probes exercised
greedy, logprob, stochastic and token-bias requests successfully. These are
smoke checks, not a broad model-quality evaluation or a bitwise-equivalence claim.

This earlier integration acceptance uses uniform token-ID prompts; the README's
current [profile comparison](native-fidelity.md) uses frozen ShareGPT
prefixes and separately validated stock baselines. Do not combine the two
workloads' throughput numbers.

Acceptance serving comparison: two RTX 5090 GPUs, TP2/PP1, BF16 activations/KV, V2 runner,
TRITON_ATTN, no prefix cache, 4096 scheduled tokens, maximum length 16384,
graph sizes 1/2/4/8/16/24/32, fixed KV allocation 8218214400 bytes/rank.
Each arm used a 32-request warmup followed by 64 scored requests at concurrency
32, with 3000 input / 1000 output tokens, greedy sampling and ignored EOS.
Both retained arms completed 192000 input and 64000 output tokens, with no
preemption warnings. Runs were sequential on the same GPU pair, once per arm.

| Configuration | Output token/s | Mean TPOT | BS32 decode iteration median |
|---|---:|---:|---:|
| Default Mach profile: AR/Norm + compact BF16 greedy TP argmax | 1419.6 | 19.81 ms | 13.97 ms |
| Full Mach profile: AR/Norm, FP16 SSM, lossless/owner prefill, NVFP4 head | 1646.8 | 17.20 ms | 12.25 ms |

The historical default profile above retained FP32 recurrent state without
prefill extensions or NVFP4 search. The current Dense default enables both
prefill extensions. The full profile uses FP16 and approximate
NVFP4 head search. This measures the combined profile, not an isolated kernel
gain. Iteration medians include warmup; request throughput and TPOT exclude it.
One run per arm does not establish a confidence interval.

[Machine-readable configuration and results](data/native-mxfp6-acceptance.json)
use identical workload settings for both retained arms. The full profile ran
before the default profile. To reproduce the default arm,
use `vllm-mach-serve --model MODEL --kv-cache-memory-bytes 8218214400
--no-gdn-persistent --no-gdn-ba-overlap
--no-lossless-prefill --no-owner-prefill` without the optional acceleration flags.
The opt-outs are needed to reproduce these September 15 measurements with
the current launcher.

To reproduce the scored workload against either service:

```bash
python tools/benchmark_native_mxfp6.py \
  --base-url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL_NAME \
  --num-prompts 64 --input-tokens 3000 --output-tokens 1000 \
  --max-concurrency 32 --warmup-requests 32 --warmup-output-tokens 128 \
  --contract-seed 20260915 --request-seed-base 2026091500 \
  --json-out result.json
```

Start the optimized service with the full command in the installation guide.
For the historical default profile, use the opt-outs above and keep
the same KV allocation.

The native extension wheels used for acceptance matched the existing compiler
and ABI contracts; this turn did not rebuild them. Docker was unavailable in
the validation environment, so the revised Dockerfile was not built here.

Qwen3.5-35B-A3B uses the separate [MoE TP2 integration](qwen35-moe.md),
with adapters imported from `mxfp6_sm120` revision `cd4e964c391fcb8aaf1a27d28a63d778e3a38ece`.

## MXFP8 TP1 Gemma normalization and attention (September 30)

Native MXFP8 TP1 models on SM120 now fuse GemmaRMSNorm, its weight offset,
optional residual addition, and output casts in one Triton kernel. This covers
all 65 backbone/final norms in Qwen3.5-4B. The kernel accumulates in FP32 and
stores BF16 outputs; its reduction order can differ from PyTorch. Unsupported
inputs retain the original path. TP2 norm/communication producers are unchanged.
Set `VLLM_MACH_FUSED_GEMMA_NORM=0` before launching to restore the old norm path.

On RTX 5090 GPU4, TP1, BS16, 3000 input tokens, 20 stable decode steps:

| Configuration | GPU kernels/step | Sum of kernel time | Full attention time |
|---|---:|---:|---:|
| Original norms + Triton attention | 1412 | 7.089 ms | 1.075 ms |
| Fused norms + Triton attention | 635 | 5.940 ms | 1.076 ms |
| Fused norms + FlashAttention 2 | 651 | 5.948 ms | 1.058 ms |
| Fused norms + FlashInfer XQA | 627 | 5.972 ms | 1.099 ms |

These are GPU kernel sums under PyTorch profiling, not client ITL. The 65 fused
norm kernels total about 0.10 ms/step. Neither alternative attention backend
improved total kernel time materially, so the launcher retains `TRITON_ATTN`.
Users can select `--attention-backend FLASH_ATTN` (FA2 on this SM120 build) or
`--attention-backend FLASHINFER` explicitly.

Frozen256 fidelity used physical M32 and 10,479 teacher-forced target tokens.
Mean per-query absolute gold-token logprob error relative to the existing BF16
reference was 0.061120 for the original path, 0.059773 for fused norms/Triton,
0.060268 for fused norms/FA2, and 0.061874 for fused norms/FlashInfer. All repeated
cohorts matched exactly. Paired bootstrap intervals for changes relative to the
original path include zero; this sample does not establish an accuracy change.

The fidelity tool accepts `--attention-backend` for reproducing these comparisons.
Numerical, changed-input CUDA graph, warmup, deployment, and fidelity regression
checks passed (70 tests). Detailed traces and scripts are retained locally under
`audit/mxfp8-norm-attention/`; the [result summary](data/mxfp8-norm-attention-20260930.json)
includes profiles, fidelity, and serving measurements.

The matched serving run used 64 requests, concurrency 16, 3000 input tokens and
1000 output tokens, with 16 warmup requests. Original norms/Triton achieved
1924 output tokens/s and 7.243 ms median ITL; fused norms/Triton achieved
2266 output tokens/s and 6.099 ms median ITL. Mean ITL fell from 7.949 to
6.735 ms. This is a single run per configuration, including prefill contention.

### Optional GDN TP1 adapter

`VLLM_MACH_GDN_TP1=1` enables the existing GDN routes for native
Qwen3.5-4B MXFP8 TP1: persistent at M1/2/4/8 and BA overlap at M16/24/32.
The adapter derives local head counts from TP size; the TP1 geometry has
hidden width 2560, 16 key heads, 32 value heads, and packed QKV width 8192.
Prefill, mixed, and unsupported calls use the original vLLM path. Existing
`--no-gdn-persistent` and `--no-gdn-ba-overlap` options control the two routes.

At BS16, BA overlap increased serving output throughput from 2266 to 2279
tokens/s in one run (0.6%); this is too small to establish a stable gain.
At BS4, persistent reduced profiled model-forward time from 4.189 to 3.766 ms
on GPU7, with fused Gemma norms enabled in both arms.
Matched BS4 serving (16 requests, 3000 input / 1000 output tokens) improved from
771 to 839 output tokens/s (+8.8%); mean ITL fell from 5.041 to 4.622 ms.

The M32 overlap fidelity results were identical to the fused-norm baseline.
At M4, mean per-query logprob error relative to the matched BF16 reference was
0.061322 without persistent and 0.060525 with persistent; the paired difference
95% interval was [-0.002983, 0.001416]. Both repeats matched exactly. These
Frozen256 results show no measured fidelity regression, but do not imply
identical generation. The adapter remains opt-in while throughput evidence is
limited. Combined regression checks passed 146 tests, including TP1 persistent
M1/2/4/8 and overlap M16/24/32 against native state updates.

### Native versus FlashInfer MXFP8 GEMM

The [paired measurements](data/mxfp8-flashinfer-gemm-20260930.json) use
FlashInfer `mm_mxfp8(backend="cutlass")`, autotuned for the exact batch. The
model experiment replaces only GEMM: packed weights, the native activation
quantizer, fused Gemma norms, and Triton attention stay the same; TP1 GDN is off.
Each profile covers 20 stable decode steps at a 3000-token input length.

| Batch | Native: 128 GEMMs/step | FlashInfer: 128 GEMMs/step | FlashInfer/native |
|---|---:|---:|---:|
| 1 (GPU6) | 2.563 ms | 2.774 ms | 1.082x |
| 16 (GPU4) | 2.690 ms | 3.021 ms | 1.123x |

Separate single-GEMM cold-cache measurements on GPU4 use 8 GB eviction, five
interleaved trials of 30 replays, and exclude quantization and eviction time.
Their model-weighted sums are 2.913/3.565 ms at BS1 and 2.964/3.595 ms at
BS16 (native/FlashInfer). These sums are microbenchmark estimates, not ITL.

The native pool is frozen before graph capture, with three stream lanes totaling
984,192 bytes in the measured model. Stable decode profiles have no CUDA
allocation/free calls and only one small non-graph memset per step (4 bytes at
BS1, 64 bytes at BS16), rather than repeated Stream-K workspace clears.
Launch counters count host dispatch/capture, not graph replays.

The original planning set left an eager/prefill coverage gap: an isolated
shape sweep found workspace fallbacks at M64/128/256 for out/down projections,
and M512 for down projection. The model's cumulative fallback counter was 64
after startup and increased by 64 per BS16 generation including prefill; it
remained unchanged across the BS1 measured generation. In an isolated expanded
plan including M64/128/256/512/1024/2048, capacity rose from 328,064 to 5,243,520
bytes per lane and all 75 probed shapes ran without fallback. The expanded planning policy is now integrated by default. A complete model
startup and four generation calls kept the fallback counter at zero; three
stream lanes reserve 15,730,560 bytes in total. Decode GEMM time stayed at
2.690 ms/step. The fix removes the observed prefill allocation fallback; it
does not explain the original steady decode ITL.

### FlashInfer B12x integration from vllm-shpgy

The reference `/data/luyufan/vllm-shpgy` at commit `1d0591245c` selects
FlashInfer `b12x` for Qwen3.5-4B's five projection shapes at M <= 32, and
`cutlass` above that boundary. Its `b12x-decode` name is a vLLM wrapper, not
a backend accepted directly by FlashInfer. The earlier CUTLASS comparison
therefore did not cover its optimized decode route.

Set `VLLM_MACH_MXFP8_BACKEND=flashinfer` before `vllm-mach-serve` to select the
ported dispatch. `native` remains the default. Both paths use the existing
packed weights and native activation quantizer, fused Gemma norms, BF16 lm-head,
and Triton attention for the comparison. This is a GEMM integration, not a
benchmark of the entire reference repository. The reference's SiLU/norm plus
quantization pattern passes require compilation and do not run in the current
`mode=NONE` profile.

Warmup tunes each distinct weight geometry for the decode capture buckets and
maximum prefill size, before graph capture. Runtime calls disable autotuning.
The reference explicitly adds M32 to its maximum-prefill tuning pass; this
adapter also tunes M1/2/4/8/16/24 separately. FlashInfer caches its 32 MiB
workspace, compiled CuTe kernels, and device alpha constant during warmup.
The B12x runner itself does not use the native Stream-K reduction/barrier arena.

Matched profiles, 20 decode steps, 3000 input tokens, TP1, GDN TP1 disabled:

| Batch | Native GEMM sum | FlashInfer CUTLASS | FlashInfer B12x |
|---|---:|---:|---:|
| 1, GPU6 | 2.563 ms | 2.774 ms | 2.578 ms |
| 16, GPU4 | 2.690 ms | 3.021 ms | 2.945 ms |

Each row sums 128 GEMMs per step. Total GPU kernel sums were respectively
4.194/4.419/4.225 ms at BS1 and 5.942/6.274/6.196 ms at BS16. These are
profiled GPU durations, not serving ITL. No CUDA malloc/free events occurred
in these stable decode traces. B12x improves on CUTLASS here but does not
improve on the native backend; the BS1 difference from native is small.

The cold-cache comparison on GPU5 uses identical quantized operands, 8 GB
eviction per replay, five interleaved trials of 30 replays, and reports median
trial means. Each cell below is native / CUTLASS / B12x in microseconds:

| N | K | Calls/step | BS1 | BS16 |
|---:|---:|---:|---:|---:|
| 12288 | 2560 | 24 | 24.74 / 24.87 / 24.52 | 24.57 / 24.89 / 24.72 |
| 10240 | 2560 | 8 | 21.10 / 21.32 / 21.30 | 21.31 / 21.24 / 21.22 |
| 2560 | 4096 | 32 | 11.72 / 16.21 / 14.45 | 11.94 / 17.11 / 16.59 |
| 18432 | 2560 | 32 | 35.16 / 35.30 / 35.44 | 35.27 / 35.49 / 35.38 |
| 2560 | 9216 | 32 | 20.27 / 35.49 / 33.04 | 22.12 / 36.07 / 35.80 |

Native's main advantage is the out/down projections. This is consistent with
its shape-specific Stream-K dispatch, but this comparison alone does not
isolate the causal contribution of scheduling from other kernel differences.

Frozen256 fidelity at physical M32 covered 10,479 teacher-forced tokens. Mean
per-query absolute gold-token logprob error versus BF16 was 0.059773 for
native and 0.060854 for FlashInfer. The paired change was +0.001081, with
95% bootstrap interval [-0.001282, 0.003416]; this sample does not establish
a fidelity change. Repeats matched exactly. Changed-input and zero-input
CUDA graph checks covered all five projection shapes at M1/16/32, with
relative output error below 0.005 versus native.

Three GPU integration checks and 15 warmup/fidelity regression checks passed.
The [result summary](data/mxfp8-b12x-integration-20260930.json) contains the
timings, cold-cache trial samples, fidelity, and before/after workspace counters.
Reproduction scripts and traces are under `audit/mxfp8-flashinfer-gemm/`.

### MXFP8 with vLLM's default compiler and CUDA graphs

The MXFP8 launcher now leaves compilation configuration to vLLM. In this build
that selects `VLLM_COMPILE` and `FULL_AND_PIECEWISE`, including piecewise prefill
graphs. Dense MXFP6 retains its previous compilation policy. Explicit
`--compilation-config` arguments still override the launcher's configuration.
Other launcher choices, including batch limits and Triton attention, remain
the same; this change restores the default **compilation policy**, not every
vLLM CLI default.

Eligible Qwen3.5-4B TP1 MLPs now fuse SiLU, multiplication, E8M0 scale generation,
and FP8 conversion in a Triton kernel. This removes 32 activation kernels and
their BF16 intermediate tensors per step. The native and FlashInfer GEMM
routes consume the same packed output. Set `VLLM_MACH_MXFP8_FUSED_MLP=0` to
disable it. Kernel specialization depends on K and tile size, not the prefill
token count, avoiding a fresh JIT specialization for each new batch size.

The earlier CUDA SwiGLU producer removed launches but did not improve compiled
model time. A hot CUDA graph microbenchmark on GPU7 measured 2.19/2.37 us for
that producer at M1/M16, versus 0.92/1.03 us for the Triton producer. Full-model
profiling showed a smaller improvement: BS16 model-forward time fell from
5.165 to 5.116 ms with the MLP change alone.

GDN's previous Python metadata branch was evaluated during Dynamo tracing,
where decode metadata was absent, so the compiled graph kept the stock path.
The optional `VLLM_MACH_GDN_TP1=1` path now selects persistent/overlap at runtime
inside `mach_gdn_forward`. The op is a piecewise graph boundary, preserving
uncaptured prefill state updates. Auxiliary CUDA streams remain in a process
registry; only their device indices are stored on the model, allowing AOT
serialization. Both fresh compilation and AOT cache loading were exercised.

With both changes enabled, profiles contained 24 persistent GDN kernels per
BS1 step and 32 fused SiLU/quantization kernels per step at both batches.
Native workspace fallbacks remained zero. The 20-step profiles used 3000 input
tokens, GPU6 for BS1 and GPU4 for BS16:

| Batch | Default compilation, before these changes | MLP + GDN TP1 | Kernels/step |
|---|---:|---:|---:|
| 1 | 3.398 ms | 3.261 ms | 530 → 450 |
| 16 | 5.165 ms | 5.021 ms | 626 → 594 |

Times above are median GPU model-forward spans, excluding lm-head. Each step
uses the union of its intervals across streams. The initial 4.835 ms BS16
estimate incorrectly mixed main/auxiliary interval lengths and is superseded
by 5.021 ms. Kernel-duration sums count concurrent work twice and are not an
elapsed-time metric for the overlap path.

Matched serving runs on GPU4 used 3000 input / 1000 output tokens, with 64
requests at BS16 and eight requests at BS1. Each arm used vLLM's default
compilation, with a warmup cohort equal to the concurrency:

| Batch | Median ITL before → after | Mean ITL before → after | Output tokens/s before → after |
|---|---:|---:|---:|
| 1 | 4.206 → 4.073 ms | 4.242 → 4.072 ms | 234.4 → 241.9 |
| 16 | 6.031 → 5.906 ms | 6.665 → 6.528 ms | 2294.7 → 2337.2 |

Median ITL improved by 3.2%/2.1% and throughput by 3.2%/1.8% at BS1/BS16.
These are single matched runs; small changes should not be treated as a
guarantee across workloads. The [measurement summary](data/mxfp8-compiled-20260930.json)
includes the serving metrics, profiler samples, fidelity, and workspace counters.

The MLP fusion is enabled by default on the supported geometry. To also enable
the measured GDN TP1 optimization, prefix the existing launch command with
`VLLM_MACH_GDN_TP1=1`. The launcher now uses default compilation without requiring
`--compilation-config '{}'` explicitly. Native MXFP8 remains the GEMM default.

Frozen256 fidelity used 256 queries and 10,479 teacher-forced target tokens at
each physical batch size. Mean per-query absolute gold-token logprob error
against matched BF16 references was:

| Physical batch | Before | MLP + GDN TP1 | Paired difference, 95% interval |
|---|---:|---:|---:|
| 32 | 0.058143 | 0.060281 | +0.002137, [-0.000396, 0.004796] |
| 4 | 0.060708 | 0.060059 | -0.000649, [-0.003410, 0.002109] |

Both intervals include zero and repeated cohorts matched exactly. These
measurements do not establish unchanged generation or lossless arithmetic.
The fused producer preserves BF16 rounding after SiLU and multiplication;
the default compiler's fused expression may round differently.

Validation covered 134 distinct tests, including compiled changed-input/zero-input
CUDA graph replay, native/FlashInfer producer outputs, live GDN context changes,
TP1/TP2 GDN state updates, cache factors, and deployment guards. Use
`tools/fidelity_native_mxfp6.py --default-compilation` for this fidelity policy.
Scripts and raw traces are in `audit/mxfp8-compiled/`.

#### AOT cache handling

Mach's graph-changing MXFP8/GDN flags and a hash of its Python/CUDA integration
sources now participate in vLLM's compilation cache key. This prevents an old
AOT graph from silently masking a changed optimization. The previous build
could reuse such a graph when only these flags or patched forwards changed.

To bypass the cache for a diagnostic launch, set
`VLLM_DISABLE_COMPILE_CACHE=1`; compilation still runs. To remove the default
cache, stop services that share it and delete
`${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/torch_compile_cache`, which also contains
the `torch_aot_compile` directory. No shared cache was deleted during this work.

### TP1 MXFP8 producer/consumer PDL

`VLLM_MACH_MXFP8_PDL=1` is an opt-in decode experiment for physical M1–32.
It requires native PDL version 3 and both rebuilt libraries:

```bash
cmake --build /data/lxy/detailed_benchmark/mxfp6_sm120/build/mxfp8 \
  --target mxfp6_torch mxfp8_torch --parallel 8
export MXFP6_LIBRARY_PATH=/data/lxy/detailed_benchmark/mxfp6_sm120/build/mxfp8/mxfp6_torch.so
export MXFP8_LIBRARY_PATH=/data/lxy/detailed_benchmark/mxfp6_sm120/build/mxfp8/mxfp8_torch.so
export VLLM_MACH_MXFP8_PDL=1
```

The chain covers GemmaRMSNorm, native activation quantization, fused
SwiGLU/quantization, optional residual/norm/quantization, CUTLASS dense GEMMs,
and the cache-hint, occupancy, irregular and TMA256 GEMMs selected by this
model at M1/16/32. Consumer kernels wait before reading dependent activations;
producer triggers allow independent successor setup to overlap. The native
quantizer overlaps output-scale padding before its input wait, and norm
producers may load immutable weights before waiting. M>32 retains ordinary
launches. The experimental norm/quant fusion also requires
`VLLM_MACH_MXFP8_NORM_QUANT=1`; it can change BF16 rounding.

MXFP8 CUTLASS builds need `CUTLASS_ENABLE_GDC_FOR_SM100`. Dense cooperative
and ping-pong templates inherit SM90 kernel bodies that excluded SM120
producer triggers. The patch adds SM120 triggers after MMA without using
`is_last_tile`, which its SM100 scheduler does not provide. Consumers still
wait for the entire predecessor grid, including remaining tiles, split
reductions and epilogue stores. The runtime version check rejects earlier
libraries that accepted a PDL launch argument without completing this chain.

Attention, GDN, BF16 BA projection and the LM head remain ordinary launch
boundaries. They do not need PDL for local chains to benefit, but they limit
how far overlap can extend. Their PDL integration is deferred.

The fixed 3000-input/1000-output TP1 report is
[available here](mxfp8-tp1-3000-1000-20260930.html). The complete v3 comparison
uses the same rebuilt libraries in every arm, three scored runs after warmup,
and GDN TP1 disabled. Reductions in full-request HTTP TPOT versus that baseline:

| Configuration | BS1 | BS16 | BS32 |
|---|---:|---:|---:|
| PDL | 3.10% | 1.94% | 1.21% |
| Residual/norm/quant fusion | 1.02% | 0.53% | 0.38% |
| Residual/norm/quant fusion + PDL | 3.76% | 2.30% | 1.42% |
| Existing NVFP4 head + PDL | 16.40% | 9.70% | 6.87% |

The [v3 raw results](data/mxfp8-tp1-pdl-complete-20260930.json) show identical
generated text for all 147 PDL-only requests versus baseline. Norm/quant fusion
and NVFP4 candidate search can change generated text; both remain opt-in.
These are incremental comparisons against the compiled MXFP8 baseline,
not stock FP8 speedups, and three repeats do not establish long-term stability.
The historical v1 PDL measurements lacked the complete native GDC path and
cannot establish the benefit of v3. Original libraries and newer build patches
are preserved under `audit/`.

### FP8 KV with NVFP4 head and PDL (October 8)

Qwen3.5-4B-MXFP8 TP1 was measured on RTX 5090 GPU4 with native MXFP8,
Triton attention, fused Gemma norms and SwiGLU, and NVFP4 candidate search
with BF16 refinement plus PDL v3. GDN TP1 and norm/quant fusion were disabled.
The comparison changes `--kv-cache-dtype auto` (BF16) to
`--kv-cache-dtype fp8_e4m3`; Mamba cache options retain their defaults.

Each batch uses the first BS frozen ShareGPT prompts repeated to 3000 tokens,
greedy sampling with `ignore_eos=true`, and exactly 1000 output tokens.
After one full warmup cohort, three unprofiled cohorts are scored. Both arms
were freshly measured sequentially on GPU4 with default compilation and
CUDA graphs; these gains do not use the earlier GPU2 measurements as baseline.

| KV cache | BS16 output tokens/s | BS32 output tokens/s | BS16 mean TPOT | BS32 mean TPOT |
|---|---:|---:|---:|---:|
| BF16 | 2518.6 | 3400.8 | 5.762 ms | 8.257 ms |
| FP8 E4M3 | 2775.1 | 3867.4 | 5.187 ms | 7.150 ms |
| Improvement | +10.18% | +13.72% | -9.98% | -13.40% |

All 288 scored requests completed the fixed-token contract, with no preemption
messages or errors before measurement completion. Generated text matched
between KV configurations for 12/48 BS16 requests and 39/96 BS32 requests;
within each arm all repeat comparisons matched. The checkpoint provides no KV
calibration scheme, and this experiment did not measure numerical fidelity or
task accuracy. Clocks were not locked, and three repeats in one server lifecycle
per arm do not establish long-term stability.

The [result data](data/mxfp8-fp8-kv-20261008.json) includes per-repeat throughput,
TPOT, token checks, library hashes and settings. Raw receipts are retained in
`audit/mxfp8-fp8-kv-20261008/`. Reproduce the FP8 arm with:

```bash
python tools/profile_mxfp8_breakdown.py \
  --output audit/mxfp8-fp8-kv-repeat \
  --arms nvfp4_head_pdl --device 4 --port 8263 \
  --batches 16 32 --repeats 3 --output-tokens 1000 --warmup-tokens 1000 \
  --kv-cache-dtype fp8_e4m3 --serving-only
```

Use both native PDL v3 libraries described above. Run a separate output
directory with `--kv-cache-dtype auto` for a fresh BF16 comparison.
