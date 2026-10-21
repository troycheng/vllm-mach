# Stochastic LM-head sampling

Native MXFP6 and both MXFP8 launchers enable hybrid LM-head by default under
one shared `--hybrid-head` switch. This extends Mach's greedy paths with vocabulary-parallel
top-k/top-p and presence-penalty sampling operations from `vllm-shpgy` revision
`30ffd6f841c60afc346ad75578c9cc8380225ad9`. It leaves the original logits processor,
BF16 head parameter and ordinary sampler in place. No additional vLLM source
patch is required beyond the selected Mach profile's existing installer.

The dispatch policy was updated after the measurements below: unfiltered and
top-p-only requests now always use the original sampler, regardless of the
switch. Greedy and bounded stochastic hybrid paths now share the same switch;
the sampling-only switch used for the historical ablation is no longer a
separate public option. Historical Gumbel measurements remain as evidence for excluding that
path; they do not describe current production dispatch. The measurements below
predate this admission/default change. A later
[same-process admission check](best-path-recheck-20261015.md#same-process-check-of-unchanged-paths)
did not reproduce the earlier unchanged-path slowdown at c4; it does not
establish absence of regressions for every configuration.

## Usage and admission

For native MXFP6 on the supported SM120 runtime:

```bash
vllm-mach-serve --model /models/MODEL
# Explicitly disable greedy and stochastic hybrid paths:
vllm-mach-serve --model /models/MODEL --no-hybrid-head
```

The shared switch controls compact greedy dispatch and bounded stochastic
dispatch together. `--no-hybrid-head` disables these compact paths and retains
each profile's original head/sampler. It does not change recurrent-state precision.
`--nvfp4-lm-head` and `--lm-head-sampling`, including their negative forms,
remain aliases of the same switch; when multiple aliases are supplied, the
last occurrence wins. There is no separate sampling-only production switch.

The 4B and 2B MXFP8 launchers use the same default and aliases. The 4B K2048
full-vocabulary replacement head is independent of this switch and remains
installed when compact hybrid is off. Its packed weights are also reused by
the compact paths when enabled. Native MXFP6 and 2B prepare their NVFP4 sidecar
only when the switch is enabled.
These model-specific profiles retain their existing installation/checkpoint
requirements. Enabling this flag does not widen their compatibility contract.

Use the selected Mach launcher to configure the shared switch. It derives the
profile-specific greedy-head and sampling environment settings together;
`VLLM_MACH_LM_HEAD_SAMPLING` in receipts is internal wiring, not a separate
production control. Diagnostic ablation tools may override it explicitly.
The block-FP8 launcher retains its original sampler and resets this flag to 0.
The plugin installs an idempotent V2 sampling hook, independently of the generic
compact-head flags, which remain disabled for the MXFP8 profiles.

Eligible calls are non-speculative decode batches whose active requests have
identical positive temperature, top-k and top-p. The supported mode is:

* `1 <= top_k <= 64`, optional top-p, and optional presence-only penalties.
  Local top-k scores/token IDs are gathered, globally filtered and sampled on
  TP rank 0, then the sampled IDs are broadcast. NVFP4 proposes candidates and
  BF16 weights refine them when its geometry/row guards admit the call.

Explicit request seeds, logprobs/token-score requests, logit bias, allowed-token
lists, minimum-token constraints, bad words, thinking budgets, repetition or
frequency penalties, min-p, grammar, LoRA, speculative decode, batch sharding,
prefill/mixed batches, diagnostics and heterogeneous sampling parameters retain
the previous path. Unfiltered sampling and top-p without a supported bounded
top-k always retain the previous path, even with the switch enabled.
The ordinary head/sampler remain callable with their original signatures.

## Optional NVFP4 head CUDA Graphs

Set `VLLM_HYBRID_NVFP4_LM_HEAD_CUDAGRAPH=1` before starting a supported
2B or 4B compact-head profile to capture independent head stages during
startup. The default is `0`; setting it to `0` keeps eager head execution.
This switch does not enable the hybrid head itself or broaden model admission.
The native MXFP6 startup path and the 4B K2048 full-vocabulary head are not
changed by this capture integration.

The TP1 compact sidecar captures row counts 1 through 64, bounded by its
configured limit. Captured stages cover greedy token selection, bounded
candidate refinement for stochastic top-k, and coarse logits used before
presence penalties. Penalties and final stochastic sampling retain their
existing order and admission rules. Both compact greedy and stochastic
paths share the prepared sidecar.

Replay requires matching weights, candidate configuration, shape, device and
execution stream. Unsupported calls retain eager execution. Returned outputs
own their storage, so subsequent replays cannot overwrite them. Graphs use a
shared memory pool; startup time and persistent GPU memory increase. Capture
occurs after existing warmup and does not change the selected GEMM tactic.

## Numerical contract

Candidate search is approximate: BF16 refinement cannot recover tokens omitted
by NVFP4. Positive or negative presence penalties are applied before candidate
selection and again to the refined scores. Candidate width follows the existing
Mach rule: K128 for top-k=20 and K512 for top-k=64. Native launcher NVFP4 admits
up to 32 rows; the MXFP8 sidecars admit up to 64. Larger calls use complete local
projection before compact sampling.

The retained diagnostic full-distribution Gumbel utility preserves softmax
sampling in distribution, but is not reachable from serving dispatch.
Explicitly seeded requests deliberately use the original sampler.
The 4B MXFP8 profile's complete head remains its existing NVFP4/K2048 score
replacement path; fallback in that profile does not imply a pure BF16 projection.

## Validation

Run the regression suite on an idle CUDA device:

```bash
python -m pytest -q tests/native_mxfp6/test_sampling.py
```

The subsequent unified-switch/admission regression run passed 173 tests and
52 subtests across sampling, CLI aliases, launchers, installation, deployment, original heads and compile
cache contracts. Three CUDA-only kernel tests were skipped in that CPU-only
run; no model service was started. The added checks exercise enabled-hook
fallback for unfiltered/top-p-only sampling, explicit opt-out, all three
launcher defaults, positive/negative presence-penalty top-k refinement, and
disabled 4B compact greedy installation while preserving the K2048 full head.

`tools/compare_lm_head_sampling.py` runs baseline/new/new/baseline service
lifecycles with the unified hybrid switch off/on. Both arms keep the same
checkpoint, TP2, FP32 SSM, graph sizes, KV pool, request tokens and arrival schedule.
Unlike the historical sampling-only ablation below, off now also disables
compact greedy hybrid. The 4B K2048 full head remains enabled in both arms.
Each sampling/concurrency case has its own
warmup. Stochastic requests intentionally omit per-request seeds; the benchmark
records this in the workload contract. Timing contains no precision observer.

```bash
python tools/compare_lm_head_sampling.py --model /models/MODEL \
  --devices 0,1 --output /results/head-serving --rounds 2
```

For the dedicated MXFP8 profiles, use `--profile mxfp8-2b` or
`--profile mxfp8-4b` and one device. The comparison uses the corresponding
production launcher without changing its capacity, cache dtypes or graph
policy. The 2B comparison retains `--attention-backend TRITON_ATTN` and
`--hybrid-head` in the enabled arm. For example:

```bash
python tools/compare_lm_head_sampling.py --profile mxfp8-2b \
  --model /models/PREPARED-2B --devices 0 --concurrency 4 32 64 \
  --output /results/2b-head-serving --rounds 2
```

`tools/verify_lm_head_sampling.py` uses the frozen 256-query fidelity manifest.
At each scored decode state it evaluates both the proposed distribution and a
complete BF16 projection on the identical hidden states. The reference uses
public vLLM PyTorch top-k/top-p filtering, including cutoff ties. It reports global
candidate overlap, missing reference probability mass and total variation,
including presence penalties, at physical M4 and M32. Scheduling is paused
until each complete cohort is enqueued, and every scored call asserts its
physical row count. Generated token identity
is not a correctness metric for independent stochastic draws. Its observer is
diagnostic-only and must never be installed in a throughput service.

```bash
python tools/verify_lm_head_sampling.py --model /models/MODEL \
  --rows 4 32 --output /results/head-precision.json
```

MXFP8 diagnostics use each profile's accepted quality geometry and exact
checkpoint/runtime validation. Run each physical batch in a separate process:

```bash
python tools/verify_lm_head_sampling.py --profile mxfp8-4b \
  --model /models/PREPARED-4B --rows 32 --output /results/4b-m32.json
python tools/verify_lm_head_sampling.py --profile mxfp8-4b \
  --model /models/PREPARED-4B --rows 64 --output /results/4b-m64.json
```

Use `--profile mxfp8-2b` for the 2B equivalent. In addition to the complete
BF16 reference, `mean_baseline_total_variation` and
`max_baseline_total_variation` compare against the original serving head:
BF16 for 2B, K2048 for 4B. This separates the new sampling path's incremental
effect from the 4B baseline's existing approximation. Explicit-seed and
logprobs smoke checks compare the actual old/new outputs before installing
the diagnostic observer.

Total variation is `0.5 * sum(abs(p_new - p_reference))`, with zero meaning
identical conditional sampling probabilities. Missing reference mass measures
reference probability assigned to IDs outside the proposed compact set. Top-k
ID overlap can change at equal-score ties and is reported separately. These
are head-distribution metrics, not answer accuracy or BF16-model fidelity.

Kernel distribution checks and equal-seed shard/unsharded Gumbel comparisons
are separate from model numerical measurements. Neither establishes broad
application accuracy or support for untested hardware/model combinations.

## Dedicated 2B/4B MXFP8 head fidelity (2026-10-15)

Qwen3.5-2B and Qwen3.5-4B used their exact packaged champion checkpoints,
complete tensor/file verification and installed runtime-source hashes. Each
ran TP1 on an RTX 5090 with official vLLM 0.29.0, Torch 2.13.0, Triton 3.7.1,
FlashInfer Python/cubin 0.6.18, CUTLASS DSL 4.6.2, cuda-bindings 13.3.1,
standalone b12x 1.2.6 and mxfp6-sm120 0.2.1. No runtime or checkpoint identity
checks were bypassed.

Both models retain BF16 activations and FP32 SSM. 2B uses TRITON_ATTN, BF16
KV and its K128 hybrid sidecar. 4B uses FLASHINFER, model-bound FP8 E4M3 KV
scales and the original K2048 serving head. Each quality geometry ran in its
own process: accepted FULL_AND_PIECEWISE M4/M32 captures with M32/4 GiB KV;
M4/M32/M64 captures with M64/4 GiB KV for 2B and M64/19 GiB KV for 4B. The
scheduled-token budgets are 8,192 and 16,384 respectively. These are precision
configurations, not production throughput configurations.

For each model, six M32/M64 × top-k20/top-k20+presence1.5/top-k64 cases
completed 256 frozen real-text prompts and 128 output tokens each. All use
temperature 0.7, top-p0.95, no request seed and ignored EOS. Each case verified
32,512 actual NVFP4 decode positions at its asserted physical batch size:
195,072 positions per model, 390,144 total. First-token prefill is excluded.
The full BF16 reference shares each new path's actual hidden state, not an
independently generated sequence or an unquantized backbone.

| Model | Physical rows | top-k | Presence | Mean TV vs original head | Max TV vs original head | Mean TV vs BF16 head | Max TV vs BF16 head | Missing BF16 support rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B | 32 | 20 | 0 | 0.024555% | 6.2597% | 0.024555% | 6.2597% | 4 / 32512 |
| 2B | 32 | 20 | 1.5 | 0.027997% | 7.6233% | 0.027997% | 7.6233% | 11 / 32512 |
| 2B | 32 | 64 | 0 | 0.027124% | 4.6070% | 0.027124% | 4.6070% | 0 / 32512 |
| 2B | 64 | 20 | 0 | 0.024574% | 5.4710% | 0.024574% | 5.4710% | 8 / 32512 |
| 2B | 64 | 20 | 1.5 | 0.029139% | 6.9080% | 0.029139% | 6.9080% | 6 / 32512 |
| 2B | 64 | 64 | 0 | 0.026859% | 5.3243% | 0.026859% | 5.3243% | 0 / 32512 |
| 4B | 32 | 20 | 0 | 0.015667% | 6.9657% | 0.016314% | 6.9657% | 5 / 32512 |
| 4B | 32 | 20 | 1.5 | 0.014909% | 4.1809% | 0.016288% | 5.1895% | 2 / 32512 |
| 4B | 32 | 64 | 0 | 0.022038% | 4.9748% | 0.023163% | 4.9748% | 0 / 32512 |
| 4B | 64 | 20 | 0 | 0.015369% | 4.9541% | 0.016565% | 4.9541% | 4 / 32512 |
| 4B | 64 | 20 | 1.5 | 0.016791% | 4.9541% | 0.018286% | 4.9541% | 3 / 32512 |
| 4B | 64 | 64 | 0 | 0.020365% | 4.5838% | 0.022117% | 5.1399% | 0 / 32512 |

Original-head means BF16 for 2B and the existing K2048 approximation for 4B.
Each of the four quality processes also passed exact switch-off/on token
comparisons for explicitly seeded top-k20 requests and exact token/logprob
comparisons for greedy logprobs requests (four prompts × 16 tokens per arm).
All 27 sampling regression tests passed under these fixed MXFP8 dependencies;
the original 2B/4B head CPU suites passed eight tests and 12 subtests with the
profiles' required stock-hybrid modes disabled.

Small average TV does not establish lossless sampling or business answer
accuracy. Rare omitted reference support and pointwise changes remain, and
the current default enables only admitted bounded top-k requests. The historical
unfiltered experiment retained each profile's original complete head; its
Gumbel utility is tested separately against the softmax distribution and does
not use the K128 candidate truncation. Serving no longer dispatches to it.


## Dedicated 2B/4B MXFP8 serving (2026-10-15)

Each model used its dedicated production launcher on one RTX 5090, TP1,
with the same verified checkpoints and pinned MXFP8 dependencies as the
quality runs above. 2B used TRITON_ATTN, BF16 KV, FP32 SSM, maxseq160,
maxlen16384, maxbatch2048, a 19 GiB KV pool and the hybrid head in both arms.
4B used FLASHINFER, checkpoint-bound FP8 E4M3 KV, FP32 SSM, maxseq128,
maxlen8192, maxbatch2048, a 19 GiB KV pool and its original K2048 full head
in both arms. Both retained their complete production FULL_AND_PIECEWISE
graph policies, including exact PW2048. The original sampler used its
standard V2 Triton Gumbel path (`VLLM_USE_FLASHINFER_SAMPLER=0`) in both arms. Only
`--lm-head-sampling` changed. No precision observer ran in timing services;
each model's four lifecycles used the same GPU.

Workload: 1,024 frozen uniform input-token IDs, 256 output tokens,
temperature0.7, ignored EOS and no request seeds. Top-k cases use top-p0.95;
the presence case adds penalty1.5. At each concurrency c=4/32/64 and each
sampling case, c requests × 32 tokens warm up separately before scoring
2c requests × 256 tokens. Baseline/new/new/baseline lifecycles completed all
3,200 scored requests and 819,200 output tokens per model, 6,400 requests
and 1,638,400 output tokens combined. All optimized lifecycles logged the
actual NVFP4 K128/M64 setup and stochastic dispatch. Top-k64 automatically
widens transient candidates under the shared candidate-width rule.

The table uses arithmetic means of two lifecycles per arm. These are short
paired head-ablation samples, not confidence intervals or replacements for
the published 3k/1k full-profile curves. Low-concurrency timings are more
variable. Throughput includes the complete HTTP request and model service.

| Model | Sampling | Concurrency | Baseline tok/s | New tok/s | Change | Baseline / new mean TPOT |
|---|---|---:|---:|---:|---:|---:|
| 2B | top-k20 / top-p0.95 | 4 | 1170.00 | 1375.74 | +17.58% | 3.080 / 2.561 ms |
| 2B | top-k20 / top-p0.95 | 32 | 6435.03 | 7042.53 | +9.44% | 4.289 / 3.863 ms |
| 2B | top-k20 / top-p0.95 | 64 | 9102.74 | 9729.79 | +6.89% | 5.910 / 5.448 ms |
| 2B | top-k20 + presence1.5 | 4 | 1161.18 | 1351.60 | +16.40% | 3.114 / 2.618 ms |
| 2B | top-k20 + presence1.5 | 32 | 6563.66 | 6828.13 | +4.03% | 4.196 / 4.013 ms |
| 2B | top-k20 + presence1.5 | 64 | 9005.51 | 9513.98 | +5.65% | 5.938 / 5.534 ms |
| 2B | top-k64 / top-p0.95 | 4 | 1178.31 | 1410.36 | +19.69% | 3.059 / 2.490 ms |
| 2B | top-k64 / top-p0.95 | 32 | 6573.61 | 7071.45 | +7.57% | 4.163 / 3.846 ms |
| 2B | top-k64 / top-p0.95 | 64 | 9086.94 | 9794.75 | +7.79% | 5.899 / 5.404 ms |
| 2B | Unfiltered (extension now disabled) | 4 | 1360.21 | 1384.27 | +1.77% | 2.640 / 2.617 ms |
| 2B | Unfiltered (extension now disabled) | 32 | 6763.97 | 6156.78 | -8.98% | 4.059 / 4.528 ms |
| 2B | Unfiltered (extension now disabled) | 64 | 9372.99 | 8201.95 | -12.49% | 5.673 / 6.713 ms |
| 4B | top-k20 / top-p0.95 | 4 | 785.93 | 872.67 | +11.04% | 4.608 / 4.107 ms |
| 4B | top-k20 / top-p0.95 | 32 | 3652.59 | 3815.32 | +4.46% | 7.547 / 7.173 ms |
| 4B | top-k20 / top-p0.95 | 64 | 4769.71 | 4943.32 | +3.64% | 11.368 / 10.908 ms |
| 4B | top-k20 + presence1.5 | 4 | 787.78 | 869.90 | +10.42% | 4.607 / 4.120 ms |
| 4B | top-k20 + presence1.5 | 32 | 3631.14 | 3798.92 | +4.62% | 7.583 / 7.209 ms |
| 4B | top-k20 + presence1.5 | 64 | 4712.30 | 4891.33 | +3.80% | 11.507 / 11.003 ms |
| 4B | top-k64 / top-p0.95 | 4 | 788.39 | 866.64 | +9.93% | 4.601 / 4.132 ms |
| 4B | top-k64 / top-p0.95 | 32 | 3635.74 | 3777.81 | +3.91% | 7.571 / 7.246 ms |
| 4B | top-k64 / top-p0.95 | 64 | 4700.47 | 4883.52 | +3.89% | 11.512 / 11.031 ms |
| 4B | Unfiltered (extension now disabled) | 4 | 865.01 | 852.17 | -1.48% | 4.157 / 4.222 ms |
| 4B | Unfiltered (extension now disabled) | 32 | 3703.90 | 3491.43 | -5.74% | 7.416 / 7.940 ms |
| 4B | Unfiltered (extension now disabled) | 64 | 4753.66 | 4423.39 | -6.95% | 11.375 / 12.376 ms |

Top-k gains do not imply gains for every sampler mode. The unfiltered TP1
Gumbel path regresses at larger concurrency on both models in this historical
run; the extension is now disabled for Unfiltered requests. These rows explain
that decision and do not measure current default dispatch. The historical path
retained the original complete head and added no candidate truncation, but its
execution cost was different. Serving now always excludes unfiltered requests
from the extension, even when bounded top-k optimization is enabled. These data do not validate MoE,
other GPU architectures, different quantized checkpoints or business answer
accuracy. Precision limits remain those reported above.


## NVFP4 head CUDA Graph comparison (2026-10-21)

Both arms enable the same Mach K128 greedy hybrid head; only
`VLLM_HYBRID_NVFP4_LM_HEAD_CUDAGRAPH` changes. A test-only adapter attaches
that existing head to the multimodal NVFP4 model. This does not add an NVFP4
production launcher or widen the model-specific launchers' admission.

The same Qwen3.5-based 2B NVFP4 checkpoint runs on one RTX 5090 with TP1,
BF16 SSM, FP8 KV, prefix caching disabled, 19 GiB KV allocation, maximum
sequence length 16384, 160 sequence slots and a 2048-token scheduling budget.
Both arms retain the same GEMM backend/tactic policy and decoder CUDA Graph
configuration. Startup compilation and head capture are excluded from timing.
Four fresh engines run graph off → on → on → off. Values below are arithmetic
means of two runs, not confidence intervals. All 3840 business requests and
4160 fixed-length requests succeed.

Fixed-length requests use 3000 input and 1000 output tokens with EOS ignored:

| Concurrency | Graph off tokens/s | Graph on tokens/s | Gain |
|---:|---:|---:|---:|
| 4 | 1546.54 | 1732.62 | +12.03% |
| 16 | 4957.98 | 5479.76 | +10.52% |
| 24 | 6615.39 | 7729.36 | +16.84% |
| 32 | 7940.11 | 9008.11 | +13.45% |
| 48 | 9707.45 | 11002.07 | +13.34% |
| 64 | 10972.97 | 12192.87 | +11.12% |

Business testing uses 80 records from each Chinese/math/English text/image
group, five warmup requests per group/concurrency, temperature 0, thinking
disabled and a 2048-token output cap. Requests are non-streaming; latency is
full HTTP response time, not TTFT. Aggregate QPS is total successful requests
divided by total group wall time within each run, then averaged across runs.

| Concurrency | Graph off QPS | Graph on QPS | Gain |
|---:|---:|---:|---:|
| 4 | 24.62 | 26.39 | +7.20% |
| 16 | 51.67 | 54.99 | +6.43% |

Both graph-enabled runs record more than 47000 greedy replays. Head-level
checks using the actual checkpoint weights produce identical eager/graph
greedy tokens at rows 1, 4, 16, 24, 32, 48 and 64. Separate GPU regression
checks verify candidate scores by token ID, returned-output ownership and
fallback guards. Unsorted candidate order is not required to match.
These checks do not establish full-model or labeled task accuracy. Stochastic
top-k and presence-penalty paths have regression coverage; their serving
performance and other model sizes are not measured by this comparison.

[Machine-readable results](data/nvfp4-head-cudagraph-20261021.json) include
individual runs, business groups and the test configuration. The graph switch
remains opt-in; these measurements do not guarantee a benefit for every model
or workload.
