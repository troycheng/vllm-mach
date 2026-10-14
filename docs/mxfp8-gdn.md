# Ordered W4 GDN profile

`vllm_mach.mxfp8.gdn` packages the ordered-state part of the Qwen3.5-4B MXFP8 champion. It runs inside vLLM's existing GDN opaque operation. Installation, native MXFP8, FP8 KV calibration, graph policy, the projection kernels and the full-vocabulary head must be composed by the complete model profile. This module alone is not a champion deployment or a performance qualification.

The serving boundary is vLLM 0.29.0, one SM120 GPU, TP1, BF16 model and convolution state, FP32 recurrent state by default, `max_num_seqs=128`, `max_model_len=8192`, and 19 GiB KV memory. BF16 and `fp8_e4m3` attention KV configurations are accepted; the latter still requires the complete profile to install its calibrated scales before profiling/capture. Prefix caching, speculative decoding/MTP, ReplaySSM, connectors, cache offload and CPU weight offload are rejected. Source hashes pin vLLM's GDN implementation, packed FLA implementation and GPU runner, independent of their install location.

| Actual decode rows | Route |
| --- | --- |
| 1, 2, 8 | Ordered full-store, age zero |
| 4, 16 | Ordered W4 deferred, dedicated kernels |
| 24 through 128, step 8 | Ordered W4 deferred |
| Other shapes, mixed batches, prefill | Materialize relevant slots, then stock vLLM |

All routes use caller-owned FP32 pending K, pending delta, raw alpha, compatibility prefix, and int32 tile age. The eight unique state views retain their own scratch through graph capture and replay; 24 layers keep their original association with those views. A positive slot may appear once per kernel call; repeated zero padding is harmless. The current convolution call, FP operations, reduction expressions, replay FMA operand order, W4 flush rule, warp count, stage count and launch grids are preserved.

## Integration API and installation order

Parent and spawned worker processes must both select the same complete profile. In each process, register native MXFP8 hooks first, then call `gdn.install_worker_hook()` before the Worker lifecycle starts. This call installs Python wrappers only; package import and hook registration do not load torch, Triton, native libraries, or query CUDA. `install_worker_hook()` returns `True` on the first installation and `False` on repeat registration.

The wrappers execute the following sequence:

1. Original `Worker.init_device`, including native backend installation; then `install_runtime(worker)` registers GDN and scheduler lifecycle interception.
2. Original `Worker.load_model`; then `prepare_layers(worker)` checks native policy `all`, 24 unique GDN layers, and the layer geometry.
3. Original `Worker.initialize_from_config`; then `initialize_cache(worker)` checks the actual FP32/BF16 pool binding without allocating scratch.
4. `prepare_pools(worker)` allocates scratch, checks eight state views and available memory, initializes metadata, and warms every supported decode and fallback materialization signature; then original `compile_or_warm_up_model` runs.
5. `inspect_worker(worker)` returns a CPU receipt after compilation. It reports capture construction separately from eager dispatch; Python capture counts are not replay counts.

A complete profile may use these individual lifecycle functions instead of the wrapper. It must preserve their sequence, and must not call both installation styles for one Worker. Pool initialization warms row stride 12288 and uses zero slot indices to avoid updating live base state. Scratch allocation requires its computed size plus a 512 MiB reserve. The implementation does not reduce the configured KV budget to make the allocation fit.

One process owns one Worker and one cache binding. Repeated model/cache preparation, swapping the owner, cache reallocation after capture, or legacy competing GDN hooks are rejected. Complete profile startup should also reject sleep/offload or weight-update paths that it has not qualified; this module has no state-transfer or cache-rebinding protocol for them.

## Allocation, fallback and explicit lifecycle operations

The runner's original new/cached request updates run before metadata resets. Newly assigned positive IDs invalidate pending terms while leaving base-state zeroing and convolution-state initialization with stock vLLM. New requests with retained computed context and KV block-copy requests are rejected. Slot reuse follows the same allocation reset path. On prefill or unsupported decode, raw pending factors are applied on the current stream before stock reads dense state in the configured dtype; materialization resets age and prefix before the stock operation.

`materialize_state(worker)` is a drain-boundary RPC that flushes all positive SSM slots and synchronizes. `reset_state(worker, zero_base=False)` discards pending terms; `zero_base=True` additionally zeros dense SSM state. These operations require ready pools, an empty runner request table, and no graph capture. They preserve scratch storage and captured pointer identity. They do not reset the convolution pool or the scheduler. Ordinary requests should use the scheduler hooks; explicit reset is only for an already drained lifecycle boundary.

Allocation IDs share the runner's global block-pool namespace across KV groups. Every group manager allocates from the same `BlockPool`, and a live block remains reference-counted outside its free queue. Flattening newly allocated IDs across all groups therefore cannot reset another live GDN slot under this no-prefix/no-copy contract. It also resets unused GDN metadata at attention-only IDs, as the original implementation did. The optional diagnostic Mamba-only filter is excluded to preserve that launch behavior.

## Verification

CPU verification:

```bash
python -m unittest discover -s tests -p 'test_mxfp8_gdn*.py' -v
```

The tests run without torch/vLLM and check import safety, production/FP8 KV guards, once-only wrapping, cache phase order, allocation validation, owner identity, fallback-before-stock, scheduler reset order, and AST fingerprints of kernel operations plus launch expressions. They establish packaging/source contracts, not GPU numerical or throughput equivalence.

A GPU qualification must use the installed profile in a fresh process. Compare output and materialized FP32 state against the frozen reference at W4 ages 0–3 for M1/2/4/8/16 and M24..128 step8. Then cover unsupported rows, actual PW2048 prefill, mixed prefill/decode, zero padding, new/cached assignments, slot reuse, request drain plus explicit materialize/reset, graph replay with changing indices, and process restart. Confirm 24 layers, eight state views, all supported capture signatures, unchanged 19 GiB capacity, no scratch reallocation, and the expected FP8 KV dtype/scales when enabled. Fixed-M gold and six-point end-to-end qualification belong to the complete profile.


The independent quality mode uses `VLLM_MACH_MXFP8_MODE=quality` and
`VLLM_MACH_MXFP8_QUALITY_ROWS=32` or `64`. It follows the accepted raw contracts:
M32 has maxseq32/maxlen1024/maxbatch8192, 4 GiB KV and FULL captures [4,32]; M64
has maxseq64/maxlen1024/maxbatch16384, 19 GiB KV and FULL captures [4,32,64].
Both have no PW2048 and retain FP32 recurrent state. As in the accepted worker,
the ordered row set is clipped by scheduler capacity: M32 includes
1,2,4,8,16,24,32; M64 adds 40,48,56,64. This preserves capture and tail routes;
unsupported counts still materialize before stock computation.

Worker initialization pins the mode, so an environment change cannot silently
switch an existing state pool. Production remains the default with
maxseq128/maxlen8192/19 GiB and the complete row set. Quality mode must not be
presented as a production throughput measurement. Its graph/projection checks
are managed by the complete profile. The formal runner source SHA is pinned
against the versioned MXFP8 runtime manifest; the historical frozen runner is
not an accepted installation requirement.

## BF16 ordered GDN default

Both launchers use ordered W4 GDN by default for BF16 and FP32 state.
Select BF16 storage with `--ssm-dtype bfloat16`; FP32 storage remains the default.
Use `--no-ordered-bf16-ssm` for the stock BF16 backend. Direct profile users
can set `VLLM_MACH_MXFP8_SSM_DTYPE=bfloat16`; ordered defaults to `1`,
and `VLLM_MACH_MXFP8_ORDERED_BF16_SSM=0` explicitly selects stock BF16.
The run contract records the selected state dtype and ordered backend.

Ordered replay rounds state to BF16 after every pending token and keeps
K/delta/alpha history in FP32. The small Gluon kernel preserves the stock
Q/K norm and state-dot reduction trees, with the existing BV32/W4 scratch.
Mixed decode, stock fallback, materialization and slot reset share that history.

## PDL + GDN: not fully tuned

**PDL + GDN is not fully tuned.** Both experimental switches default to `0`:

- `VLLM_MACH_MXFP8_BF16_GDN_GROUPED=1`: group four state rows per thread
  on deferred decode and extend Gluon to configured large shapes.
- `VLLM_MACH_MXFP8_BF16_GDN_PDL=1`: enable GDN dependent launch and wait
  before any input/state/history load.

Flags are frozen at module import before warmup/capture and accept only
`0` or `1`. They require the ordered BF16 serving option. With flags unset,
Gluon handles up to 1536 head-rows: full writeback/eight warps through 256,
deferred/four warps through 1024, then two warps. Larger shapes retain Triton,
which now also honors the PDL switch for BF16 state.
Grouped deferred decode uses four warps through 768 head-rows, then two.

[Consolidated measurements and validation](data/mxfp8-benchmark.json)
retain per-concurrency serving results, backend selections and unresolved
qualification findings. Full kernel/debug samples and removed prototype
sources remain in the audit archives referenced by the dataset.

### Throughput attribution and cache-capacity checks

The October 14 GPU 3 four-way precision comparison holds the native binary,
runtime hashes, request contracts, graph policy and 19 GiB cache budget fixed.
At c64, ordered BF16 SSM gains 8.27% with BF16 KV and 10.39% with FP8 KV.
FP8 KV gains 15.91% with FP32 SSM and 18.18% with BF16 SSM. These are
precision/configuration changes, whereas ordered versus stock BF16 changes
implementation at the same state precision. Their gain percentages should
not be conflated. Single full-curve trials do not establish confidence bounds.
For equal output counts, normalized c64 elapsed time is 1.00000 for
FP32/BF16, 0.92361 for BF16/BF16, 0.86272 for FP32/FP8, and 0.78152
for BF16/FP8. The combined time saving is close to the sum of the two
separate savings (a 0.48-percentage-point residual). This is consistent
with different contributions to total serving time; it is not a per-kernel
time attribution. The fixed-FP8 comparison exposes a larger relative SSM
gain after reducing attention/cache cost, and vice versa.

A KV-capacity admission limit is not supported by the c64 service evidence.
FP32 SSM/BF16 KV exposes 491,520 cached tokens, BF16 SSM/BF16 KV exposes
549,428, and the workload requires at most 256,000 live prompt-plus-output
tokens before page rounding. The startup estimate of 60 concurrent requests
for the former assumes 8,192 tokens per request; it is not a scheduler cap.
Both configurations reach 64 running requests. Within their scored c64
intervals, sampled cache occupancy peaks at 59.3% and 52.7%, respectively,
and neither log reports preemption/recomputation. After initial admission,
these periodic samples show 63–64 running requests and no waiting queue.
They measure scheduler occupancy, not each executed decode batch. Chunked
prefill, completion/admission transitions and the final drain still change
the actual per-step batch; graph padding must not be counted as active rows.

Both KV dtypes resolve BF16 prefill/decode queries and dedicated XQA decode
on SM120. The pinned runtime rejects TRTLLM prefill on SM120, selecting native
FlashInfer prefill. Thus the default precision comparison does not switch
between XQA and another decode backend, nor quantize Q. Different KV dtypes
still select different XQA specializations and change bandwidth/cache behavior.
A native-FlashInfer decode control is measured separately with
`--attention-config '{"use_trtllm_attention":false}'`; it also preserves BF16
queries and native prefill. At c64, the native control measures 4452.85
output tokens/s with BF16 KV and 5202.50 with FP8 KV, a 16.84% KV gain,
versus 18.18% under XQA. XQA raises absolute throughput by 2.11% for BF16
KV and 3.28% for FP8 KV relative to native decode; the KV-gain difference
is 1.35 percentage points. This supports a modest implementation effect,
not an automatic BF16-to-FP8 backend switch or an explanation of the entire
SSM-versus-KV gain difference. Both native controls reach 64 running requests
without logged preemption; sampled c64 cache peaks are 51.3% and 29.8%.
Exact output equality between attention implementations was not checked;
this control verifies successful fixed-length serving and throughput, not
cross-backend numerical equivalence. End-to-end throughput includes the cost of
3000-token chunked prefills, not solely the state-update or decode kernel.

### Serving comparisons

The high-row ordered kernel now replays only the pending W4 terms instead of
testing all four slots. On 4B/M64 BF16, the single-pool four-step CUDA graph
measurement fell from 28.45 to 26.93 microseconds per step on GPU 3. Exact
output and materialized-state comparison against stock passed for M24/32/48/64/128
with BF16 and FP32 state over 128 changing-input steps. The complete October 14 serving recheck
includes this kernel change;
its six-point curves are recorded in the serving benchmark report.

With PDL enabled and grouped layout disabled, the Triton high-row route passed
1024-step exact output and materialized-state checks at 4B/M48/64/96/128.
Its M64 four-step single-pool timing was 26.68 microseconds versus 26.93
without PDL. Grouped+PDL Gluon also passed 1024 steps at M24/32/48/64/128,
but the eight-pool graph timing differed little from the prior route. Neither
kernel result establishes an end-to-end serving improvement.

A preliminary October 14 4B pair on GPU 3 used the older native binary
`c4522aff` and the same FP8 KV model and
3000-input/1000-output request protocol at c32/c64. Grouped+PDL measured
4154.63/5408.91 output tokens/s; stock BF16 measured 4222.16/5408.00.
Grouped+PDL was 1.6% slower at c32 and effectively tied at c64. The local
layout sweep found no meaningful gain from state-row/warp settings 2/4,
8/2 or 1/2 versus the existing grouped 4/2 policy at M32/48/64. The later full six-point
recheck uses the main matrix native binary
`de8a02f4` (revision `ce0cd05`), and supersedes this preliminary pair.
Neither recheck justifies enabling grouped or PDL by default for 4B.

Nsight Compute on 4B/M64 BF16 measured about 68.2 MB of DRAM reads for one
ordered decode, approximately the dense state size for 64 rows. The grouped
Gluon kernel used 127 registers/thread; the high-M Triton kernel used 255.
The kernel is dominated by state traffic when that working set is cold.
Eight rotating state pools in the CUDA-graph benchmark exceed L2 capacity,
while the paired c32/c64 service runs include the actual 24-layer state,
attention KV, projections and cache contention. Two experimental kernels,
one processing BV64 together and another processing sequential BV16 halves,
passed exact output/state checks but lost 1-3% against the existing grouped
BV32 kernel with eight pools. A W2 flush increased M64 eight-pool time from
57.8 to 66.1 microseconds per step, and restricting Triton to 128 registers
caused heavy spilling. None was promoted to production dispatch.

An additional async global-to-shared state-loading prototype overlapped
`cp.async` with Q/K normalization and gate computation. It passed exact
128-step BF16 output/materialized-state checks at M32/M64 with PDL, but
retained 127 registers and regressed at one, eight and 24 pools. M64 times
were 25.99/59.75/60.04 microseconds versus 24.45/57.90/58.10 for existing
grouped decode. Delaying state conversion or loading did not improve cold
timing either. All three prototypes were removed. State loads already use
128-bit vector instructions; simply introducing shared memory did not reduce
the dominant DRAM traffic. The kernel cache matrix is recorded here;
the serving benchmark report
contains throughput only.

The [single MXFP8 benchmark report](mxfp8-benchmark.md) contains current
per-concurrency throughput and the baseline recheck. Its dataset retains the
October 13 stock/previous/grouped+PDL serving results under
`historical_validation.ordered_bf16`. The historical 4B runs used BF16 SSM and **FP8 E4M3 KV**, confirmed
by their archived launch arguments. They are not BF16 KV measurements.
See the [historical configuration alignment](mxfp8-benchmark.md#4b-historical-configuration-alignment)
for the matching ordered/FP8 KV comparison and the separate BF16 KV row. The main matrix now uses ordered BF16; stock comparisons retain their
historical backend choice and require `--no-ordered-bf16-ssm` for reproduction.

**Correctness qualification remains incomplete.** Three earlier kernel runs
reported unexplained output differences: 4B/M128 without PDL (16 elements,
max 0.0001296997), 4B/M128 under racecheck (494, max 0.0032501221), and
4B/M64 with PDL (16, max 0.0017547607). Passing long repeats and zero
sanitizer hazards/errors do not resolve those failures. Latest candidate versus
previous response texts also differ in 31/1040 2B and 34/1040 4B requests;
the cause is not isolated and model quality is not qualified.

Conv/QK/gate fusion helped only tiny batches and regressed at larger ones.
The unused fusion prototypes and duplicate benchmark script were removed;
their timing and debug records remain in the referenced audit archives.

### Reproduction

Use the qualified runtime packages; the serving report records their versions,
native binary hash, runtime source hashes, launch arguments and local paths.
Kernel timing rotates state pools and includes complete W4 cycles. The verifier
checks changing inputs/indices, graph replay, BF16 rounding, materialization,
stock fallback, resets and page canaries.

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
  .venv/bin/python tools/benchmark_ordered_bf16_layout.py --pools 0 \
  --rows 32 48 64 128 --output /tmp/bf16-layout.json
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
  VLLM_MACH_MXFP8_BF16_GDN_GROUPED=1 VLLM_MACH_MXFP8_BF16_GDN_PDL=1 \
  .venv/bin/python tools/verify_ordered_bf16_gdn.py --dtypes bfloat16 \
  --steps 1024 --failure-dir /tmp/bf16-failures --output /tmp/bf16-validation.json
```

The layout tool selects grouped decode; set the PDL environment flag explicitly
when comparing PDL. The verifier saves failing inputs/output/state with
`--failure-dir`. Add `--snapshot-before-step` for replay diagnosis; those
extra copies alter timing and memory traffic.
