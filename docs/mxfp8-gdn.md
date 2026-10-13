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

## Experimental BF16 ordered GDN

Both launchers accept `--ssm-dtype bfloat16 --ordered-bf16-ssm`.
BF16 state uses stock GDN unless the ordered flag is supplied. FP32 keeps
its existing ordered route. Direct profile users can set
`VLLM_MACH_MXFP8_SSM_DTYPE=bfloat16` and
`VLLM_MACH_MXFP8_ORDERED_BF16_SSM=1`; launchers reset the latter when absent.
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
deferred/four warps through 1024, then two warps. Larger shapes retain Triton.
Grouped deferred decode uses four warps through 768 head-rows, then two.

[Consolidated measurements and validation](data/ordered-bf16-experiments-20261013.json)
retain final single/multiple-pool kernel samples, latest serving aggregates,
source hashes, passing checks and unresolved discrepancies. Superseded
reports and removed prototype sources are archived outside the repository
at the location recorded in that file.

### Latest serving comparison (2026-10-13)

Each model uses one RTX 5090, TP1, BF16 SSM, 19 GiB KV and the same GPU for
all variants. 2B uses TRITON_ATTN/BF16 KV/hybrid head; 4B uses
FLASHINFER/FP8 KV/compact greedy head. Stock means stock GDN inside the
same MXFP8 profile. The two models run concurrently on distinct GPUs;
variants run sequentially as candidate, previous, stock, each on a fresh service.
The sixpoint benchmark uses 3000 input/1000 output tokens, greedy decoding
with EOS ignored and identical prompts. Every suite completes 1040/1040
scored requests. Warmup is excluded; each variant has only one full trial.

Output throughput is in tokens/s. Previous ordered disables both new flags;
candidate enables both. Fusion and RMSNorm PDL are absent from serving.

| Model | Concurrency | Stock | Previous ordered | Grouped + PDL | Change vs previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2B | 4 | 1674.26 | 1712.67 | 1679.95 | -1.91% |
| 2B | 16 | 5150.52 | 5307.85 | 5296.31 | -0.22% |
| 2B | 24 | 6761.96 | 6945.09 | 6930.04 | -0.22% |
| 2B | 32 | 7687.79 | 7934.10 | 7961.93 | +0.35% |
| 2B | 48 | 9044.97 | 9486.32 | 9513.77 | +0.29% |
| 2B | 64 | 10121.71 | 10472.36 | 10449.81 | -0.22% |
| 4B | 4 | 939.45 | 936.45 | 937.36 | +0.10% |
| 4B | 16 | 2737.06 | 2741.93 | 2753.05 | +0.41% |
| 4B | 24 | 3626.55 | 3573.93 | 3594.11 | +0.56% |
| 4B | 32 | 4148.11 | 4085.16 | 4104.93 | +0.48% |
| 4B | 48 | 4784.13 | 4705.02 | 4747.79 | +0.91% |
| 4B | 64 | 5304.74 | 5275.68 | 5327.97 | +0.99% |

| Model | Pooled stock | Pooled previous | Pooled grouped + PDL | vs stock | vs previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2B | 6857.27 | 7084.47 | 7061.35 | +2.98% | -0.33% |
| 4B | 3672.51 | 3643.05 | 3664.99 | -0.20% | +0.60% |

Pooled throughput is total output tokens divided by summed scored durations.
The measured incremental changes do not establish a repeatable serving gain.

**Correctness qualification remains incomplete.** Three earlier kernel runs
reported unexplained output differences: 4B/M128 without PDL (16 elements,
max 0.0001296997), 4B/M128 under racecheck (494, max 0.0032501221), and
4B/M64 with PDL (16, max 0.0017547607). Passing long repeats and zero
sanitizer hazards/errors do not resolve those failures. Latest candidate versus
previous response texts also differ in 31/1040 2B and 34/1040 4B requests;
the cause is not isolated and model quality is not qualified.

Conv/QK/gate fusion helped only tiny batches and regressed at larger ones.
The unused fusion prototypes and duplicate benchmark script were removed;
selected timings remain in the consolidated report for future tuning.

### Reproduction

Use the qualified runtime packages; the serving report records their versions,
native binary hash, runtime source hashes, launch arguments and local paths.
Kernel timing rotates state pools and includes complete W4 cycles. The verifier
checks changing inputs/indices, graph replay, BF16 rounding, materialization,
stock fallback, resets and page canaries.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  .venv/bin/python tools/benchmark_ordered_bf16_layout.py --pools 0 \
  --rows 32 48 64 128 --output /tmp/bf16-layout.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  VLLM_MACH_MXFP8_BF16_GDN_GROUPED=1 VLLM_MACH_MXFP8_BF16_GDN_PDL=1 \
  .venv/bin/python tools/verify_ordered_bf16_gdn.py --dtypes bfloat16 \
  --steps 1024 --failure-dir /tmp/bf16-failures --output /tmp/bf16-validation.json
```

The layout tool selects grouped decode; set the PDL environment flag explicitly
when comparing PDL. The verifier saves failing inputs/output/state with
`--failure-dir`. Add `--snapshot-before-step` for replay diagnosis; those
extra copies alter timing and memory traffic.
