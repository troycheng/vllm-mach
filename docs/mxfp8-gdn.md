# Ordered W4 FP32 GDN profile

`vllm_mach.mxfp8.gdn` packages the ordered-state part of the Qwen3.5-4B MXFP8 champion. It runs inside vLLM's existing GDN opaque operation. Installation, native MXFP8, FP8 KV calibration, graph policy, the projection kernels and the full-vocabulary head must be composed by the complete model profile. This module alone is not a champion deployment or a performance qualification.

The serving boundary is vLLM 0.29.0, one SM120 GPU, TP1, BF16 model and convolution state, FP32 recurrent state, `max_num_seqs=128`, `max_model_len=8192`, and 19 GiB KV memory. BF16 and `fp8_e4m3` attention KV configurations are accepted; the latter still requires the complete profile to install its calibrated scales before profiling/capture. Prefix caching, speculative decoding/MTP, ReplaySSM, connectors, cache offload and CPU weight offload are rejected. Source hashes pin vLLM's GDN implementation, packed FLA implementation and GPU runner, independent of their install location.

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

The runner's original new/cached request updates run before metadata resets. Newly assigned positive IDs invalidate pending terms while leaving base-state zeroing and convolution-state initialization with stock vLLM. New requests with retained computed context and KV block-copy requests are rejected. Slot reuse follows the same allocation reset path. On prefill or unsupported decode, raw pending factors are applied on the current stream before stock reads dense FP32 state; materialization resets age and prefix before the stock operation.

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
