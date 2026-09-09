# BA/QKV stream overlap

Port of `ba_overlap_direct_sum_service_v1`, on top of the direct SUM profile. It shipped disabled by default in `0.1.0a7`; version 0.1.0a8 has now completed Mach GPU acceptance and enables it in the optional long-prefill profile. Base package defaults are unchanged. Build/configuration: [GDN profile](../profiles/flashinfer-0.6.18-gdn/README.md#optional-m32-ba-overlap).

The auxiliary stream forks before QKV, runs the existing BA projection and split/contiguous operations, and joins after convolution but before packed recurrent decode. A warmup initializes the auxiliary cuBLAS path outside Graph capture. Captured producer buffers are retained; eager outputs use cross-stream allocator tracking. The original recurrent, norm, output projection and sampling paths are unchanged.

The source's private `mp_merged_weight` and `_qkv_mxfp6_decode_rows` checks are replaced with Mach's `state_for_rows(layer, 32)`, requiring `PREFILL_AND_M32` and a merged weight. The rest of the eligibility and fork/join order are retained. Enabling the new switch does not enable the separate fused-GDN gate automatically.

## Evidence and limits

The source experiment used Qwen3.8-27B EXL3 K5/K6, checkpoint MXFP6, TP2 SM120, direct SUM, and fixed 1024-input/256-output requests. A/B/B/A with a complete unscored c32 warmup per lifecycle recorded c32 1413.568349 → 1434.001227 output tokens/s (+1.445482%). c4/c16/c24 changes were −0.025483% / +0.074790% / +0.193063%. All 2432 scored requests completed, with no preemption and matching text hashes across the four rounds. Two lifecycles per arm do not provide a reliable confidence interval.

Source boundary checks covered real inputs at 48 layers per rank, including QKV, BA, convolution outputs and convolution state. They matched bitwise. This is retained source evidence, not a rerun on Mach. The scheduling mechanism does not depend on greedy sampling, but random sampling, other models, MoE and other shapes were not separately validated. Existing batch-dependent EXL3/MXFP6 routing remains unchanged.

## Mach GPU acceptance — September 9

The diagnostic service captured BA overlap and bitwise serial-reference assertions at all 48 eligible layers on each rank. QKV, BA and split b/a outputs were checked by device assertions during replay with changing real requests. All 40 task cases and 128 short c32 requests passed. The assertion nodes are enabled only by `VLLM_MACH_BA_OVERLAP_VERIFY=1`; they add reference work and must stay off for performance measurement. A separate full convolution/recurrent state dump was not repeated on Mach; the source state checks above remain supporting evidence.

A fresh service with both BA and collective verification disabled captured all 96 BA layer/rank paths, passed 40/40 tasks with zero pass/fail regressions, and completed 256 short c32 requests plus 128 long requests each at c16/c24/c32. This closes the previous GPU integration blocker for the tested stack. Use the matching v0.1.0a8 wheel/overlay and [long-prefill profile](../profiles/vllm-0.28.0/README.md#extended-prefill-profile). No isolated BA speedup or arbitrary batch-invariant output is claimed.
