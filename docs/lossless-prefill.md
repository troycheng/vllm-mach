# Lossless prefill port

## Direct SUM increment

Release `0.1.0a7` includes `lossless_direct_sum_prefill_service_v1`, selected and retained by the source experiment on September 8, 2026. Native package `0.1.0a3` adds a separate direct library; enable it with `VLLM_MACH_LOSSLESS_PREFILL_DIRECT=1` alongside the input and SUM switches. The previous modes remain available. See [build and configuration](../native/lossless_prefill/README.md#direct-sum).

The source experiment's A/B/B/A service comparison recorded c32 throughput of 1403.5140 → 1412.2002 output tokens/s (+0.6189%) against input+SUM, with two lifecycles per arm. Its c4/c16/c24 changes were −0.0525% / +0.1226% / +0.2540%. These are historical source-stack measurements, not Mach release results, and must not be added to earlier codec gains.

The source investigation found that the shared model adapter is sensitive to batch composition: decode requests can switch between Temporal EXL3 and checkpoint MXFP6 routes when mixed with prefill. In controlled cohorts with matching recorded batch paths, old SUM and direct SUM produced identical token sequences for all 192 requests. Uncontrolled c24 runs differed, including repeated runs of old SUM. This does not establish arbitrary batch-invariant output or complete business-quality acceptance. The direct collective preserves its tested BF16 boundary; it does not fix or redefine the model adapter's route-dependent numerical behavior. The complete producer stream/event contract for residual preloads has not been independently verified at runtime; the source audit found the preload ordering also present in the reference implementation and did not establish it as the cause of the text differences.

Mach acceptance: all 126 package tests passed after clearing the serving image's profile environment. The native extension built with CUDA 13.0; installed FlashInfer, old SUM and direct SUM matched bitwise on eight fixtures per rank, including alias and changing-input Graph checks. Three mixed-workspace PDL rounds passed. One complete service lifecycle activated direct SUM on both ranks and verified 128 full 4096×5120 residual/norm instances per rank bitwise against installed FlashInfer. All 40 task cases passed with zero pass/fail regressions; 35/40 outputs exactly matched the stored reference text. All 256 fixed-token c32 requests completed (1024 input / 256 output tokens each). Verification was enabled, so this run does not establish a Mach performance gain. Source exhaustive, poison and sanitizer checks were not repeated for this port.

## Input+SUM increment

Release `0.1.0a7` also carries the incremental `lossless_sum_prefill_service_v1` path. Set `VLLM_MACH_LOSSLESS_PREFILL_SUM=1` in addition to the original opt-in switch and rebuild native package `0.1.0a2`. Input-only remains available. The additional SUM header region raises the minimum workspace capacity to 84,213,760 bytes; no new workspace pool or KV allocation is introduced. The new device source preserves the source experiment's arithmetic and synchronization, with only registration namespace changes.

Its source experiment recorded +0.4854% c32 in the first full sweep and +0.6178% in a separate, fully conditioned c32 confirmation round, each with two lifecycles per arm. These rounds must not be pooled and neither establishes a general speedup. They compare input+SUM against input-only, not against unmodified FlashInfer or released Mach.

The following records describe the pre-release integration checks included with `0.1.0a7`. Build and configuration are documented in [native/lossless_prefill](../native/lossless_prefill/README.md).

## Input+SUM acceptance

The native package built both modes with CUDA 13.0 for SM120. All 125 package tests passed. On two RTX 5090 GPUs, installed FlashInfer, input-only and input+SUM matched bitwise across eight fixtures per rank. In-place residual aliasing, changing-input Graph replay, and three mixed-workspace rounds of M32 Graph / M4096 / M24 Graph / M4096 with PDL also passed.

The complete Mach service activated `mode=input+sum` on both ranks. Each rank verified full 4096×5120 residual and normalized outputs at 128 distinct norm instances against installed FlashInfer, bitwise. The task suite passed 40/40 with zero pass/fail regressions and 36/40 exact text matches to its stored reference. All 256 c32 requests completed with 1024 input and 256 output tokens each. Temporal M24, specialized GDN and decode Graph capture remained active.

This was one diagnostic service lifecycle with reference verification enabled, not an A/B performance measurement. Exhaustive codec and sanitizer results remain source-experiment evidence; those runs were not repeated for this port. The task suite and collective checks do not establish full-model bitwise equivalence.

## Earlier input-only acceptance

This ports the frozen `lossless_prefill_service_v1` communication replacement into Mach. Device arithmetic and block coding are unchanged. The native registration uses a Mach namespace; the Python adapter replaces private binary paths and hard-coded library hashes with a separately installed extension. It validates device, dependency versions and workspace metadata, and adds an opt-in gate. Unknown workspace metadata or a missing enabled extension raises an error.

The native extension built with CUDA 13.0, `sm_120f` and fast math against FlashInfer 0.6.18. The package tests passed 124 cases. On two RTX 5090 GPUs, the installed FlashInfer reference, recompiled control and packed implementation matched bitwise for six fixtures per rank, including three retained model captures. Those captures repeat sampled rows and are not full-model matrices. Changing-input Graph checks and in-place residual alias checks passed. Three mixed-workspace rounds alternated M32 one-shot Graph, M4096 two-shot, M24 one-shot Graph and M4096 two-shot with PDL enabled; both ranks matched the reference.

The native source includes a no-model synthetic validation mode. Full model capture data is not distributed. No new codec-wide exhaustive sweep or sanitizer run was performed for this unchanged device implementation; those remain evidence from the source experiment, not new port test results.

The integrated Mach service passed 40/40 tasks with zero pass/fail regressions. Both workers used the new path and verified 128 distinct norm instances each, comparing full 4096×5120 real-model residual and normalized outputs bitwise against installed FlashInfer. All 256 fixed-token c32 requests completed with 1024 input and 256 output tokens per request. Temporal QKV/QKVZ, specialized GDN and seven decode Graph sizes remained active. The final development wheel passed installation and import without initializing CUDA.

The task suite had 34/40 exact text matches to its stored reference. Task pass/fail retention and communication-boundary bitwise checks are different criteria; neither establishes full-model bitwise equivalence.

Diagnostic verification adds reference collectives and synchronization; its throughput must not be cited as release performance. Historical +0.88% c32 evidence belongs to the frozen source experiment and does not transfer automatically to the Mach port. No new Mach A/B performance claim is made.
