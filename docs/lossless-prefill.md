# Lossless prefill port

Development integration after a6; not a published release. Build and configuration are documented in [native/lossless_prefill](../native/lossless_prefill/README.md).

This ports the frozen `lossless_prefill_service_v1` communication replacement into Mach. Device arithmetic and block coding are unchanged. The native registration uses a Mach namespace; the Python adapter replaces private binary paths and hard-coded library hashes with a separately installed extension. It validates device, dependency versions and workspace metadata, and adds an opt-in gate. Unknown workspace metadata or a missing enabled extension raises an error.

The native extension built with CUDA 13.0, `sm_120f` and fast math against FlashInfer 0.6.18. The package tests passed 124 cases. On two RTX 5090 GPUs, the installed FlashInfer reference, recompiled control and packed implementation matched bitwise for six fixtures per rank, including three retained model captures. Those captures repeat sampled rows and are not full-model matrices. Changing-input Graph checks and in-place residual alias checks passed. Three mixed-workspace rounds alternated M32 one-shot Graph, M4096 two-shot, M24 one-shot Graph and M4096 two-shot with PDL enabled; both ranks matched the reference.

The native source includes a no-model synthetic validation mode. Full model capture data is not distributed. No new codec-wide exhaustive sweep or sanitizer run was performed for this unchanged device implementation; those remain evidence from the source experiment, not new port test results.

The integrated Mach service passed 40/40 tasks with zero pass/fail regressions. Both workers used the new path and verified 128 distinct norm instances each, comparing full 4096×5120 real-model residual and normalized outputs bitwise against installed FlashInfer. All 256 fixed-token c32 requests completed with 1024 input and 256 output tokens per request. Temporal QKV/QKVZ, specialized GDN and seven decode Graph sizes remained active. The final development wheel passed installation and import without initializing CUDA.

The task suite had 34/40 exact text matches to its stored reference. Task pass/fail retention and communication-boundary bitwise checks are different criteria; neither establishes full-model bitwise equivalence.

Diagnostic verification adds reference collectives and synchronization; its throughput must not be cited as release performance. Historical +0.88% c32 evidence belongs to the frozen source experiment and does not transfer automatically to the Mach port. No new Mach A/B performance claim is made.
