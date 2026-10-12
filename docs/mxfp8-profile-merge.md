# MXFP8 profile integration — 2026-10-12

The comparison starts at main `f1ecdff` and development `4c2fcb2`.
The development branch has two additional commits (`9c73336`, `4c2fcb2`).
Its October 10 measurements used an older main reference and different runtime
settings; they do not establish incremental gains over the October 11 champion.

| Component | Current main | Development path |
|---|---|---|
| Native adapter | Worker installs behind FlashInfer CUTLASS; explicit workspace receipts | Native kernel selector, shared runtime workspace; optional FlashInfer backend |
| 4B accuracy | Dual activation for gate/up M32 and QKVZ M32/M64 | Ordinary single MXFP8 activation |
| Norm / MLP | Source-qualified compiler recipes; 2B complete FULL16/24 PDL chains | Explicit Gemma norm and fused SwiGLU + MXFP8 quantization; optional norm + quantization |
| PDL | 2B selected actual FULL16/24 captures | Native eligible M1–32, controlled by a flag |
| GDN | Ordered FP32 state, ownership/reset/materialization and mixed batches | Stock TP1 by default; optional older 4B adapter |
| 2B attention/head | Triton + K128 hybrid head already available | Same broad choices; different integration |
| 4B head / KV | K2048 score replacement, calibrated FP8 KV | K128 hybrid option; KV experiments have separate precision tradeoffs |
| Graph / scheduler | Qualified per-model capture policies; 2048 prefill boundary | General vLLM graph policy; published runs use 4096 max batch tokens |

## Integration decision

Both implementations can coexist, but replacing the champion kernel selector
would bypass 4B dual dispatch. Replacing its forwards also changes the compiler
recipes and 2B PDL chain. Therefore this merge retains the champion entry points
and integrates the development adapter through `vllm-mach-serve` without claiming
an additive champion speedup.

The merge keeps main's newer official native revision `a4cdc1b23a72cdedecb29707534fa4e3a595933f`,
all current profile/source guards, and GDN dispatch/variant diagnostics. The generic
PDL guard now accepts the official `use_pdl` schema as well as the development
`pdl_version >= 3` marker, while checking both GEMM entries and the companion
quantizer; the unmodified development guard rejected the qualified official library. Generic
Mach source hashes and graph-changing flags participate in AOT cache keys. The
standalone `VLLM_MACH_NATIVE_MXFP8=1` adapter keeps its FlashInfer selector, and an
absent optional runtime preserves import-light plugin behavior. Workspace tests
use a fresh process for a new model because captured native arenas cannot resize.

The development profile is an alternative configuration. FP8 KV, BF16 SSM and
more aggressively quantized checkpoints remain separate experiments, not new
champion defaults.

## Why b12x appears

The standalone Python distribution supplies compiler/intrinsics utilities for
Mach's BF16 BA GEMV at M1–8. FlashInfer's `backend="b12x"` for MXFP8/NVFP4 is a
separate backend selection. Neither establishes higher model accuracy by itself.
The standalone package was already required by main; it is not introduced by
this merge.

Fresh BA microbenchmarks compare actual checkpoint weights, BF16 activations,
FP64 references and CUDA-graph timing against PyTorch and FlashInfer BF16 backends.
End-to-end ablations replace only M1–8 BA execution with the original BF16 linear
operation. These diagnostic runs still import the pinned package and are not a
claim that dependency removal has been implemented.

## BA and QKVZ scheduling

The original development revision was inspected directly, including launcher
flags and archived run environments. `--gdn-ba-overlap` defaults to true but
MXFP8 TP1 additionally requires `VLLM_MACH_GDN_TP1=1`. Published 2B and several
4B runs explicitly set that flag to zero. The generic runs below also use zero.

| Path | Execution |
|---|---|
| Main 2B | No projection parallel hook; QKVZ and BA share the current stream |
| Main 4B, M32/64, GDN layer ordinals 1–23 | Auxiliary BA linear overlaps main QKVZ; join immediately after projections |
| Main 4B, M1–8 | Small BF16 b12x BA GEMV; no explicit QKVZ/BA stream overlap |
| Dev default TP1 | Original forward; no explicit QKVZ/BA stream overlap |
| Dev optional 4B TP1, M16/24/32 | Auxiliary BA + b/a contiguous overlaps main QKVZ + causal conv; join before recurrence |
| Dev optional 4B TP1, M1/2/4/8 | QKVZ first, then persistent kernel fusing BA, conv, gating and recurrence |

Dev's optional route admits pure decode, supported shapes and FP32/FP16 state;
prefill, mixed batches, speculative decoding and unsupported rows fall back.
Its geometry gate excludes 2B TP1. Main preserves layer ordinal zero because
its AOT code has intervening executable BA split/clone work. Main's M32/64 BA
uses ordinary BF16 linear, not the M1–8 b12x kernel. M means physical token rows,
not HTTP concurrency. PDL is a separate mechanism and does not establish this
independent two-stream schedule. Captured stream dependencies permit concurrency;
actual overlap requires a GPU timeline and depends on resource contention.

Dev's later join is a useful candidate for a separate controlled experiment,
not a demonstrated gain over current main. Its historical optional 4B adapter
report measured BA overlap at BS16 as 2266→2279 tokens/s (+0.6%, inconclusive),
and persistent at BS4 as 771→839 (+8.8%) against older baselines. These are
separate mechanisms. This integration does not change champion scheduling.

## Validation

[Machine-readable results and artifact hashes](data/mxfp8-profile-merge-20261012.json)
record fresh runs on RTX 5090. Raw harnesses, receipts and logs are retained in
`/data/lxy/detailed_benchmark/mxfp8-merge-validation-20261012`.

Main and merged champion runs use the same checkpoint, runtime, settings and
GPU per model, sequentially. Each arm completes 1,040 requests: 3,000 input and
1,000 output tokens, concurrency 4/16/24/32/48/64. Startup/JIT is excluded.
These are single runs, not confidence intervals.

| Model | Concurrency | Main output tok/s | Merged output tok/s | Change |
|---|---:|---:|---:|---:|
| 2B | 4 | 1661.0 | 1653.3 | -0.47% |
| 2B | 16 | 4905.6 | 4871.4 | -0.70% |
| 2B | 24 | 6294.7 | 6287.8 | -0.11% |
| 2B | 32 | 7292.3 | 7288.2 | -0.06% |
| 2B | 48 | 8086.1 | 8081.8 | -0.05% |
| 2B | 64 | 9048.4 | 9039.1 | -0.10% |
| 4B | 4 | 894.4 | 894.4 | +0.00% |
| 4B | 16 | 2539.9 | 2535.1 | -0.19% |
| 4B | 24 | 3277.7 | 3277.0 | -0.02% |
| 4B | 32 | 3737.2 | 3764.0 | +0.72% |
| 4B | 48 | 4284.3 | 4273.2 | -0.26% |
| 4B | 64 | 4768.5 | 4763.8 | -0.10% |

Pooled output throughput is 6390.3→6373.5 tok/s for 2B (-0.26%) and
3352.0→3351.6 for 4B (-0.01%). No incremental champion speedup is established.

Frozen teacher forcing covers 256 queries and 10,479 gold tokens for each
model at each physical M32 and M64. All four main/merged comparisons have
**zero changed gold logprobs**, and repeated first cohorts match exactly.

| Model | Physical rows | Main and merged MAE versus BF16 anchor |
|---|---:|---:|
| 2B | 32 | 0.0486508191 |
| 2B | 64 | 0.0491833655 |
| 4B | 32 | 0.0490381132 |
| 4B | 64 | 0.0532860474 |

These are fresh matched quality configurations, not a claim to reproduce every
historical production-quality setting. Dynamic serving response text differed
for 16/1040 requests on 2B and 23/1040 on 4B. Fixed-M logprob equality does not
establish bitwise free-generation equivalence under dynamic batching.

The alternative generic adapter completes the same 1,040 requests per model:
pooled 6297.9 tok/s (2B), 2917.4 (4B). Its M32 gold-token MAEs are 0.0489293843
and 0.0514010481 respectively; repeated first cohorts match. Those runs use
BF16 KV, FP32 state, the same model checkpoints, different runtime/GPU and
scheduler/head settings from the champions. Their performance and accuracy
cannot be attributed solely to the kernel adapter. They do not justify replacing
the champion defaults. The optional TP1 GDN overlap/persistent path is not
end-to-end revalidated by these generic runs.

Replacing only M1–8 BA execution with BF16 linear reduces c4 throughput from
1653.3 to 1540.3 tok/s on 2B (-6.83%), and 894.4 to 829.9 on 4B (-7.21%).
The 2B fixed-M4 gold logprobs remain identical across all 10,479 targets.
M32 also matches for both models, but does not exercise the small BA route.
No corresponding 4B M4 full-model precision claim is made.

CUDA-graph BA microbenchmarks use actual first-layer weights and FP64 references.
At M4, b12x takes 1.76/2.06 us for 2B/4B versus FlashInfer tinygemm 2.75/3.52 us.
At M8 they are nearly tied (2.72/3.42 versus 2.75/3.50 us). Results do not show a
meaningful precision advantage; the purpose of this existing dependency is
small-row latency. Microbenchmark samples do not prove universal equivalence.

CPU regression: 522 passed, 93 skipped, 1 CUDA-only case deselected, 84 subtests.
The relevant new MXFP8 GPU suite with the official native ABI: 60 passed,
including changing-input CUDA Graph replay, fused producers and workspace tests.
An exploratory broader TP2 multiprocess GPU run hung and was interrupted; it
is not reported as passing. Champion source files remain unchanged.
