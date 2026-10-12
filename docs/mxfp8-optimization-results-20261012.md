# Merged MXFP8 profile validation — 2026-10-12

Selected implementation: `e5a444e`, integrated into main at `04ffdb4`.
[Machine-readable merged results and artifact hashes](data/mxfp8-optimization-20261012.json).

## Throughput

One RTX 5090 per model, TP1, 3,000 input and 1,000 output tokens. The six
concurrency points are 4/16/24/32/48/64, with 40/160/120/160/240/320 requests.
Each complete model run scores 1,040 requests and 1,040,000 output tokens.
Inputs are deterministic uniform token IDs. Startup/JIT and prewarm are
excluded. Throughput is output tokens per second.

The open-source FP8 columns below use the existing historical reference:
[2B source](data/mxfp8-2b-triton-hybrid-sixpoint-20261011.json) and
[4B source](data/mxfp8-qualification-20261006.json). FP8 was not rerun in this
validation. The 2B reference used a different physical GPU; both comparisons
are descriptive historical-reference comparisons, not new paired trials.

### Qwen3.5-2B

| Concurrency | Open-source FP8 | Merged MXFP8 | Gain |
|---|---:|---:|---:|
| 4 | 1261.9 | 1665.6 | +31.99% |
| 16 | 3814.3 | 4953.6 | +29.87% |
| 24 | 5069.7 | 6350.4 | +25.26% |
| 32 | 5870.2 | 7336.5 | +24.98% |
| 48 | 6837.6 | 8683.4 | +26.99% |
| 64 | 7578.9 | 9568.6 | +26.25% |

Merged pooled throughput: **6578.05 tok/s** (total scored output tokens divided by summed measured duration).

### Qwen3.5-4B

| Concurrency | Open-source FP8 | Merged MXFP8 | Gain |
|---|---:|---:|---:|
| 4 | 684.0 | 921.3 | +34.70% |
| 16 | 2053.5 | 2617.7 | +27.47% |
| 24 | 2559.3 | 3386.0 | +32.30% |
| 32 | 2887.8 | 3832.0 | +32.69% |
| 48 | 3220.5 | 4360.1 | +35.38% |
| 64 | 3402.8 | 4836.2 | +42.13% |

Merged pooled throughput: **3432.96 tok/s** (total scored output tokens divided by summed measured duration).

## Selected execution

* Fuse source-qualified SwiGLU with MXFP8 quantization at supported M≤128,
  preserving the single final BF16 rounding and CUDA E8M0 conversion semantics.
  M16/M24 retain PDL participation. Unsupported prefill sizes retain their
  compiled producer and quantizer.
* The 4B BA/QKVZ fork/join covers physical M1/2/4/8/16/24/32/64 at GDN ordinals
  1–23. BA runs on the auxiliary stream; the current stream joins after QKVZ.
  Ordinal 0 retains its original execution. Ordered GDN state handling is kept.
* K128 proposals with BF16 refinement cover eligible greedy sampling through
  M64. The 2B launcher uses `--hybrid-head`; the 4B launcher enables it by default
  and supports `--no-hybrid-head`. Scoring, constrained/stochastic sampling,
  prefill and unsupported cases retain the complete head.

Post-benchmark receipts record 41,165/41,121 successful compact-head calls for
2B/4B, including the six target row counts. The 4B receipt records parallel
capture for all selected row counts and ordinals. Projection/fusion counters
measure construction/capture, not graph replays; sampler counters count calls.

## Precision

Frozen teacher forcing uses 256 queries and 10,479 gold tokens per run.
The values below are gold-token logprob MAE against the frozen BF16 anchors.
These are numerical fidelity measurements, not general task-accuracy scores.

| Model | Physical rows | MAE | First-cohort repeat difference |
|---|---:|---:|---:|
| 2B | 32 | 0.04865081908 | 0.0 |
| 2B | 64 | 0.04918346296 | 0.0 |
| 4B | 32 | 0.04903811320 | 0.0 |
| 4B | 64 | 0.05328604736 | 0.0 |

Each model's untimed head audit checked 16,500 rows at M4, 24,448 at M32 and
29,440 at M64, with zero differing argmax tokens and zero positive logit losses
against its complete head on the same hidden states. Counts include padding
and repeats. The 2B reference head is complete BF16; the 4B reference is its
complete K2048 refined head. Candidate search remains approximate; arbitrary
future prompts and free-running generations are not claimed to be identical.

M4/M64 precision runs used the selected kernels with a temporary head limit
of 128; final serving limits it to 64. M32 was rerun with that final limit.
The 4B physical-M4 directory name records M32 capacity; no physical-M4 BF16
anchor MAE is reported for it.

## Checks and artifacts

The CPU regression suite passed 529 tests with 84 passing subtests, 105 skips
and one existing CUDA-only test deselected. The fused-producer suite passed
18 tests on SM120, including source/layout/consumer guards, capture receipts
and changing-input graph replay at M1/4/16/24/32/64 for both widths. FP8 bytes
and valid E8M0 scales match the compiled reference plus FlashInfer quantization.

The serving runtime is pinned vLLM 0.29.0, FlashInfer 0.6.18 and native revision
`a4cdc1b23a72cdedecb29707534fa4e3a595933f`. Retained merged serving and precision
artifacts are in `/data/lxy/detailed_benchmark/mxfp8-opt-20261012`.
Runtime dependencies remain in
`/data/lxy/detailed_benchmark/mxfp8-merge-validation-20261012`.
The report JSON contains paths and hashes for retained results and receipts.
