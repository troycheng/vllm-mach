# TP2 optimization measurements — September 16, 2026

## P0: real-checkpoint baseline

Qwen3.8-27B native MXFP6, TP2/PP1, two RTX 5090s, BF16 activations/KV,
FP32 SSM and original BF16 head. Persistent GDN and BA overlap are enabled.
No optional precision changes. Five unprofiled repetitions follow a complete
workload warmup. The fixed KV allocation is 8,218,214,400 bytes/rank, with
512 scheduled tokens, maximum length 4096 and graphs at 1/2/4/8/16/24/32.

| Logical requests | Output tokens/request | Output tokens/s, mean ± sample SD | Mean request ITL |
|---|---:|---:|---:|
| 1 | 129 | 77.39 ± 0.09 | 9.788 ms |
| 4 | 1025 | 329.21 ± 0.73 | 10.909 ms |
| 16 | 1025 | 813.35 ± 3.69 | 15.398 ms |
| 32 | 1025 | 1113.77 ± 2.33 | 20.366 ms |

Each prompt has 2048 tokens. Throughput includes prefill and scheduling;
ITL uses each request's first/last decode timestamps. It includes mixed-batch
scheduling delays and is not a constant-M GPU iteration time. These are
separate decode-development workloads, **not** the published 3000/1000 serving
comparison. The initial 129-output B32 pilot reached only physical M24 at the
profiling point; it is excluded. Longer multi-request generations ensure a
sustained full batch. Profile collection asserts logical and padded rows
separately and rejects mislabeled results.

![TP2 development throughput](images/tp2-optimization-throughput.png)

The unprofiled baseline uses GPUs 4/5; diagnostic traces use 6/7. Tracing adds
external CUDA Graph events and synchronizes three selected iterations per rank.
Those timings cannot replace the unprofiled measurements. Each rank has its own
trace and interval union; rank times and overlapping stream time are never
added into an ITL estimate. Module durations are inclusive: GDN contains BA,
conv, recurrence, output norm and projection; MLP contains gate/up, activation
and down. AR/residual/GemmaNorm is recorded separately, including final norm.
BA overlap is included in its enclosing GDN duration, not added again.

Both ranks report 65 fused-AR-enabled model/layer modules, 48 prepared GDN
layers and the expected persistent/overlap routes. Actual packed weight shapes
confirm 48 × (8192,5120), 16 × (7168,5120), 64 × (5120,3072),
64 × (17408,5120) and 64 × (5120,8704) native GEMMs per rank.

Rank-0 kernel duration sums per profiled decode (diagnostic only):

| Physical M | Activation quantization | Scale initialization | SwiGLU | GDN output norm |
|---|---:|---:|---:|---:|
| 1 | 0.269 ms | 0.209 ms | 0.148 ms | 0.060 ms |
| 4 | 0.281 ms | 0.208 ms | 0.156 ms | 0.062 ms |
| 16 | 0.297 ms | 0.216 ms | 0.162 ms | 0.074 ms |
| 32 | 0.295 ms | 0.197 ms | 0.162 ms | 0.079 ms |

There are 256 independent activation quantizers and 256 scale initialization
kernels per decode. Source inspection identifies these initializations as the
packed activation-scale padding fill in `quantization.cu`; they are not GDN
barrier resets. This supports evaluating scale initialization first, then the
64 SwiGLU/down-quant pairs. Collective kernels include communication and fused
norm arithmetic; their time is not an estimate of removable communication wait.
BF16 projection and local/global argmax remain distinct in the raw traces.

## Fresh fidelity baseline

Both physical-M4 and physical-M32 runs score all 256 frozen queries and 10,479
gold tokens with the existing teacher-forced tool. Each repeats cohort zero
exactly. Every gold logprob equals the corresponding archived Mach-default
result, so the archived matched-M BF16 reference remains applicable.

| Physical rows | Gold-logprob MAE vs BF16 | 95% query-bootstrap CI |
|---|---:|---:|
| 4 | 0.085396 | 0.077317–0.093790 |
| 32 | 0.090643 | 0.082647–0.098954 |

![TP2 fidelity](images/tp2-optimization-fidelity.png)

[Collected measurements, contracts, shapes and per-rank module timings](data/tp2-p0.json).
Raw traces and trials are in `../tp2-optimization-20260916/`; the initial stalled
startup and short-output pilot are excluded. The host is shared and clocks are
not locked. The first stalled startup was terminated; the subsequent complete
startup and all retained trials succeeded.

Reproduce with the native-profile environment:

```bash
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=src python tools/benchmark_tp2_decode.py \
  --model MODEL --output RESULTS/baseline
CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=src python tools/benchmark_tp2_decode.py \
  --model MODEL --profile --output RESULTS/profile
# Repeat fidelity_native_mxfp6.py --arm gdn --skip-head-probe for M4 and M32.
python docs/data/collect_tp2_optimization.py --label P0 \
  --baseline RESULTS/baseline --profile RESULTS/profile \
  --fidelity-prefix RESULTS/fidelity --output docs/data/tp2-p0.json
python docs/data/plot_tp2_optimization.py docs/data/tp2-p0.json
```

The collector requires three correctly sized profile samples on both ranks.
The overlap accounting tests pass. Runtime optimization acceptance and serving
measurements follow independently; P0 alone makes no optimization-gain claim.
