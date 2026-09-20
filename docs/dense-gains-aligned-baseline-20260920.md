# Dense throughput gains: aligned MXFP6 baseline

The table starts from the measured, aligned MXFP6 baseline and calculates each increment against the immediately preceding displayed row using unrounded throughput. No new benchmarks were run. Other rows retain measurements from the original Mach profiles, with differences in compilation, attention, scheduling, GPU pairs, and measurement batches. These are configuration-level ratios, not isolated causal gains. Each configuration was measured in a single sweep with unlocked GPU clocks.

Each cell shows **gain over the previous row / total gain over the user-hosted stock FP8 service**. Incremental gain is `(current throughput / previous throughput - 1) × 100%`. The first row has no predecessor, shown as “—”. The mean is the equally weighted average of the four concurrency-specific gains, not the difference between rounded mean percentages.

| Configuration | c4 | c16 | c24 | c32 | Mean |
|---|---:|---:|---:|---:|---:|
| Aligned MXFP6 baseline | — / +18.69% | — / +17.91% | — / +19.22% | — / +18.64% | — / +18.61% |
| + AR/Norm | +1.08% / +19.98% | +1.40% / +19.56% | +5.39% / +25.64% | +5.18% / +24.78% | +3.26% / +22.49% |
| + persistent GDN | +14.16% / +36.97% | -0.05% / +19.50% | -0.00% / +25.64% | -0.02% / +24.76% | +3.52% / +26.72% |
| + BA overlap | +3.99% / +42.44% | +5.51% / +26.09% | +1.95% / +28.09% | +1.57% / +26.72% | +3.26% / +30.84% |
| + SwiGLU quant fusion | +0.60% / +43.29% | +0.02% / +26.11% | -0.28% / +27.73% | -0.19% / +26.48% | +0.04% / +30.90% |
| + GDN output quant fusion | -2.23% / +40.10% | -1.23% / +24.57% | +0.68% / +28.60% | +0.55% / +27.18% | -0.56% / +30.11% |
| + AR quant fusion | +1.79% / +42.61% | +1.29% / +26.17% | +0.90% / +29.76% | +0.55% / +27.88% | +1.13% / +31.60% |
| + lossless prefill | +0.12% / +42.77% | +1.19% / +27.67% | +1.62% / +31.86% | +1.97% / +30.40% | +1.23% / +33.18% |
| + owner prefill: **default Mach** | +0.64% / +43.69% | +2.15% / +30.41% | +3.06% / +35.90% | +3.48% / +34.94% | +2.33% / +36.24% |
| + FP16 SSM | +2.51% / +47.30% | +5.17% / +37.15% | +5.65% / +43.58% | +7.59% / +45.18% | +5.23% / +43.30% |
| + NVFP4 head: **full Mach** | +5.50% / +55.40% | +3.67% / +42.18% | +3.22% / +48.20% | +2.83% / +49.29% | +3.80% / +48.77% |

The SwiGLU quant fusion, Lossless prefill, and FP16 SSM transitions cross GPU groups or measurement batches. Their increments use the preceding displayed row without inserting the original bridge controls, so they include cross-group or batch variation. The AR/Norm transition also includes compact greedy sampling and a serving-profile change; it is not an isolated AR/Norm gain.

## Absolute throughput and data sources

Throughput is measured in output tokens/s, with 3,000 input and 1,000 output tokens per request.

| Baseline | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| opensource FP8 | 264.57 | 802.79 | 1011.43 | 1150.97 |
| MXFP6 | 314.03 | 946.57 | 1205.82 | 1365.46 |

FP8 uses CUTLASS block-FP8. Both baselines use Inductor, FULL_AND_PIECEWISE CUDA Graphs, FlashAttention 2, stock CUDA GDN, and CUSTOM/PYNCCL AllReduce. MXFP8 quantization fusion still has integration gaps, and NCCL versions differ (FP8: 2.30.7; Mach: 2.29.7), so this is not a strictly identical-environment, GEMM-only comparison. CUDA Graphs remain enabled throughout.

- [Core measurements and launch metadata](data/dense-core-ablation-20260920.json)
- [Prefill, SSM, and head measurements](data/dense-ablation-user-fp8-restarted-20260920.json)
- [Reverse-order fusion confirmation data](data/dense-core-confirmation-20260920.json)
- [User-provided FP8 startup evidence](data/fp8-user-startup-evidence-20260920.json)
- [Machine-readable gain table](data/dense-gains-aligned-baseline-20260920.json)
