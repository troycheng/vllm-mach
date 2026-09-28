# MoE throughput gains: MXFP6 baseline

MXFP6 was retested on September 28, 2026, using physical GPUs 6/7 (RTX 5090, TP2). Each throughput is the arithmetic mean of two sweeps with unlocked GPU clocks. The sequence starts with decode CUDA Graphs and adds one optimization flag or graph mode per row. The denominator remains the historical user-hosted open-source vLLM 0.29.0 FP8 service measured September 17, as requested; its launch configuration was not inspected. Total gains compare deployments across dates.

Each cell shows **gain over the previous row / total gain over the open-source vLLM 0.29.0 FP8 service**. Incremental gain is `(current throughput / previous throughput - 1) × 100%`; total gain replaces the denominator with FP8 throughput. Calculations use unrounded rates. The first row has no predecessor, shown as “—”. Mean is the equally weighted average of the four concurrency-specific percentages.

| Configuration | c4 | c16 | c24 | c32 | Mean |
|---|---:|---:|---:|---:|---:|
| MXFP6 baseline (decode graphs only) | — / +20.20% | — / +19.08% | — / +20.52% | — / +20.76% | — / +20.14% |
| + prefill CUDA Graph | +8.32% / +30.20% | +23.28% / +46.81% | +22.22% / +47.30% | +25.58% / +51.66% | +19.85% / +43.99% |
| + compact greedy | +0.60% / +30.98% | +2.17% / +49.99% | +2.15% / +50.46% | +2.29% / +55.12% | +1.80% / +46.64% |
| + AR/Norm | +13.31% / +48.42% | +5.80% / +58.69% | +9.29% / +64.44% | +7.41% / +66.61% | +8.95% / +59.54% |
| + persistent GDN | +0.63% / +49.35% | +0.10% / +58.84% | +0.07% / +64.56% | +0.28% / +67.08% | +0.27% / +59.96% |
| + BA overlap: **default Mach (projection off)** | -0.68% / +48.34% | +0.07% / +58.94% | +0.01% / +64.58% | -0.04% / +67.01% | -0.16% / +59.72% |
| + FP16 SSM | +1.89% / +51.15% | +4.54% / +66.15% | +4.79% / +72.46% | +4.33% / +74.25% | +3.89% / +66.00% |
| + NVFP4 head | +7.81% / +62.96% | +4.86% / +74.22% | +0.63% / +73.55% | +1.56% / +76.97% | +3.72% / +71.93% |
| + projection dispatch: **full Mach** | -0.99% / +61.34% | -0.08% / +74.09% | -0.21% / +73.19% | +1.33% / +79.32% | +0.01% / +71.98% |

Persistent GDN and BA overlap rows measure flag changes, not demonstrated kernel gains. A separate cold-cache c4/c32 check reports 30 prepared layers per rank and zero persistent/overlap host dispatch counts, including graph capture. Small changes in these rows should not be attributed to the kernels. Adjacent percentages depend on the displayed order and cannot be added to obtain total gain.

## Absolute throughput and data sources

Throughput is measured in output tokens/s, with 3,000 input and 1,000 output tokens per request. The rows below correspond to the gain table above.

| Configuration (output tokens/s) | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| opensource FP8 (vLLM 0.29.0) | 679.89 | 1471.58 | 1774.64 | 1962.14 |
| MXFP6 baseline (decode graphs only) | 817.23 | 1752.43 | 2138.81 | 2369.52 |
| + prefill CUDA Graph | 885.24 | 2160.41 | 2613.96 | 2975.69 |
| + compact greedy | 890.55 | 2207.21 | 2670.15 | 3043.69 |
| + AR/Norm | 1009.12 | 2335.18 | 2918.21 | 3269.12 |
| + persistent GDN | 1015.44 | 2337.45 | 2920.36 | 3278.40 |
| + BA overlap: **default Mach (projection off)** | 1008.56 | 2338.97 | 2920.74 | 3277.02 |
| + FP16 SSM | 1027.67 | 2445.05 | 3060.57 | 3418.97 |
| + NVFP4 head | 1107.95 | 2563.79 | 3079.94 | 3472.37 |
| + projection dispatch: **full Mach** | 1096.93 | 2561.85 | 3073.56 | 3518.44 |

MXFP6 settings: vLLM 0.29.0, VLLM_COMPILE, TRITON_ATTN, 2,048 batched tokens, 64 maximum sequences, 8 GiB/rank KV memory, and no prefix caching. The baseline uses FULL_DECODE_ONLY; the prefill step switches to FULL_AND_PIECEWISE. MXFP6 starts with FP32 SSM and all optional flags off. Projection dispatch from `c507ef3` is added only in the final row; it is enabled by default in the updated runtime.

Each sweep scores 16/32/48/64 requests at c4/c16/c24/c32, after a concurrency-sized 128-output-token warmup. Prompts are uniform token IDs with the archived seeds. Initialization and warmup are excluded. All 2,880 newly scored MXFP6 requests passed status, token-count and throughput-arithmetic checks. Two sweeps do not establish confidence intervals.

- [Retest measurements, exact launches, request validation and gain calculations](data/moe-recheck-20260928.json)
- [Historical open-source FP8 measurements](data/qwen35-default-full-20260917.json)
- [Projection dispatch implementation and validation](moe-projection-20260923.md)
