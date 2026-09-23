# Qwen3.5 MoE projection dispatch

Mach packages 37 measured native W6A8 configurations for Qwen3.5-35B-A3B
MXFP6 non-expert projections. The table is registered on each worker's CUDA
device before workspace planning and graph capture. The existing MXFP6 native
kernels, expert dispatch, FP32 recurrent state, BF16 head and checkpoint stay
unchanged. No new kernel build is needed with the currently pinned dependency.

Eligibility requires the measured Qwen3.5 MoE geometry, native MXFP6 projection
weights, BF16 output, SM120, TP2/PP1/DP1 and no context parallelism, expert
parallelism, microbatching, LoRA or speculative decoding. Other configurations
retain their existing dispatch. The exact M values are 32, 36, 40, 48, 56, 64,
72, 80, 96, 112, 128 and 160; only the selected `(M,N,K)` entries change. They
do not limit request concurrency or KV capacity.

The table is bundled at `vllm_mach/mxfp6/profile/qwen35_moe_projection.json`.
It requires native config ABI `native-w6a8-30-v5`; an incompatible eligible
worker fails at startup before installing any entry. Native overrides are
process/device/shape/output-dtype scoped. Use the normal vLLM worker lifecycle
(one model per worker), and restart workers when changing the profile.

`VLLM_MACH_MOE_PROJECTION_TUNING` accepts `auto` (default), `0` (disabled) or
`1` (enabled for eligible models). `auto` leaves native autotuning in control
when it is enabled; `1` rejects that conflict. The Mach launcher already sets
`MXFP6_AUTOTUNE=off`, so the packaged table is used by default without any new
launch arguments or private paths. Disabling this feature requires a fresh
worker; it does not erase overrides from an already running process.

## Validation

The [measurement manifest](data/moe-projection-20260923.json) records the
dispatch selection hash, software identity, exact capture sizes, paired
results and fidelity references. Use the [existing serving benchmark](qwen35-moe.md)
with these settings; start separate workers with projection tuning disabled
and enabled to compare. Keep the checkpoint, GPU pair and request seeds fixed.

On two RTX 5090 GPUs with vLLM 0.29.0, a reverse-order HTTP pair used 3,000
input tokens and 1,000 output tokens, C warmup requests and 2C scored requests
per point. Both arms used max sequences 256, batched tokens 2048, automatic
KV at 90% memory, FULL_AND_PIECEWISE graphs and no prefix cache. The unchanged
Mach runtime at `c8c8600` was the baseline; subsequent main commits through
`f6de8c0` change build tooling and benchmark documentation, not runtime code.

| Concurrency | Baseline output tok/s | Projection output tok/s | Change |
| --- | ---: | ---: | ---: |
| 32 | 3151.07 | 3181.34 | +0.96% |
| 64 | 4036.29 | 4040.96 | +0.12% |
| 96 | 4439.42 | 4470.13 | +0.69% |
| 128 | 4920.94 | 4948.51 | +0.56% |
| 160 | 5128.22 | 5150.10 | +0.43% |

All 1,920 scored requests completed at the requested length, with no KV
preemption. All 960 matched response text hashes agreed. C64 is effectively
parity; these paired observations are not statistical confidence intervals.
Mean TTFT changed between -0.36% and +0.28%. Earlier forward-order tests
showed the same pattern. These are gains over Mach, not a new FP8/NVFP4 test.

A separate frozen 256-query diagnostic covers mathematics, code, Chinese and
English. At physical M32/M128/M160, all 31,437 gold raw logprobs match the
baseline and an independent prior projection run exactly, with zero repeat
difference. Matched-BF16 query-mean MAE remains 0.07573537, 0.07496286 and
0.07605973 respectively. M32 reuses the September 17 BF16 reference; the
other two use September 23 references. This measures numerical fidelity on
the tested data, not business task accuracy or universal bitwise equivalence.

The installed wheel was also reloaded independently: C32/C128/C160 measured
3174.04/4953.19/5150.42 output tok/s, respectively -0.23%/+0.09%/+0.01%
against the earlier projection run. All 640 response texts matched, with no
preemption and unchanged KV capacity. The packaged fidelity run matched all
31,437 frozen gold values. These reload checks validate packaging; they are
not a new paired comparison with the untuned baseline.

The compact/fused expert experiment is excluded: its added numerical error
does not meet this release's no-additional-loss requirement.
