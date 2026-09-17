# Native MXFP6 fidelity and serving comparison

GDN measurements: September 16, 2026; stock FP8/NVFP4 baselines reused
from September 15 at the user’s request. Native vLLM 0.29.0, FlashInfer 0.6.18, mxfp6-sm120 0.2.1,
Qwen3.8-27B on two RTX 5090 GPUs per run, TP2. Model checkpoint:
[nekofish/Qwen3.8-27B-MXFP6](https://huggingface.co/nekofish/Qwen3.8-27B-MXFP6).
The [September 15 measurements](native-fidelity-20260915.md) are archived separately.

## Configurations and isolated comparisons

The launcher now enables persistent at physical M1/2/4/8 with FP32 or FP16 recurrent
state, and BA overlap at M16/24/32 with either state dtype. FP16 persistent
retains FP32 accumulation and rounds only when storing updated state. Prefill, mixed/speculative and
unsupported calls retain native execution. See [GDN implementation and validation](gdn-decode.md).

The raw arm names retain their experiment meaning:

| Arm | Meaning | Persistent | BA overlap |
|---|---|---:|---:|
| default | Previous default, fused AR/Norm and compact BF16 greedy sampling | off | off |
| persistent | Previous default plus persistent only | on | off |
| gdn | September 16 default, before lossless/owner prefill | on | on |
| full | Previous full: FP16 SSM, lossless/owner prefill, NVFP4 head | off | off |
| full_ba | Full before the c4 fix: BA overlap only | off | on |
| full_gdn | Corrected full options, including FP16 persistent | on | on |

This keeps the two optimizations independently measurable. `--no-gdn-persistent`
and `--no-gdn-ba-overlap`, together with `--no-lossless-prefill` and
`--no-owner-prefill`, reproduce the previous default. Stock FP8/NVFP4
use unpatched official packages, no Mach plugin and FlashInfer AllReduce disabled.

## September 17 Dense default retest

Dense default now enables lossless/owner prefill alongside both GDN routes,
while retaining FP32 SSM and the BF16 head. Full adds only FP16 SSM and
NVFP4 candidate search; its execution settings are unchanged.

The changed default was remeasured on RTX 5090 GPUs 6/7 with TP2, fixed
8,218,214,400-byte KV allocation, 4096 batched tokens and the original decode
graphs. All 380 scored requests completed with exactly 3000 input and 1000
output tokens. Prompts, seeds, arrivals, sampling and warmups match the
archived per-concurrency contracts. Full and stock baselines are retained.

| Configuration | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| FP8 · vLLM 0.29 baseline | 266.97 | 806.06 | 1018.55 | 1160.44 |
| NVFP4 · vLLM 0.29 baseline | 378.97 | 1142.87 | 1424.61 | 1607.56 |
| MXFP6 · Mach default | 377.17 | 1043.08 | 1352.54 | 1537.00 |
| MXFP6 · Mach full | 384.90 | 1124.27 | 1453.57 | 1674.83 |

Equal-weight mean gain over the retained stock FP8 baseline: **33.98%**.
This is a single-run throughput sweep. The September 16 fidelity and GDN
ablation results below retain their original configurations.

The `prefill_default` arm in [raw serving results](data/native-serving.json)
contains launch settings and request-level timings; `gdn` retains the prior
default. Both-rank owner initialization was observed. All 43 prefill tests
passed, including native lossless/owner transport and graph checks.

Reproduce both affected profiles and collect their data:

```bash
PYTHONPATH=src python tools/retest_profile_defaults.py --output RESULTS --devices 6,7
python docs/data/collect_profile_defaults.py --results RESULTS
python docs/data/plot_comparison.py --throughput-only
```

## September 16 serving throughput

| Configuration | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| FP8 · stock vLLM 0.29 | 266.97 | 806.06 | 1018.55 | 1160.44 |
| MXFP6 · previous default | 325.91 | 978.17 | 1255.64 | 1423.86 |
| MXFP6 · persistent only | 371.18 | 977.21 | 1254.91 | 1421.59 |
| MXFP6 · September 16 default | 371.18 | 999.87 | 1278.40 | 1444.99 |
| MXFP6 · previous full | 353.19 | 1099.27 | 1422.97 | 1648.55 |
| MXFP6 · full without persistent | 353.05 | 1126.29 | 1453.48 | 1679.71 |
| MXFP6 · Mach full | 384.90 | 1124.27 | 1453.57 | 1674.83 |
| NVFP4 · stock vLLM 0.29 | 378.97 | 1142.87 | 1424.61 | 1607.56 |


Persistent alone improves c4 by **13.89%**; its inactive
c16/c24/c32 points differ by at most 0.16%.
BA overlap alone improves full-profile c16/c24/c32 by
**2.46% / 2.14% / 1.89%**;
its inactive c4 point differs by -0.04%.
The combined new default improves c4/c16/c24/c32 over the previous default by
**+13.89% / +2.22% / +1.81% / +1.48%**.
Equal-weight mean gains over stock FP8 are **28.28%** for the new
default and **42.67%** for the new full profile.

![Serving throughput](images/throughput-comparison.png)

All 3,040 measured requests completed with exactly 3000 input and 1000 output tokens.
The frozen [ShareGPT-prefix manifest](data/serving-prompts.json) supplies identical
token IDs, request seeds and a 100 requests/s exponential arrival schedule;
greedy, top-k 20, top-p 0.95, ignore EOS. c4/c16/c24/c32 measure
20/80/120/160 requests, each after min(32, request count) warmups of 128 tokens.

Mach uses maximum 32 sequences, no compilation, FULL_DECODE_ONLY graphs and
8,218,214,400 KV bytes per rank. Stock retains default compilation/graphs,
64 maximum sequences and 90% GPU memory utilization. All use TRITON_ATTN,
no prefix caching, 16,384 maximum context and a 4096-token prefill budget.
The new serving arms run sequentially on GPUs 6/7; fidelity uses GPUs 4/5 on the
same host. Clocks are not locked and other GPUs run unrelated work.
These short single-run measurements have no throughput confidence interval.
This compares deployable profiles, rather than isolating quantization alone.

[Raw requests, latencies, warmups and launch settings](data/native-serving.json).

## Numerical fidelity

| Configuration | M32 MAE | 95% query-bootstrap interval |
|---|---:|---:|
| FP8 · stock vLLM 0.29 | 0.061430 | 0.054315–0.069084 |
| MXFP6 · previous default | 0.090643 | 0.082647–0.098954 |
| MXFP6 · persistent only | 0.090643 | 0.082647–0.098954 |
| MXFP6 · September 16 default | 0.090643 | 0.082647–0.098954 |
| MXFP6 · previous full | 0.089624 | 0.082036–0.097311 |
| MXFP6 · full without persistent | 0.089624 | 0.082036–0.097311 |
| MXFP6 · Mach full | 0.089624 | 0.082036–0.097311 |
| NVFP4 · stock vLLM 0.29 | 0.170886 | 0.155098–0.187517 |


All arms and BF16 reproduce cohort zero exactly. The persistent-only M32 arm
falls back to native and matches the previous default's gold-token logprobs
exactly. Both FP32 BA overlap (new default) and FP16 BA overlap (new full)
also match their respective previous profiles exactly on all scored tokens.
The corrected full_gdn arm also exactly matches full_ba at M32, where
persistent is inactive. This establishes measured equivalence for BA scheduling
on this corpus.

![Physical-M32 fidelity](images/accuracy-comparison.png)

The separate physical-M4 diagnostic activates persistent: previous default
MAE **0.089853**, persistent-only and combined new default
MAE **0.085396**. The combined and persistent-only gold
logprobs match exactly. Paired Δ MAE is **-0.004457**,
95% query-bootstrap interval **[-0.007531,
-0.001629]**. All M4 arms repeat exactly.
Persistent changes arithmetic; this lower numerical error on the frozen corpus
is not evidence of improved task accuracy or a universal quality guarantee.

The corrected FP16 full profile records M4 MAE **0.086200**, compared
with **0.084917** without persistent. Paired Δ **+0.001282**,
95% interval **[-0.002213, +0.004822]**, crosses zero;
this does not establish a fidelity improvement or degradation on this corpus.
Both repeated cohorts match exactly.

![Physical-M4 fidelity](images/gdn-m4-fidelity.png)

Both diagnostics use the original [256-query manifest](data/fidelity-samples.json),
64 queries each from GSM8K, HumanEval, HellaSwag and CMMLU Chinese history,
and 10,479 gold target tokens. The new M32 BF16 reference exactly matches
the archived reference; the archived FP8/NVFP4 query errors can be reused
without changing the reference. M32 uses eight cohorts plus a repeat; M4 uses
64 cohorts plus a repeat, with separately measured BF16 references. Every
scored step asserts its physical row count. The teacher hook changes only
selected tokens, reinstating the last prompt token in an unscored step and
then scoring frozen continuations from raw logits, with unscored tail padding.
Each query's mean absolute gold-logprob difference is weighted equally;
95% intervals use 20,000 query bootstrap resamples, seed 20260910.

Fidelity uses BF16 activations/KV, maximum context 768, 8192 prefill tokens,
3,489,660,928 KV bytes/rank, no prefix caching, V2 runner and TRITON_ATTN.
BF16 uses eager execution and 4 GiB CPU offload; stock quantized arms retain
compilation. Mach uses full decode graphs at 1/2/4/8/16/24/32. Full profile
owner-prefill admission remains 512–4096 rows; larger cohorts use its fallback.
Logprob requests use the full BF16 head rather than candidate search.

The separate same-hidden-state head probe with new full options reports
**11,996 eligible rows**, global BF16 top-20 recall
**100.00%** (239,920/239,920)
and final greedy top-1 agreement **100.00%**.
It generates 48 tokens for each of the same 256 prompt prefixes, excluding
ineligible calls. The two ranks see the same rows, so their denominators are
not added. These observations do not imply bitwise logit equality.

[Raw M32 fidelity and head observations](data/native-fidelity.json),
[raw M4 fidelity and dispatch counts](data/gdn-m4-fidelity.json), and
[kernel graph/eager, null-slot and slot-reuse validation](data/gdn-kernel-validation.json)
are retained. Dispatch counts describe host calls/captures, not graph replays.

## Fidelity and throughput

![Numerical fidelity and throughput](images/quality-throughput-tradeoff.png)

Horizontal bars are M32 MAE 95% intervals. Vertical bars span the four
throughput gains over stock FP8; they are not confidence intervals. M32
fidelity does not exercise persistent; consult the separate M4 plot.

## Reproduction

Install the [native profile](installation.md). Use clean official vLLM/FlashInfer
for BF16/FP8/NVFP4 and disable Mach registration and FlashInfer AllReduce.
Offline callable RPC serialization is for the local diagnostic only.

```bash
CUDA_VISIBLE_DEVICES=4,5 python tools/fidelity_native_mxfp6.py \
  --arm gdn --skip-head-probe --model /models/Qwen3.8-27B-MXFP6 \
  --tokenizer /models/Qwen3.8-27B-official \
  --manifest docs/data/fidelity-samples.json --output RESULTS/fidelity/gdn
```

Repeat for default, persistent, full, full_ba and full_gdn; in the stock environment
repeat for bf16. The collection command below reuses archived stock FP8/NVFP4
results; omit `--reuse-stock-baselines` and measure those arms only for a fresh
complete comparison. Keep default
stock compilation enabled. Repeat bf16/default/persistent/gdn/full_ba/full_gdn with
`--physical-rows 4` into `RESULTS/fidelity-m4/ARM`. Run `--arm full_gdn --head-only`
into `RESULTS/head` for the independent greedy head probe.

```bash
python tools/compare_native_serving.py \
  --arms persistent full_ba default full gdn full_gdn \
  --models /models --stock-runtime /path/to/stock/site-packages \
  --output RESULTS/serving --devices 6,7 \
  --prompt-manifest docs/data/serving-prompts.json --concurrencies 32 4 16 24
python docs/data/collect_native_comparison.py --results RESULTS --reuse-stock-baselines
python docs/data/collect_gdn_m4.py --results RESULTS/fidelity-m4
python docs/data/plot_comparison.py
python -m pytest -q
```

Model directory names are Qwen3.8-27B-MXFP6, Qwen3.8-27B-FP8-official,
Qwen3.8-27B-NVFP4 and Qwen3.8-27B-official (BF16/tokenizer).
Use a separate plotting environment with `docs/data/requirements-plot.txt`.
