# Native MXFP6 fidelity and serving comparison

## Fidelity protocol

The September 15 rerun retains the original test logic: 256 fixed queries,
64 each from GSM8K, HumanEval, HellaSwag and CMMLU Chinese history;
10,479 gold target tokens; eight physical-M32 cohorts plus a repeat of cohort zero.
The [token manifest](data/fidelity-samples.json) contains the exact inputs, without
retokenization or a new chat template. Dataset revisions remain in its sources.

The teacher hook changes only token selection. The submitted prompt drops its
last token; an unscored forced output reinstates it. Gold continuations are then
scored from unchanged raw logits. Unscored padding keeps all 32 requests at the
same decode offset. Scheduling is paused only during enqueue, then resumed;
worker observations assert physical M32 for every scored step.

For each query, average the absolute difference in gold-token logprob against
fresh BF16. Average those 256 query errors equally. Intervals use 20,000 query
bootstrap resamples with seed 20260910. This measures numerical fidelity,
not benchmark task accuracy. No file or tensor fingerprint calculations are used.

All five arms (including BF16) reproduced cohort zero's gold logprobs exactly. The paired full
minus default MAE difference is −0.001020, with 95% interval
[−0.003602, +0.001559]; it does not establish a fidelity improvement.

| Configuration | MAE | 95% query-bootstrap interval |
|---|---:|---:|
| FP8 · stock vLLM 0.29 | 0.061430 | 0.054315–0.069084 |
| MXFP6 · Mach default | 0.090643 | 0.082647–0.098954 |
| MXFP6 · Mach full options | 0.089624 | 0.082036–0.097311 |
| NVFP4 · stock vLLM 0.29 | 0.170886 | 0.155098–0.187517 |

Settings: TP2, BF16 activations/KV, TRITON_ATTN, V2 runner, maximum sequence
length 768, 32 requests, prefill budget 8192, 3,489,660,928 KV bytes per rank,
no prefix caching. BF16 uses eager execution and 4 GiB CPU weight offload;
MXFP6 arms use `FULL_DECODE_ONLY` graphs without compilation. Stock FP8/NVFP4
retain default compilation and graph mode; all use capture sizes
1/2/4/8/16/24/32. All scored
logprobs remain on the ordinary raw-logprob path. FP8/NVFP4/BF16 use unpatched
official vLLM 0.29.0; Mach is not registered for those arms. Mach default includes
native kernel/workspace integration, fused AR/Norm and
compact greedy sampling; full also enables FP16 SSM, lossless/owner prefill
and the NVFP4 head option. The latter falls back to BF16 for logprob requests.

The full profile's owner-prefill eligibility is unchanged: 512–4096 rows. The
4343-row cohort takes the ordinary fallback; the diagnostic does not force
owner execution outside its supported range. All other initial cohort token
counts are between 1676 and 3948. Fidelity was run on GPUs 4/5; the head probe
on GPUs 6/7. Other GPU work can coexist, so these are not throughput timings.

## NVFP4 head accuracy

The actual greedy head was tested separately on the same 256 prompt prefixes,
generating 48 tokens per request. At each eligible head call, compare against
the full BF16 head on **the same hidden states**. This isolates candidate search
from backbone/state precision. Initial prefills and ineligible calls retain
their normal fallback and are excluded from the candidate-search denominator.

- 11,998 eligible rows; 128 initial candidates per TP rank.
- Global BF16 top-20: 239,960 / 239,960 retained (**100%**).
- Every tested row retained all 20 tokens; zero missing global winners.
- Final greedy top-1: 11,998 / 11,998 agreement (**100%**).
- Selected BF16 logits are not bitwise identical to the full GEMM: maximum
  absolute difference 0.125, with 1407/1741 differing selected values on ranks 0/1.
  This did not change the tested top-1 decisions.

Global top-20 is formed from both ranks' local top-20 values, then checked
against the corresponding rank's candidates. The two ranks observe the same
global rows; they are not counted as twice as many independent samples.
These observed recall/agreement results are not universal guarantees.

## Serving baseline correction

The historical NVFP4 result included patched FlashInfer AllReduce on RTX 5090
and must not be presented as stock NVFP4. Its original data is retained as an
accelerated historical configuration in [the archive](benchmarks.md).

The first new short sweep forced `mode=NONE`, `FULL_DECODE_ONLY`, 32 maximum
sequences and fixed KV allocation. It measured 1268.8 token/s at c32, but this
does not match the user's ordinary open-source launch configuration (1578.67
token/s, 160 requests). **That sweep is not the final stock comparison.**
Using the user's local server and benchmark-client commands with unpatched
vLLM reproduced **1590.95 token/s**, 160 successful requests and 18.66 ms mean
TPOT. The server selected CUSTOM/PYNCCL communication, not FlashInfer AllReduce.
This reproduction uses the original client's decoded-text prompts and
retokenized output counts (159,996 tokens); its
[configuration and result excerpt](data/nvfp4-stock-reproduction.txt) are retained.

The final comparison freezes 160 ShareGPT prompt prefixes in the
[serving manifest](data/serving-prompts.json). Each prefix is repeated/truncated
to exactly 3000 token IDs and submitted directly, avoiding text retokenization.
Every request generates exactly 1000 tokens, checked using the server's usage
counts. All arms share the same prompts, request seeds and exponential arrival
schedule (100 requests/s), with greedy sampling, top-k 20, top-p 0.95 and EOS
ignored. Each c4/c16/c24/c32 point measures 20/80/120/160 requests after
min(32, request count) warmups of 128 output tokens at that concurrency.

Stock FP8/NVFP4 retain default compilation/graph settings, 64 maximum sequences
and 90% GPU memory utilization. Both import unpatched official vLLM/FlashInfer,
disable Mach registration and explicitly disable FlashInfer AllReduce.
The two MXFP6 arms use the native profile's 32-sequence, non-compiled
`FULL_DECODE_ONLY` configuration and equal KV allocation of 8,218,214,400 bytes
per rank. All arms use TP2, TRITON_ATTN, no prefix caching, maximum model length
16,384 and a 4096-token prefill budget, on the same RTX 5090 pair (GPUs 6/7).
This is a comparison of deployable profiles, not an isolated quantization-only
ablation. Default/full MXFP6 use common memory and graph settings; only the
full profile enables the optional state, prefill and head optimizations.

These are short, single-run sweeps, not sustained-load estimates or throughput
confidence intervals. The final stock NVFP4 c32 result is **1607.56 token/s**;
the difference from 1590.95 reflects the frozen-token client/workload and run
variation, not enabling FlashInfer AllReduce.

Output tokens/s:

| Configuration | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| FP8 · stock vLLM 0.29 | 266.97 | 806.06 | 1018.55 | 1160.44 |
| MXFP6 · Mach default | 325.85 | 977.78 | 1255.42 | 1422.81 |
| MXFP6 · Mach full options | 353.10 | 1098.74 | 1422.61 | 1646.61 |
| NVFP4 · stock vLLM 0.29 | 378.97 | 1142.87 | 1424.61 | 1607.56 |

Mean TPOT, milliseconds:

| Configuration | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| FP8 · stock vLLM 0.29 | 14.29 | 18.35 | 21.69 | 25.41 |
| MXFP6 · Mach default | 11.74 | 15.18 | 17.65 | 20.79 |
| MXFP6 · Mach full options | 10.88 | 13.59 | 15.67 | 18.04 |
| NVFP4 · stock vLLM 0.29 | 10.13 | 13.06 | 15.67 | 18.55 |

All **1520/1520** scored requests completed with exactly 3000 input and 1000
output tokens. The equal-weight mean gains over stock FP8 are **22.31%** for
Mach default, **37.53%** for Mach full and **40.53%** for stock NVFP4.
The tradeoff plot's vertical bars span the four gains; they are not confidence
intervals. Full MXFP6 has lower measured logprob error than NVFP4, but does not
have higher throughput at every concurrency.

[Raw request counts, latencies, throughput, warmup settings and launch configurations](data/native-serving.json)
support the serving tables. Only the default and full MXFP6 profiles are retained.

## Reproduction

Install the [current native profile](installation.md). Use a separate clean
official vLLM 0.29 environment for BF16/FP8/NVFP4, with no runtime patches and
`VLLM_ALLREDUCE_USE_FLASHINFER=0`. The offline diagnostic needs access to this
checkout's Python package and teacher hook, but disables the Mach plugin for
those stock arms. It enables local callable RPC serialization only inside the
offline process; do not enable this on a public service.

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/fidelity_native_mxfp6.py \
  --arm default --model /models/Qwen3.8-27B-MXFP6 \
  --tokenizer /models/Qwen3.8-27B-official \
  --manifest docs/data/fidelity-samples.json --output RESULTS/fidelity/default
```

Repeat with `--arm full` and, in the unpatched environment, `bf16`,
`fp8`, `nvfp4` with their respective checkpoints and output directories
`RESULTS/fidelity/ARM`. Each output directory must
be new. The full arm also runs the head probe; run `--arm full --head-only`
with output `RESULTS/head` to reproduce the separate head measurement.

Stock compiler settings matter: the initial no-compilation diagnostic measured
0.051708 MAE for FP8 and 0.165461 for NVFP4. The plotted points above were rerun
with stock compilation enabled to match the throughput profiles. Use
`--no-stock-compile` only to reproduce that diagnostic ablation, not as the
stock serving baseline.

```bash
python docs/data/collect_native_comparison.py --results RESULTS --fidelity-only
python docs/data/plot_comparison.py --accuracy-only
python -m pytest tests/native_mxfp6/test_fidelity.py -q
```

For serving, supply a directory containing the clean official `vllm/` and
`flashinfer/` packages from a separate stock environment. Keep its native
dependencies compatible with the current environment; do not point this at
the patched package directory.

```bash
python tools/compare_native_serving.py \
  --arms nvfp4 fp8 default full \
  --models /models --stock-runtime /path/to/stock/site-packages \
  --output RESULTS/serving --devices 0,1 \
  --prompt-manifest docs/data/serving-prompts.json --concurrencies 32 4 16 24
python docs/data/collect_native_comparison.py --results RESULTS
python docs/data/plot_comparison.py
```

The model root contains `Qwen3.8-27B-MXFP6`, `Qwen3.8-27B-FP8-official` and
`Qwen3.8-27B-NVFP4`. Plotting requires NumPy and Matplotlib; install plotting
dependencies from `docs/data/requirements-plot.txt` in a separate environment
to avoid changing inference dependencies.

[Raw per-query logprobs, bootstrap inputs, runtime settings and head observations](data/native-fidelity.json)
are sufficient to recompute every published fidelity number.
