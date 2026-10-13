# 4B FP8 KV ablation — 2026-10-13

The default remains FlashInfer E4M3 KV with the packaged scales. BF16 KV
and explicit scale overrides are supported; model identity checks remain
active, scale owners are installed before profiling, and actual KV storage
is verified after allocation. This report contains aggregate results only.
The supplied multimodal data, identifiers, fingerprints, local paths,
evaluation scripts and reproduction commands are not included.

## End-to-end throughput

One fresh server per policy, sequentially on the same RTX 5090, with the
production compact head, default FP32 ordered GDN, and experimental BF16
and PDL disabled. Each sixpoint suite completes 1,040 requests and 1,040,000
output tokens, with 3,000 input and 1,000 output tokens per request and
19 GiB KV cache. Input seeds and boot/prewarm/scored phases are identical.
Aggregation divides total scored output tokens by summed scored durations.
There is one full trial per policy; speed differences have no repeated-trial
significance estimate.

| KV policy | Aggregate output tok/s | vs BF16 KV |
| --- | ---: | ---: |
| BF16 KV | 3101.99 | +0.00% |
| FP8, independent random calibration | 3425.07 | +10.41% |
| FP8, all 1 | 3433.45 | +10.69% |
| FP8, historical bank | 3433.24 | +10.68% |

| Concurrency | BF16 KV | Random FP8 | All-1 FP8 | Historical FP8 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 899.4 | 919.6 | 920.8 | 919.9 |
| 16 | 2417.6 | 2609.1 | 2616.0 | 2614.5 |
| 24 | 3057.4 | 3357.3 | 3385.4 | 3378.4 |
| 32 | 3411.0 | 3828.0 | 3836.5 | 3835.3 |
| 48 | 3882.6 | 4360.2 | 4367.5 | 4366.7 |
| 64 | 4175.4 | 4831.0 | 4833.9 | 4845.5 |

All-1 and historical-bank throughput differs by less than 0.01% in this
single suite. The measured FP8 benefit is available without the historical
calibration corpus. Default dtype and scale selection remain unchanged.
These throughput results use the production text workload; they do not
measure multimodal serving throughput.

## Text numerical fidelity

The reference has the **same MXFP8 language weights and FP32 SSM state**,
changing only KV to BF16. Quality uses fixed physical M32, full-head raw
gold logprobs and small FULL_AND_PIECEWISE graphs. First-cohort repeats
are exact for every policy. This measures numerical fidelity, not task accuracy.

Three independent random calibration corpora were frozen before evaluation,
with 192 requests of 512 uniform non-special tokenizer IDs per seed. The
fixed scale rule is `float32(1.1 * post_RoPE_K_or_V_absmax / 448)`, with zero
mapping to 1. Random inputs calibrate scales; numeric scale values are not
sampled arbitrarily. The random result averages the errors of all three
banks without choosing the best seed. Historical calibration contains
61,653 input tokens versus 98,304 per random corpus. This budget difference
limits attribution to content distribution alone. All-1 uses no calibration.
The historical bank is included only as a separate leakage diagnostic.

MAE is absolute raw gold logprob error in nats/token; lower is better.
Intervals use 20,000 paired whole-request bootstrap draws. 99% intervals
are per comparison; secondary comparisons are exploratory.

| Evaluation | Requests / gold tokens | Random, mean of 3 | All 1 | Historical | Historical − random, 99% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| Original four-domain fixture | 256 / 10,479 | 0.042224 | 0.042460 | 0.040420 | −0.001804 [−0.003325, −0.000318] |
| Unused GSM8K/HumanEval sources | 128 / 5,500 | 0.037101 | 0.036240 | 0.035631 | −0.001471 [−0.003793, +0.000823] |
| Independent random tokens | 128 / 6,144 | 0.156313 | 0.153230 | 0.153147 | −0.003166 [−0.007683, +0.001375] |

Random calibration does not significantly outperform all-1 at the 99%
interval level in these groups. Fresh prompts retain the last 192 tokens,
with at most 48 gold tokens. The original fixture is unchanged. The former
CSV text slice is excluded; it is not a valid multimodal evaluation.

## Matching calibration lengths and budget

A supplementary control preserves the historical calibration's 192 training
input lengths and 61,653-token budget, replacing contents with independently
seeded uniform token IDs. It reads no evaluation content, keeps the fixed
formula and all three seeds, and does not select scales using scores. This
control was added after detecting the budget difference and is exploratory.

| Evaluation | Matched random MAE, mean of 3 | Historical − matched random, 99% CI |
| --- | ---: | --- |
| Original fixture | 0.041828 | −0.001408 [−0.002891, −0.000007] |
| Fresh sources | 0.036540 | −0.000910 [−0.003205, +0.001525] |
| Independent random tokens | 0.154631 | −0.001484 [−0.006195, +0.003344] |

Random vs all-1 99% intervals still include zero in all three groups.
The historical advantage is 3.37% on the original fixture and 2.49% on fresh
sources; the original-vs-fresh interaction CI is [−8.28, +6.03] percentage
points. The original-vs-random-token interaction also includes zero.

## Actual multimodal fidelity with unchanged MXFP8 language weights

The production checkpoint has no vision tower. This experiment retains its
595 language tensors unchanged and adds the same source model's 297 BF16
vision tensors. Runtime checks confirm 112 MXFP8 language linear modules,
BF16 vision parameters and eight correctly configured KV attention layers.
It uses standard vLLM eager FlashInfer, physical M1 and FP32 SSM for every
policy. It does not qualify the production Mach profile for multimodal serving.

The 128 fixed requests retain all **392 images and 640 preceding messages**.
Images are not manually resized; identical processor pixel limits of
65,536–262,144 apply to every policy. Prompt lengths are 793–6,523 tokens,
302,251 total; each request scores 48 gold tokens, 6,144 total. There are
390 distinct images, with no image shared across selected requests.

**Answer leakage was found in the input format:** every selected request
included its target output as the final assistant reply. That reply is now
excluded before inference. All images remain, and no complete target occurs
in the remaining history. The former CSV text-slice and full-BF16 multimodal
transfer results are invalidated and removed from the report. This input
format issue does not establish that KV scales fitted the supplied answers.

A 256-request original-text control runs in the **same multimodal engine**,
with identical language weights, backend, physical batch size and scale
policies. This avoids attributing a change of execution mode to distribution.
Both groups use the three already frozen, matched-length random banks;
no supplied multimodal data fits scales. The reference changes only KV to
BF16. First-request repeats are exact in both groups for all six policies.

| Same-engine evaluation | Random, mean of 3 | All 1 | Historical | Historical − random, 99% CI |
| --- | ---: | ---: | ---: | --- |
| Original text: 256 / 10,479 gold tokens | 0.048967 | 0.049776 | 0.048945 | −0.000022 [−0.001987, +0.001967] |
| Multimodal: 128 / 6,144 gold tokens | 0.051316 | 0.052377 | 0.052422 | +0.001106 [−0.001570, +0.003747] |

On multimodal requests, random minus all-1 MAE is −0.001061, with 99% CI
[−0.003713, +0.001599]. No scale policy has a statistically established
advantage over the others in this comparison.

The historical bank reduces original-text MAE by 0.045% and increases
multimodal MAE by 2.154% relative to the random mean. Their relative-change
interaction is **−2.199 percentage points, 99% CI [−8.736, +4.460]**.
Negative values mean a stronger original-set advantage. The interval includes
zero, so this diversity comparison does not establish original-specific
overfitting. Prompt lengths and content differ between the groups; even a
significant interaction would not isolate content distribution alone.

## Leakage assessment

No direct calibration overlap was observed in the audited source-ID and
token boundaries of the original/fresh/random-token text groups. Historical
training excluded its holdout and the frozen evaluations. These checks
cannot exclude shared source-family structure or past experiment selection.

The historical bank has a modest advantage on the production text fixture,
but its advantage is not significantly larger than on fresh or random-token
inputs. The corrected same-engine multimodal comparison also finds no
statistically established original-specific advantage. The former CSV
results cannot support this assessment because the answer was in the prompt.

**The measurements do not establish KV-scale data leakage or a significant
original-distribution preference.** They cannot prove the absence of either
or assign a numerical leakage probability. Use independently random-calibrated
or all-1 scales for comparisons that must not rely on historical calibration.

The [aggregate result report](data/mxfp8-kv-ablation-20261013.json) contains
measurements and statistical intervals only; no supplied dataset or multimodal
reproduction artifacts are stored in the repository.
