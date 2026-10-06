# Qwen3.5-4B MXFP8 packaged qualification — October 6, 2026

The source-built, installed profile preserves the accepted champion on the tested workload. Both fixed-M precision records are byte-identical to the accepted records, and all 1040 generated output hashes match the previous complete serving screen. Rechecking the four historical-negative throughput points against the old implementation on the same GPU leaves differences from −0.081% to +0.255%; these measurements show no material packaging regression. They do not establish statistical equivalence.

The code can be built from Mach and the official mxfp6 repository with the declared public dependencies. Full reproduction also requires the exact L0 asset, which is identified but not publicly distributed. No model weights or evaluation requests are included in this report.

## Qualified revisions and build

- Mach runtime: `ace36a3db9e5ca1db422346c9c307f2aa840bcdd`. Later report-only commits do not change the qualified runtime.
- Official mxfp6: `15a56aa2774552d0584d8fc6b621b41dc173f30b`, merged in [PR #8](https://github.com/Nekofish-L/mxfp6_sm120/pull/8).
- Local qualified image ID: `sha256:be095a0ae485fdf625d6e8c9cf7b836b79978f9c6d3cc30e16c5dcba7f8164dd`. This identifies the tested build; it is not a downloadable registry reference.
- All 32 MXFP8 runtime package files match the measured image. Public base, native wheel/library and runtime source hashes are in the [machine-readable report](data/mxfp8-qualification-20261006.json).

The build uses the pinned public vLLM image, fresh official kernel/CUTLASS sources and the standard two-library wheel. It applies the bundled Mach source profile and launches the installed entry point. The tested checkpoint is reconstructed from the public BF16 revision plus the exact L0 asset; all 595 tensors match. No experiment runtime module, old binary or inherited compiler cache is needed. See [build, model preparation and serving commands](mxfp8-champion.md).

## Precision and startup

| Physical rows | Queries | Gold tokens | Changed logprobs vs accepted | MAE vs same-M BF16 |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 256 | 10479 | 0 | 0.049038113196 |
| 64 | 256 | 10479 | 0 | 0.053286047365 |

The raw record files match byte-for-byte at both M32 and M64; repeating the first cohort also gives zero difference. These are short-gold logprob checks with private fixed fixtures, not task accuracy or a universal BF16-equivalence claim. Quality mode uses its own declared small FULL_AND_PIECEWISE captures. The BF16 anchor uses FLASH_ATTN/BF16 KV; the champion uses FLASHINFER/FP8 KV.

Two production launches in the same run directory each rebuild all 23 compiler bindings and all 23 eligible QKVZ/BA parallel pairs for M32 and M64. Repeated requests match across launches. Each launch owns a new cache namespace to avoid cached AOT callables bypassing construction hooks. This preserves the selected arithmetic and has an explicit startup-compilation and cache-storage cost.

## Complete production screen

One RTX 5090, TP1, 3000 input / 1000 output tokens, fixed uniform token IDs and explicit request seeds. The six points contain 40/160/120/160/240/320 scored requests, all successful with exact server-reported lengths. Production retains 19 GiB KV, maxseq128, maxlen8192, maxbatch2048 and FULL/PW2048 routing.

| Concurrency | Packaged tok/s | Historical champion tok/s | Change | Historical community FP8 tok/s | Gain vs FP8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 880.131 | 881.119 | -0.112% | 683.959 | +28.682% |
| 16 | 2481.298 | 2493.357 | -0.484% | 2053.547 | +20.830% |
| 24 | 3189.417 | 3194.545 | -0.161% | 2559.262 | +24.623% |
| 32 | 3672.696 | 3667.963 | +0.129% | 2887.825 | +27.179% |
| 48 | 4168.038 | 4196.089 | -0.669% | 3220.549 | +29.420% |
| 64 | 4653.929 | 4634.680 | +0.415% | 3402.757 | +36.769% |

The complete screen remains a single run per implementation, with the historical screens measured October 5. All 1040 request output hashes equal the old champion; fixed-M precision is assessed separately. TTFT, TPOT, ITL and latency metrics are retained in the JSON. The whole-profile gain includes the model, head, KV, native kernels and runtime configuration; it is not a single-kernel gain.

The four points below were rerun on October 6 using the unchanged old champion, after the packaged screen. Its 86 source/extra-file hashes were verified before launch. The installed benchmark client used the same prompts, seeds, warmups and request counts. All 560 outputs again match.

| Concurrency | Packaged tok/s | Current old implementation tok/s | Packaged / current old − 1 |
| ---: | ---: | ---: | ---: |
| 4 | 880.131 | 880.446 | -0.036% |
| 16 | 2481.298 | 2482.384 | -0.044% |
| 24 | 3189.417 | 3181.296 | +0.255% |
| 48 | 4168.038 | 4171.411 | -0.081% |

The current-old results account for most of the small historical-negative differences. This bounded check supports proceeding with code integration without declaring a throughput improvement from packaging itself. It does not replace the six-point result or create an automatic regression allowance.

## Verification scope

- The combined CPU suite passed 136 tests and 50 subtests before the final launcher cache-namespace adjustment. All other runtime sources are unchanged. The final image then passed the targeted launcher suite: 11 tests and 28 subtests.
- Six new native GPU tests, three shapes × five prototype input cases with bitwise limbs/scales/output and graph replay, the existing MXFP6 GPU ops suite and three CTests passed. Official kernel source CI passed on Python 3.10 and 3.12.
- Final-image testing covers actual route/capture receipts, sequential cohorts and drains, two worker starts, fixed-M precision and the complete serving contract above. It is not an exhaustive state-transition or arbitrary workload/concurrency qualification.

The benchmark client is public and generates its workload locally. Authorized access to the [identified L0 input](mxfp8-model.md) is still needed to construct the checkpoint; the fixed-gold evaluation corpus is not bundled.
