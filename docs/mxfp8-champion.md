# Qwen3.5-4B MXFP8 champion profile

`qwen35-4b-mxfp8-champion-v1` serves Qwen3.5-4B on one RTX 5090 (SM120), Linux x86-64 and Python 3.12. Download the L0 model asset from the [versioned release](https://github.com/troycheng/vllm-mach/releases/tag/qwen35-4b-mxfp8-assets-v1), then follow the [model preparation guide](mxfp8-model.md).

## Required parts

The profile combines the following optimizations:

| Part | Required behavior |
| --- | --- |
| 1. Native MXFP8 | Official mxfp6 loader, dispatcher and prepared workspaces; native policy `all` |
| 2. M32 MLP dual activation | All 32 gate/up layers, `M32/N18432/K2560`; high plus BF16-rounded residual, separate FP32 accumulators and final BF16 rounding |
| 3. Dual QKVZ | All 24 linear-attention projections, M32/M64 and `N12288/K2560` |
| 4. M64 QKVZ pipereg | The selected cfg2 register-pipeline variant; other shapes retain their declared routes |
| 5. Small-M BA | BF16 GEMV for M1–8, original fallback elsewhere; standalone `b12x==1.2.6` |
| 6. Ordered-W4 GDN | FP32 state; M1/2/8 full-store, M4/16 deferred, M24–128 step8 deferred; materialization/reset/slot ownership and fallback lifecycle |
| 7. Full-vocabulary head | NVFP4 coarse scores, `topk(sorted=False)` K2048, indexed BF16 score replacement; original sampler and tied BF16 head |
| 8. FlashInfer FP8 KV | Eight layers, E4M3 storage, 16 fixed FP32 scales and device/CPU/float mirror installation before profiling |
| 9. Graph policy | FULL small decode, exact2048 PIECEWISE, other large tails routed before padding |
| 10. Parallel QKVZ/BA | Closed fork/join at M1/2/4/8/16/24/32/64 for linear-attention ordinals 1–23; ordinal0 serial |
| 11. Compile choices | Production 23, quality M32 23 and quality M64 22 fixed choices, selected by the declared mode and rebuilt from source; no inherited compiler cache |
| 12. Fixed model | BF16 revision `851bf6…` source, exact L0 codes, 595 verified output tensors, tokenizer/config and KV metadata |
| 13. Build/run/measurement | Official source wheel, bundled Mach runtime installer, unified lifecycle/receipts and fixed six-point client |

The [FP8 KV ablation](mxfp8-kv.md) compares BF16 KV, independent random-token calibration, all-1 scales and the historical bank, including distribution checks and a multimodal comparison with identical MXFP8 language weights.

FP4 candidate scoring and FP8 KV reduce precision; measured BF16 error is listed in [Validation](#validation). The FlashInfer `b12x` head backend and the standalone BA package are separate implementations.

## Build, prepare, serve, benchmark

Use Mach runtime commit `ace36a3db9e5ca1db422346c9c307f2aa840bcdd` or a later revision documented to contain the same qualified runtime. The Dockerfile pins a public vLLM base
and requires the **full official kernel commit containing all three dual
variants** pinned below. The version label `mxfp6-sm120==0.2.1` alone does
not identify that source revision.

```bash
docker buildx build --load -f deploy/Dockerfile.mxfp8 \
  --build-arg MAX_JOBS=2 -t vllm-mach:mxfp8-champion .

docker run --rm --entrypoint vllm-mach-mxfp8-prepare \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v /path/to/bf16:/input/bf16:ro \
  -v /path/to/qwen35-4b-mxfp8-asym-l0-v1.safetensors:/input/l0.safetensors:ro \
  -v /path/to/output:/output vllm-mach:mxfp8-champion \
  --bf16 /input/bf16 --l0-codes /input/l0.safetensors --output /output/model

docker run -d --name mach-mxfp8 --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -v /path/to/output/model:/models/champion:ro -v /path/to/runs:/runs \
  vllm-mach:mxfp8-champion /models/champion --host 0.0.0.0 --run-dir /runs/serve

# Wait for service readiness before starting the client.
docker run --rm --network host --entrypoint vllm-mach-mxfp8-bench \
  -v /path/to/runs:/runs vllm-mach:mxfp8-champion \
  --base-url http://127.0.0.1:8000 --model q35-mx8-study --outdir /runs/sixpoint
```

Create the host output/run parents first; `/output/model` and `/runs/sixpoint`
must be new. Download the versioned L0 asset and check its [exact SHA-256](mxfp8-model.md#download-the-l0-asset) before model preparation.
The image builds both official kernel libraries from source and applies Mach's bundled source profile.

For a faster package mirror, set Docker build argument `PIP_INDEX_URL` to its
HTTPS simple index. Both build-tool wheels are verified against fixed PyPI
SHA256 hashes, regardless of the download mirror.

Pinned runtime: vLLM 0.29.0, Torch 2.13.0, Triton 3.7.1, FlashInfer Python/cubin
0.6.18, CUTLASS DSL 4.6.2, CUDA bindings 13.3.1 and standalone b12x 1.2.6.
CUDA 13.0/CUTLASS source pins and build details are recorded by the build tool.
See [runtime installation](mxfp8-installation.md) for source guards and the
independent installer; ordinary Mach MXFP6 patches are not a substitute.

## Configuration and receipts

The launcher defaults to `--kv-cache-dtype fp8_e4m3`; omitting the flag does not select BF16 KV. Use `--kv-cache-dtype bfloat16` explicitly for the unquantized KV baseline. Stock vLLM defaults to `auto`, which follows the model dtype.

Production defaults are BF16 model activations, FP32 SSM, FP8 E4M3 KV,
19 GiB KV capacity, maxseq128, maxlen8192 and maxbatch2048. Captures are
`[1,2,4,8..256 step8,2048]`: FULL through128 and PIECEWISE descriptors through
2048. Exact2048 uses PIECEWISE; other tails above256 use NONE without padding.
The launcher enables a K128 candidate/BF16-refinement greedy path through M64;
`--no-hybrid-head` keeps the complete K2048 head for all calls. Score requests,
constrained requests and unsupported cases use the complete head. The same `--hybrid-head` switch also admits unseeded top-k decode, including optional top-p and presence-only penalties; other sampling and penalty combinations use the complete head.
SwiGLU and MXFP8 quantization are fused at supported small row counts.
See the October 12 [optimization validation](mxfp8-benchmark.md).
Prefix caching, speculative decode, generic compact-head hooks,
vocab-parallel greedy and unrelated Dense/MoE/AR modes are disabled by the profile.

Add `--ssm-dtype bfloat16` to select BF16 state with ordered W4 GDN, now the default backend for both state dtypes. FP32 remains the default storage dtype. Use `--no-ordered-bf16-ssm` to select stock GDN/FLA for BF16 state. Grouped BF16 GDN and GDN PDL remain disabled by default.
See [BF16 SSM measurements and configuration](mxfp8-benchmark.md).

`--quality-rows 32` or `64` selects a separate precision-only mode:

| Mode | Maxseq / maxlen / maxbatch | KV | FULL captures | PW |
| --- | --- | --- | --- | --- |
| Quality M32 | 32 / 1024 / 8192 | 4 GiB | `[4,32]` | `[4,32]` |
| Quality M64 | 64 / 1024 / 16384 | 19 GiB | `[4,32,64]` | `[4,32,64]` |

Both quality modes resolve to `FULL_AND_PIECEWISE`; PIECEWISE covers only the
same small capture sizes, with no PW2048. Authorized fixed-gold fixtures use
256 queries/10479 gold tokens per M, whole-cohort queuing, raw decode logprobs and one repeat of the first cohort.
Use production mode for throughput measurements.

Keep these receipts with any qualification result:

- `/opt/mach/mxfp8-build.json`: official kernel/CUTLASS revisions, patched
  source hashes, toolchain, wheel/library hashes and registered schemas.
- `/opt/mach/runtime-source-install.json`: complete installed source profile.
- Model `mach_materialization_manifest.json` and `mach_profile.json`: full
  tensor/file identities, asset provenance and scale bindings.
- Run `launch.json` and `worker-*.json`: actual mode, versions, source guards,
  native/dual/BA/GDN/head/KV/graph/parallel/compile preparation. Python and graph
  construction counters do not count CUDA graph replay.
- Benchmark `sixpoint.json` and raw phase files: all six points, zero failures,
  1040 scored requests and 1,040,000 scored output tokens.

Each launcher invocation creates an independent `cache/startup-*` directory
inside its run directory. It preserves the original AOT compilation mode and
enables the caches required by Torch precompilation, but never restores a
previous process's vLLM/Torch graph artifacts. Both compiler recipes and
projection rewrites therefore run and are checked on every startup. Reusing a
whole saved callable bypasses these hooks in the pinned stack; globally forcing
Torch caches off is incompatible with its AOT precompiler. This design costs
startup compilation time and retained cache storage. After stopping all workers
that use a run directory, its old startup-cache subdirectories may be removed.

## Performance

See the [single MXFP8 benchmark](mxfp8-benchmark.md#qwen35-4b) for the default FP32 SSM/FP8 KV configuration, the FP32 SSM/BF16 KV baseline and optional BF16 SSM configurations, each compared with historical open-source block FP8.

## Validation

The [current precision table](performance.md#qwen35-4b-precision) reports October 20 M32/M64 measurements for FP32 SSM and optional BF16 SSM, using Mach `afcb4cf` and native `ce0cd05`. Each configuration covers 256 queries and 10,479 fixed gold tokens. The [data](data/mxfp8-fidelity-20261020.json) include query-level errors, confidence intervals, repeated-cohort checks and raw logprobs with runtime receipts.

The BF16 reference uses FLASH_ATTN and BF16 KV at each matching physical shape; this profile uses FLASHINFER and FP8 KV. Fixed-gold requests use the existing full-vocabulary NVFP4 head with K2048 BF16 score refinement. Hybrid candidate selection has separate [generation and sampling validation](lm-head-sampling.md).

The [fixed token manifest](data/fidelity-samples.json) and validation tool are public; see [reproducing the precision measurements](performance.md#reproduce-mxfp8-fidelity). Earlier release qualification remains in `historical_validation.four_b` in the [benchmark data](data/mxfp8-benchmark.json).
