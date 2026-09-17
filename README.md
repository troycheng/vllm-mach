<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/vllm-mach-horizontal-dark.png">
    <img src="assets/logo/vllm-mach-horizontal-light.png" alt="vLLM Mach" width="720">
  </picture>
</p>

<p align="center">Native MXFP6 runtime paths for vLLM</p>

<p align="center">
  <a href="https://github.com/troycheng/vllm-mach/releases"><img alt="Release" src="https://img.shields.io/github/v/release/troycheng/vllm-mach?include_prereleases&sort=semver"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.29.0-6C5CE7">
</p>

vLLM Mach adds accelerated native MXFP6 execution paths to vLLM. Its model-specific profiles target Qwen3.8-27B Dense and Qwen3.5-35B-A3B MoE on vLLM 0.29.0. Mach integrates checkpoint loading, tensor-parallel execution, fused communication and CUDA Graph workspace management. Native MXFP6 kernels are provided by [`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120). The current wheel no longer registers or ships the EXL3 provider; earlier results remain in the [benchmark archive](docs/benchmarks.md).

## Why Mach

Mach targets fast serving at small and medium batch sizes, with most tuning focused on 4 to 32 concurrent requests. Its native MXFP6 profile combines packed weights with fused tensor-parallel communication and CUDA Graph support. Dense owner-prefill and optional LM head paths further accelerate prefill and eligible greedy decode while retaining vLLM's serving interface.

The aim is higher throughput with controlled numerical error. FP16 recurrent state and NVFP4 LM head candidate search remain explicit options; the default profile retains FP32 recurrent state and the BF16 LM head.

## Support

| Path | Validated configuration |
|---|---|
| Native MXFP6 | vLLM `0.29.0`; Qwen3.8-27B-MXFP6; `mxfp6-sm120==0.2.1`; TP2/PP1; SM120; BF16 activations/KV; text-only, non-speculative decoding |
| Native MXFP6 MoE | Qwen3.5-35B-A3B-MXFP6; TP2/PP1; SM120; [setup and kernel revision](docs/qwen35-moe.md) |
| MXFP6 CUDA Graph | V2 runner; Dense uses `FULL_DECODE_ONLY` at `1, 2, 4, 8, 16, 24, 32`; MoE inherits vLLM's default graph configuration |
| Fused FlashInfer collective | The native configuration above with FlashInfer `0.6.18` and the matching runtime/local IPC patches |
| Lossless / owner prefill | Enabled by default for Dense; opt out with `--no-lossless-prefill` / `--no-owner-prefill`; requires matching [lossless](native/lossless_prefill/README.md) and [owner](native/owner_prefill/README.md) extensions |
| FP16 SSM / NVFP4 LM head | Optional `--fp16-ssm` / `--nvfp4-lm-head`; head uses FlashInfer's built-in B12X backend, without the standalone `b12x` package; changes numerical behavior |

See the [current source installation](docs/installation.md) for native dependencies and [native MXFP6 integration](docs/native-mxfp6.md) for request eligibility and fallback behavior. Earlier EXL3 and checkpoint-hybrid releases are documented in [earlier installations](docs/public-install.md) and the [0.29 dependency upgrade](docs/dependency-upgrade.md).

## Performance

### 3k/1k reference comparison

#### Qwen3.8-27B Dense

Qwen3.8-27B, two RTX 5090 GPUs, TP2, 3000 input / 1000 output tokens, default remeasured September 17, 2026 with lossless/owner prefill; full retains September 16 data, and stock FP8/NVFP4 retain September 15 data. Frozen prompts, 20/80/120/160 requests at c4/c16/c24/c32, per-point warmups. These are short, single-run output-throughput measurements.

![Serving throughput](docs/images/throughput-comparison.png)

| Configuration | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| FP8 · vLLM 0.29.0 baseline | 267.0 | 806.1 | 1018.5 | 1160.4 |
| NVFP4 · vLLM 0.29.0 baseline | 379.0 | 1142.9 | 1424.6 | 1607.6 |
| MXFP6 · Mach default | 377.2 | 1043.1 | 1352.5 | 1537.0 |
| MXFP6 · Mach full | 384.9 | 1124.3 | 1453.6 | 1674.8 |

The new default improves throughput over stock FP8 by **34.0%**, and the new full profile by **42.7%**, weighting the four concurrency levels equally.

Default enables both GDN routes and lossless/owner prefill. Full additionally enables FP16 SSM and NVFP4 head search; both state dtypes use persistent at small batches. Stock FP8/NVFP4 retain default compilation and disable FlashInfer AllReduce. Mach retains its decode graphs and fixed KV allocation. This compares deployable profiles; [exact settings, isolated comparisons and raw results](docs/native-fidelity.md) are retained.

#### Qwen3.5-35B-A3B MoE

Qwen3.5-35B-A3B, measured September 17, 2026; full remeasured with FP16 SSM. Each point is a **two-run mean**, using 3000 input / 1000 output tokens, 16/32/48/64 scored requests at c4/c16/c24/c32, and a concurrency-sized 128-output-token warmup. All four profiles use identical uniform token IDs and seeds. Mach runs on two RTX 5090 GPUs with TP2. The FP8 and NVFP4 baselines use open-source vLLM 0.29.0. The 27B comparison above uses ShareGPT prompts.

![Qwen3.5-35B-A3B MoE serving throughput](docs/images/qwen35-moe-throughput.png)

| Configuration (output tokens/s) | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| FP8 · vLLM 0.29.0 baseline | 679.9 | 1471.6 | 1774.6 | 1962.1 |
| NVFP4 · vLLM 0.29.0 baseline | 730.6 | 1654.2 | 1985.3 | 2202.1 |
| MXFP6 · Mach default | 1010.2 | 2324.2 | 2869.5 | 3237.6 |
| MXFP6 · Mach full | 1094.8 | 2493.8 | 3070.6 | 3417.9 |

The default profile improves throughput over the remeasured FP8 baseline by **58.3%**, and the full profile by **69.4%**, weighting the four concurrency levels equally.

Mach default and full now use `VLLM_COMPILE` / `FULL_AND_PIECEWISE`, the expanded capture sizes through 2048, `--max-num-batched-tokens 2048` and `--max-num-seqs 64`. Both retain native MoE schedules, GDN decode, hand-written AllReduce/residual/RMSNorm fusion and an 8 GiB/rank KV budget. Full enables `--fp16-ssm --nvfp4-lm-head`: FP16 recurrent state and NVFP4 candidate search with BF16 refinement. Default retains FP32 recurrent state and the BF16 head. Owner/lossless prefill remain Dense-only.

The profile update passed **131 focused tests** and 700 scored benchmark requests. The MoE FP16 diagnostic verified FP16 SSM and BF16 convolution state at all 30 GDN layers per rank, with all final top-1 outputs matching the same-hidden-state BF16 head across 1638 decode positions. [FP16 retest and validation](docs/qwen35-moe.md#fp16-ssm-full-profile-retest).

The earlier graph/fusion update passed **50 focused tests** and **5088 measured requests**. Earlier LM-head validation covered **1640 real decode positions**, retaining all global BF16 top-20 candidates and matching every final BF16 top-1. Candidate search and fused reduction can change numerical behavior; these checks do not establish broad model quality or BF16 fidelity.

[Deployment and graph/fusion measurements](docs/qwen35-moe.md#default-compilation-and-prefill-graphs) · [Chart data and per-request timings](docs/data/qwen35-default-full-20260917.json) · [Baseline protocol](docs/qwen35-moe.md#updated-defaultfull-chart-and-user-provided-baselines). Two repetitions do not establish confidence intervals.

### Numerical fidelity

Current Dense and MoE default/full profiles, measured September 17, 2026. Each configuration scores 256 fixed queries and 10,479 gold tokens against its own model's BF16 reference. Lower MAE is better; the horizontal axis starts at zero, with Mach default shown above Mach full. Whiskers show 95% query-bootstrap intervals. MoE NVFP4 varies across repeated inputs; these intervals do not include run-to-run variation. FP8 and NVFP4 use open-source vLLM 0.29.0.

![Dense and MoE numerical fidelity at physical M32](docs/images/accuracy-comparison.png)

At physical M32, Dense default/full MAE is **0.09064 / 0.08962**; MoE default/full is **0.07261 / 0.07409**. Both paired full-minus-default confidence intervals include zero. These results do not establish that either profile is more accurate. MAE measures numerical deviation, not task accuracy.

Default retains FP32 SSM; full adds FP16 SSM and NVFP4 head search. Persistent GDN can change arithmetic even with FP32 state, so default is not a claim of bitwise equivalence. Logprob requests use the BF16 head and do not exercise candidate search. [M4 results, exact configurations and raw data](docs/profile-fidelity-20260917.md).

The separate historical NVFP4 head probe retains **100% global BF16 top-20 recall** and **100% final top-1 agreement** across 11,996 eligible Dense rows. [Head-probe protocol](docs/native-fidelity.md).

### Fidelity and throughput

Each Dense point combines the September 17 M32 fidelity measurement with the current profile's equally weighted mean throughput gain over stock FP8. Horizontal bars show MAE 95% intervals; vertical bars span gains across c4/c16/c24/c32. Throughput measurement dates and conditions are listed above.

![Fidelity and throughput](docs/images/quality-throughput-tradeoff.png)

[September 16 ablations](docs/native-fidelity.md), [September 15 native results](docs/native-fidelity-20260915.md) and [earlier EXL3 results](docs/benchmarks.md) remain archived.

### Deployment tradeoffs

The fastest profiles require matching native extensions and pinned vLLM/FlashInfer patches, so deployment takes more setup than a stock vLLM installation. The default native profile requires no EXL3 runtime; Dense default requires the lossless and owner prefill extensions. FP16 recurrent state and NVFP4 LM head search change numerical behavior and remain opt-in. Dense performance validation covers Qwen3.8-27B, TP2 and SM120. Qwen3.5-35B-A3B has a separate [MoE profile](docs/qwen35-moe.md). Other models and GPU architectures need their own integration and validation.

## Installation

Use the [complete installation guide](docs/installation.md) for the current native MXFP6 profile: official vLLM 0.29.0, MXFP6 SM120 0.2.1 and optional prefill extensions. It includes source installation, the image build, complete startup commands and a 3k/1k check. Installing the Python wheel alone registers GEMM but does not install the full graph/communication profile.

After installing vLLM 0.29.0 and building MXFP6 SM120 against that environment's PyTorch:

```bash
git clone https://github.com/troycheng/vllm-mach.git
cd vllm-mach
python -m pip install .
vllm-mach-install --apply
```

The installer checks dependency versions and patch applicability, stages all changes before writing and is idempotent. Use a clean environment; do not combine this profile with the old EXL3 runtime/GDN patches.

To build the image with the Dense default prefill extensions:

```bash
docker buildx build --load \
  --build-context cuda132=/usr/local/cuda-13.2 \
  --build-arg MAX_JOBS=8 \
  -f deploy/Dockerfile -t vllm-mach:local .
```

This requires Linux x86-64, Docker Buildx and the CUDA 13.2 toolkit at the supplied path. The image supplies the separate CUDA 13.0 build for the lossless collective. Only the MXFP6 model checkpoint is required; no EXL3 checkpoint or rank64 assets are needed. Mach does not distribute model weights.

Release `0.1.0a10` remains documented with ExLlamaV3 1.4.9; a9 uses vLLM 0.28. See [earlier installations](docs/public-install.md) for those versions.

## Usage

Model checkpoint: [nekofish/Qwen3.8-27B-MXFP6 on Hugging Face](https://huggingface.co/nekofish/Qwen3.8-27B-MXFP6).

For the default native MXFP6 profile (install both prefill extensions for Dense):

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm-mach-serve \
  --model /models/Qwen3.8-27B-MXFP6 --host 127.0.0.1 --port 8000
```

After building the image and preparing the MXFP6 model directory in the installation guide, enable the full optional profile with:

```bash
docker run --rm --name mach --gpus '"device=0,1"' \
  --ipc=host --network=host \
  -v "$PWD/models:/models:ro" \
  -v mach-kernel-cache:/root/.cache \
  vllm-mach:local \
  --model /models/mxfp6 \
  --fp16-ssm --nvfp4-lm-head \
  --kv-cache-memory-bytes 8218214400 --host 127.0.0.1 --port 8000
```

The default launcher enables native MXFP6, fused AllReduce/residual/RMSNorm, compact BF16 greedy argmax communication, CUDA Graphs, small-batch persistent and large-batch BA overlap GDN decode. `--dry-run` prints the resolved flags. FP16 SSM, lossless/owner prefill and NVFP4 LM head search are individually opt-in; `--verify-prefill` enables diagnostic prefill comparisons.

GDN decode enables two independent routes by default: `--gdn-persistent` uses the persistent CUDA kernel at physical M1/2/4/8 with FP32 or FP16 SSM; `--gdn-ba-overlap` overlaps BA with QKV/convolution at M16/24/32 and supports FP32 or FP16 SSM. Both retain the native path for prefill, speculative and unsupported calls. With `--fp16-ssm`, persistent reads and writes FP16 state while retaining FP32 accumulation. Use `--no-gdn-persistent` and `--no-gdn-ba-overlap` to disable them independently. See [GDN validation](docs/gdn-decode.md), including a separate physical-M4 fidelity diagnostic.

Owner-prefill partitions rows between the two ranks and replicates MLP weights at 32 layers, costing approximately 3.11 GiB per GPU. The optional NVFP4 head adds approximately 341 MiB per GPU for candidate search followed by BF16 refinement. Keep KV bytes fixed when comparing throughput.

For individual feature switches and request fallbacks, see [native MXFP6 integration](docs/native-mxfp6.md) and the [installation guide](docs/installation.md).

## MXFP6 integration

[`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120) owns MXFP6 packing, MXFP8 activation quantization, W6A8 GEMM, and workspace management. vLLM Mach handles vLLM registration, checkpoint metadata, tensor-parallel slices, CUDA Graph lifecycle, fused AllReduce/RMSNorm and optional prefill/LM head routing. The native profile requires both packages.

The version-pinned runtime patch and file manifest ship in the Mach wheel. The optional prefill CUDA extensions are built separately from this repository.

## Validation

The [native MXFP6 validation record](docs/native-mxfp6.md) covers official-wheel patch installation and idempotency, changing-input CUDA Graphs, TP2 codec/owner transport, request fallbacks and real-model serving acceptance. The [fidelity and serving experiments](docs/native-fidelity.md) include per-query and per-request regression checks. The [installation guide](docs/installation.md) records the dependency contract and test commands.

The earlier [validation record](docs/validation.md) and [rank64/owner-prefill integration](docs/champion-port.md) retain the historical EXL3 release results.

## Limitations

Routed MoE models other than the validated Qwen3.5-35B-A3B configuration remain unsupported by this profile. The optional NVFP4 LM head path accelerates eligible greedy decode only; requests requiring full logits retain the original BF16 head and sampler. FP16 SSM and approximate candidate search do not promise bitwise equivalence. Broad model-quality and long-context validation remain outstanding; the revised Dockerfile has not been built in the current validation environment.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Performance results must identify the model, checkpoint format, runtime versions, topology, request shape, graph mode, correctness criterion, and baseline.

## License

vLLM Mach is licensed under [Apache-2.0](LICENSE). Derived notices are listed in [NOTICE](NOTICE). External runtimes retain their own licenses. This project is independent of the vLLM project.
