<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/vllm-mach-horizontal-dark.png">
    <img src="assets/logo/vllm-mach-horizontal-light.png" alt="vLLM Mach" width="720">
  </picture>
</p>

<p align="center">MXFP6, MXFP8 and block-FP8 inference for vLLM on NVIDIA RTX 5090</p>

<p align="center">
  <a href="https://github.com/troycheng/vllm-mach/releases"><img alt="Release" src="https://img.shields.io/github/v/release/troycheng/vllm-mach?include_prereleases&sort=semver"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.29.0-6C5CE7">
</p>

<p align="center">
  <a href="#supported-configurations">Choose a profile</a> ·
  <a href="#installation">Install and run</a> ·
  <a href="#optimizations">Optimizations</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#documentation">Documentation</a>
</p>

vLLM Mach provides model-specific runtime extensions for **vLLM 0.29.0 on SM120**: native quantized kernels, fused communication and activation operations, optimized GDN decode, and CUDA Graph policies. It packages the source patches, native builds, model preparation and launch settings needed to run each profile.

The [MXFP6 SM120 project](https://github.com/Nekofish-L/mxfp6_sm120) supplies reusable MXFP6/MXFP8 operators. Mach connects those operators to serving, and also provides an independent optimization path for existing block-FP8 checkpoints.

## Supported configurations

All profiles below target Linux x86-64, Python 3.12, RTX 5090 GPUs with 32 GiB each, BF16 model activations, and text-only, non-speculative inference.

| Model | Weight format / profile | GPUs | Start here |
|---|---|---|---|
| Qwen3.8-27B Dense | Native MXFP6 | 2, TP2 | [MXFP6 setup](#native-mxfp6-dense-and-moe) |
| Qwen3.5-35B-A3B MoE | Native MXFP6 | 2, TP2 | [MXFP6 setup](#native-mxfp6-dense-and-moe) / [MoE guide](docs/qwen35-moe.md) |
| Qwen3.5-4B | MXFP8 champion, `qwen35-4b-mxfp8-champion-v1` | 1, TP1 | [MXFP8 setup](#mxfp8-4b-champion) |
| Qwen3.5-4B | Existing block-FP8, `qwen35-4b-block-fp8-v1` | 1, TP1 | [Block-FP8 setup](#block-fp8-4b) |

MXFP8 uses E4M3 values with E8M0 group-32 scales; block-FP8 uses the checkpoint's FP32 block scales. They have separate model and runtime entry points. MXFP6 and block-FP8 retain BF16 KV by default; the MXFP8 champion uses calibrated FP8 KV.

This README describes **`main`**, including the two 4B profiles added after [v0.1.1](docs/releases/0.1.1.md). Use the source builds below for these features; the v0.1.1 tag contains the earlier native MXFP6 release. Each profile guide records its qualified dependencies and revisions.

## Installation

Use Docker Buildx, the NVIDIA container runtime and a compatible GPU driver. From a Linux host:

```bash
git clone --branch main https://github.com/troycheng/vllm-mach.git
cd vllm-mach
git rev-parse HEAD
```

Choose one of the following builds. Each Dockerfile installs the corresponding native extensions and versioned runtime patches against its vLLM 0.29.0 base. `MAX_JOBS` controls native compilation parallelism.

### Native MXFP6 Dense and MoE

```bash
docker buildx build --load --build-arg MAX_JOBS=2 \
  -f deploy/Dockerfile -t vllm-mach:mxfp6 .

docker run --rm --gpus '"device=0,1"' --ipc=host -p 8000:8000 \
  -v /absolute/path/to/mxfp6-checkpoint:/model:ro \
  vllm-mach:mxfp6 --model /model --served-model-name mach --host 0.0.0.0
```

Use a supported checkpoint, such as [nekofish/Qwen3.8-27B-MXFP6](https://huggingface.co/nekofish/Qwen3.8-27B-MXFP6). The launcher reads `config.json` to select Dense or MoE. See [manual installation and memory settings](docs/installation.md) and the [MoE launch configuration](docs/qwen35-moe.md).

| MXFP6 mode | Recurrent state | LM head |
|---|---|---|
| Default | FP32 | BF16 |
| Full: append `--fp16-ssm --nvfp4-lm-head` | FP16 | NVFP4 candidate search with BF16 refinement |

Full uses lower-precision state and approximate candidate search to improve throughput. Its candidate search handles eligible greedy decode; requests for full logits or unsupported sampling use the BF16 head. These switches belong to the MXFP6 launcher.

### MXFP8 4B champion

Build the image, then [prepare one checkpoint](docs/mxfp8-model.md) from the pinned public `Qwen/Qwen3.5-4B` BF16 revision and the [47 MB L0 asset](https://github.com/troycheng/vllm-mach/releases/tag/qwen35-4b-mxfp8-assets-v1). The preparation guide includes downloads and SHA-256 verification.

```bash
docker buildx build --load --build-arg MAX_JOBS=2 \
  --build-arg MXFP6_REVISION=15a56aa2774552d0584d8fc6b621b41dc173f30b \
  -f deploy/Dockerfile.mxfp8 -t vllm-mach:mxfp8-champion .

mkdir -p "$PWD/mxfp8-model" "$PWD/mxfp8-run"
docker run --rm --entrypoint vllm-mach-mxfp8-prepare \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v /absolute/path/to/pinned-bf16:/input/bf16:ro \
  -v /absolute/path/to/qwen35-4b-mxfp8-asym-l0-v1.safetensors:/input/l0.safetensors:ro \
  -v "$PWD/mxfp8-model:/output" vllm-mach:mxfp8-champion \
  --bf16 /input/bf16 --l0-codes /input/l0.safetensors --output /output/model

docker run --rm --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -v "$PWD/mxfp8-model/model:/model:ro" -v "$PWD/mxfp8-run:/runs" \
  vllm-mach:mxfp8-champion /model --run-dir /runs --host 0.0.0.0
```

Preparation requires a new output checkpoint directory and runs on CPU. Serving loads that single checkpoint. The profile includes FP32 SSM state, dual-activation MXFP8 projections, FP8 KV and an NVFP4 coarse head with indexed BF16 score refinement. See the [complete profile](docs/mxfp8-champion.md) for its numerical path, startup compilation and configuration.

### Block-FP8 4B

Use an existing compatible Qwen3.5-4B compressed-tensors block-FP8 checkpoint. This path keeps its weights, BF16 KV/head and FP32 SSM state.

```bash
docker buildx build --load --build-arg MAX_JOBS=2 \
  -f deploy/Dockerfile.fp8 -t vllm-mach:fp8 .

mkdir -p "$PWD/fp8-run"
docker run --rm --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -v /absolute/path/to/block-fp8-checkpoint:/model:ro \
  -v "$PWD/fp8-run:/runs" \
  vllm-mach:fp8 /model --run-dir /runs --host 0.0.0.0 \
  --features n64 ordered silu fa2
```

The example assumes tokenizer files are in the checkpoint directory. A separate tokenizer can be mounted and selected with `--tokenizer`. Choose any nonempty subset of the four features for comparisons; see the [profile guide](docs/fp8-profile.md) for operator eligibility and launch receipts.

## Usage

All three launchers expose vLLM's OpenAI-compatible API. Once the server reports ready, `http://localhost:8000/v1/models` lists its served model name. The MXFP6 example uses `mach`; the 4B defaults are `q35-mx8-study` and `q35-fp8-study`.

The 4B images include the same reproducible 3k/1k benchmark client. In a second terminal, while the MXFP8 server is running:

```bash
docker run --rm --network host --entrypoint vllm-mach-mxfp8-bench \
  -v "$PWD/mxfp8-run:/runs" vllm-mach:mxfp8-champion \
  --base-url http://127.0.0.1:8000 --model q35-mx8-study --outdir /runs/sixpoint
```

For block-FP8, use image `vllm-mach:fp8`, entry point `vllm-mach-fp8-bench`, model `q35-fp8-study`, and `--points 4 16 32 64`. Each benchmark needs a new output directory. See the [benchmark protocol](docs/mxfp8-benchmark.md) and [stock FP8 comparison setup](docs/fp8-qualification.md#serving).

## Optimizations

| Profile | Main components | Details |
|---|---|---|
| MXFP6 Dense | Native W6A8; TP2 AllReduce/residual/RMSNorm fusion; fused SwiGLU, GDN and MXFP8 activation producers; persistent GDN and BA overlap; lossless and owner prefill | [Native integration](docs/native-mxfp6.md), [producer fusion](docs/dense-producer-fusion.md), [AR/Norm/quantization](docs/ar-norm-mxfp8.md), [GDN](docs/gdn-decode.md), [prefill](docs/long-prefill.md) |
| MXFP6 MoE | Grouped native expert kernels; prefill/decode graphs; fused TP2 communication; non-expert projection dispatch tuned at selected physical M values from 32 to 160 | [MoE serving](docs/qwen35-moe.md), [projection dispatch](docs/moe-projection-20260923.md) |
| MXFP8 4B | Native W8A8 and dual-activation MLP/QKVZ kernels; small-M BA GEMV; ordered GDN; parallel QKVZ/BA; FP8 KV; coarse/refined head; fixed compilation choices and FULL/PW graph routing | [Complete champion components](docs/mxfp8-champion.md#required-parts) |
| Block-FP8 4B | N64 GEMM tiles; ordered low-M GEMM; fused SiLU/block quantization; mixed decode/prefill FA2; matched RMS compilation choices | [Linear](docs/fp8-linear.md), [activation](docs/fp8-activation.md), [attention](docs/fp8-attention.md) |

Mach packages these integrations directly. The block-FP8 linear routes are also submitted in [vLLM PR #60385](https://github.com/vllm-project/vllm/pull/60385). Mixed FA2 builds on [LiRunGuo's PR #58013](https://github.com/vllm-project/vllm/pull/58013); the SiLU work is related to [PR #45055](https://github.com/vllm-project/vllm/pull/45055).

## Performance

All throughput tables use **3000 input / 1000 output tokens**, measured in **output tokens/s**. `c` denotes concurrent requests; `M` in kernel and fidelity reports denotes physical batch rows. Results compare the complete configurations described in each linked report.

### Qwen3.5-4B MXFP8 — one RTX 5090, TP1

The October 6 source-built champion completes all 1,040 scored requests. Its output hashes match the previous champion at all six points. Community FP8 is the October 5 screen using the same public workload; each implementation has one complete screen.

| Concurrency | Community FP8 | Mach MXFP8 | Gain |
|---|---:|---:|---:|
| c4 | 683.959 | 880.131 | +28.68% |
| c16 | 2053.547 | 2481.298 | +20.83% |
| c24 | 2559.262 | 3189.417 | +24.62% |
| c32 | 2887.825 | 3672.696 | +27.18% |
| c48 | 3220.549 | 4168.038 | +29.42% |
| c64 | 3402.757 | 4653.929 | +36.77% |

[Qualified revisions, precision and latency](docs/mxfp8-qualification-20261006.md) · [raw results](docs/data/mxfp8-qualification-20261006.json).

### Qwen3.5-4B block-FP8 — one RTX 5090, TP1

October 6, one stock/all-features pair per point on the same checkpoint and vLLM 0.29.0 stack; all 680 requests per arm succeed. This is a separate stock measurement from the MXFP8 table above.

| Concurrency | Stock block-FP8 | Mach block-FP8 | Gain |
|---|---:|---:|---:|
| c4 | 678.538 | 739.843 | +9.03% |
| c16 | 2018.890 | 2143.596 | +6.18% |
| c32 | 2834.840 | 3004.000 | +5.97% |
| c64 | 3363.733 | 3598.838 | +6.99% |

Mean TTFT falls 3.82% at c32 and 7.06% at c64; it rises 4.05% at c4. [Precision, TTFT and reproduction](docs/fp8-qualification.md) · [raw results](docs/data/fp8-qualification-20261006.json).

### Native MXFP6 — two RTX 5090 GPUs, TP2

**Qwen3.8-27B Dense:** September 20 configuration study, one sweep per configuration. Default enables the communication, GDN, producer and prefill optimizations; full adds FP16 SSM and the NVFP4 candidate head.

| Configuration | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| Stock FP8 | 264.57 | 802.79 | 1011.43 | 1150.97 |
| Mach MXFP6 default | 380.17 | 1046.93 | 1374.53 | 1553.09 |
| Mach MXFP6 full | 411.14 | 1141.40 | 1498.93 | 1718.24 |

Mean of the four per-point gains over FP8: **+36.24% default / +48.77% full**. [Optimization breakdown and exact settings](docs/dense-gains-aligned-baseline-20260920.md) record the runtime, GPU-group and measurement-batch differences in this study.

**Qwen3.5-35B-A3B MoE:** September 28 MXFP6 retest, two-sweep means, compared with the September 17 FP8 deployment. The measured default row has projection tuning off; full includes it. Current launchers enable eligible projection tuning by default.

| Configuration | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| Stock FP8 | 679.89 | 1471.58 | 1774.64 | 1962.14 |
| Mach MXFP6 default, projection off | 1008.56 | 2338.97 | 2920.74 | 3277.02 |
| Mach MXFP6 full | 1096.93 | 2561.85 | 3073.56 | 3518.44 |

Mean per-point gains over FP8: **+59.72% / +71.98%**, respectively. [Optimization breakdown and settings](docs/moe-gains-fp8-baseline-20260920.md) include the individual sweeps. [Projection testing](docs/moe-projection-20260923.md) also covers c32–c160. Earlier FP8/MXFP6/NVFP4 comparisons are available for [Dense](docs/native-fidelity.md) and [MoE](docs/qwen35-moe.md).

### Numerical fidelity

The fixed-gold diagnostic uses 256 queries and 10,479 gold-token logprobs per physical batch setting. These MAEs measure deviation from a reference, rather than task accuracy; the evaluation corpus is private.

| Profile | Tested physical rows | Result |
|---|---|---|
| Block-FP8 4B | M4, M32, M64 | Raw records byte-identical to stock block-FP8; MAE 0 against stock |
| MXFP8 4B champion | M32, M64 | Raw records byte-identical to the accepted champion; MAE against same-M BF16 **0.04904 / 0.05329** |
| MXFP6 Dense default / full | M32 | September 17 BF16-reference MAE **0.09064 / 0.08962**; stock FP8 0.06143, NVFP4 0.17328 |
| MXFP6 MoE default / full | M32 | September 17 BF16-reference MAE **0.07261 / 0.07409**; stock FP8 0.05534, NVFP4 0.22836 |

MXFP6 logprob requests use the BF16 head, so that diagnostic does not assess candidate-search generation. The MXFP8 BF16 reference uses BF16 KV/FlashAttention, while the champion uses FP8 KV/FlashInfer. See the [MXFP6 methodology](docs/profile-fidelity-20260917.md), [MXFP8 qualification](docs/mxfp8-qualification-20261006.md#precision-and-startup) and [block-FP8 qualification](docs/fp8-qualification.md#precision) for the full comparisons.

## Documentation

| Guide | Contents |
|---|---|
| [MXFP6 installation](docs/installation.md) | Manual build, launch flags, graph defaults and memory sizing |
| [Native MXFP6](docs/native-mxfp6.md) / [MoE](docs/qwen35-moe.md) | Checkpoint loading, operators, graph lifecycle and model-specific setup |
| [MXFP8 champion](docs/mxfp8-champion.md) / [model preparation](docs/mxfp8-model.md) | Complete TP1 profile, public model inputs and single-checkpoint reconstruction |
| [MXFP8 runtime installation](docs/mxfp8-installation.md) | Source pins, package versions and build records |
| [Block-FP8 profile](docs/fp8-profile.md) | Four feature switches, source installation and precision contract |
| [4B benchmark protocol](docs/mxfp8-benchmark.md) | Generated requests, warmup, concurrency and metric definitions |
| [Historical releases](docs/public-install.md) / [benchmark archive](docs/benchmarks.md) | Earlier EXL3 and hybrid-checkpoint work |

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Include the checkpoint, runtime versions, GPU topology, workload, graph mode, correctness checks and baseline with performance reports.

vLLM Mach is licensed under [Apache-2.0](LICENSE); derived notices are in [NOTICE](NOTICE). External components retain their own licenses. This project is independent of the vLLM project.
