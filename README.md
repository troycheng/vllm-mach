<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/vllm-mach-horizontal-dark.png">
    <img src="assets/logo/vllm-mach-horizontal-light.png" alt="vLLM Mach" width="720">
  </picture>
</p>

<p align="center">EXL3 and MXFP6 runtime paths for vLLM</p>

<p align="center">
  <a href="https://github.com/troycheng/vllm-mach/releases"><img alt="Release" src="https://img.shields.io/github/v/release/troycheng/vllm-mach?include_prereleases&sort=semver"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.29.0-6C5CE7">
</p>

vLLM Mach adds an EXL3 provider and optional MXFP6 execution paths to vLLM. Its first validated model-specific profile targets Qwen3.8-27B Dense. The EXL3 path validates checkpoint metadata, loads tensor-parallel slices through vLLM's packed-module mapping, groups compatible QKV and QKVZ projections, and primes kernels before CUDA Graph capture. BF16 I/O and fused prefill reconstruction are optional. Native MXFP6 kernels are provided by [`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120).

## Why Mach

Mach targets fast serving at small and medium batch sizes, with most tuning focused on 4 to 32 concurrent requests. Its hybrid profiles choose EXL3 or MXFP6 by projection and row count, then combine those kernels with grouped execution, fused tensor-parallel communication and CUDA Graph support. This gives prefill and decode different execution paths while retaining vLLM's serving interface.

The aim is higher throughput with controlled numerical error. The current Qwen3.8-27B results show a useful middle ground between FP8's numerical fidelity and NVFP4's speed.

## Support

| Path | Validated configuration |
|---|---|
| EXL3 | vLLM `0.28.0`; [Qwen3.8-27B Dense K5/K6 EXL3 checkpoint](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated/tree/ab3a91a13813df8096cb4c1d560ed3669035d0cf); TP2/PP1; SM120; BF16 KV cache; non-speculative decoding |
| EXL3 CUDA Graph | The EXL3 configuration above with `FULL_DECODE_ONLY` capture sizes `1, 2, 4, 8, 16, 24, 32` |
| EXL3/MXFP6 | The EXL3 configuration above with `VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b` and `mxfp6-sm120==0.2.1` |
| a10 checkpoint-hybrid | vLLM `0.29.0`, patched ExLlamaV3 `1.4.9`; Qwen3.8-27B K5/K6; TP2/PP1, SM120, BF16 activations/KV, opt-in FP16 recurrent state; [configuration and regression](docs/dependency-upgrade.md) |
| Fused FlashInfer collective | The EXL3/MXFP6 profile with `flashinfer-python==0.6.16.post3` or `0.6.18` and the matching runtime patches |

The EXL3 provider does not require MXFP6. This table records released configurations; the [current source installation](docs/installation.md) adds ExLlamaV3 1.5.0 and the new rank64/owner-prefill integration. See [compatibility](docs/compatibility.md) for native dependencies and fallback behavior.

## Performance

### 3k/1k reference comparison

Qwen3.8-27B, two RTX 5090 GPUs, TP2, 3000 input / 1000 output tokens, measured in September 2026. Each configuration uses 192/512/672/768 requests at c4/c16/c24/c32 after a full c32 warmup. Throughput counts generated tokens only.

![Serving throughput across six configurations](docs/images/throughput-comparison.png)

Our K5/K6 hybrid source stack delivers **35.6% higher throughput than official vLLM 0.29 FP8**, averaging the four concurrency levels equally. It also improves on the accelerated vLLM 0.28 FP8 stack by **30.4%** and the native MXFP6 Champion by **10.3%**.

The K5/K6 and K4/K5 curves were measured on the optimization source stack. The [Mach 0.1.0a9 profile](docs/fp16-ssm.md) integrates the K5/K6 optimizations and passes 40/40 task checks plus 2,592 byte comparisons for serial versus overlapped execution; these full-length curves are not release-wheel measurements. K4/K5 uses a derived W6 execution cache; NVFP4 uses local calibration.

### Numerical fidelity

Gold-token logprob MAE against BF16 over 256 queries and 10,479 target tokens, using physical-M32 teacher-forced decode. Lower is better; whiskers show 95% query-bootstrap intervals.

![Gold-token logprob MAE against BF16](docs/images/accuracy-comparison.png)

K5/K6 records **0.0915 MAE**, compared with **0.0520 for official FP8** and **0.1700 for the tested NVFP4 configuration**. That is **46.2% lower MAE than NVFP4**, while retaining 88.8% to 97.4% of its throughput across the four concurrency levels. FP8 remains closest to BF16 in this test; NVFP4 remains fastest. MAE measures numerical deviation, not task accuracy.

### Fidelity and throughput

Upper-left is better: less logprob deviation from BF16 and higher throughput. Each point combines the M32 MAE above with the mean throughput gain over official vLLM 0.29 FP8, weighting c4/c16/c24/c32 equally. Horizontal bars show the MAE's 95% interval; vertical bars show the lowest and highest gains across those four concurrency levels.

![Logprob MAE and throughput gain against official FP8](docs/images/quality-throughput-tradeoff.png)

[Result tables, test setup, raw data and Mach release measurements](docs/benchmarks.md) are available.

### Deployment tradeoffs

The fastest profiles require matching native extensions and pinned vLLM/FlashInfer patches, so deployment takes more setup than a stock vLLM installation. The K5/K6 and K4/K5 results above also use opt-in FP16 recurrent state, which changes numerical behavior; the base profile keeps that option disabled. Current validation covers Qwen3.8-27B Dense, TP2 and SM120. Other models, MoE and GPU architectures need their own integration and validation.

## Installation

Use the [complete installation guide](docs/installation.md) for the current source profile: vLLM 0.29.0, ExLlamaV3 1.5.0 and matching native extensions. It includes the image build, model assets, complete startup command and a 3k/1k check. Installing the Python wheel alone does not reproduce the hybrid profile.

```bash
git clone https://github.com/troycheng/vllm-mach.git
cd vllm-mach
docker buildx build --load \
  --build-context cuda132=/usr/local/cuda-13.2 \
  --build-arg MAX_JOBS=8 \
  -f deploy/Dockerfile -t vllm-mach:local .
```

This requires Linux x86-64, Docker Buildx and the CUDA 13.2 toolkit at the supplied path. The image supplies the separate CUDA 13.0 build for the lossless collective. The complete profile also requires matching local MXFP6 and rank64 assets. Mach does not distribute these assets. See [model asset generation](docs/quantization.md) for the two generation commands and their validation status; the installation guide also covers packaging existing assets.

Release `0.1.0a10` remains available with ExLlamaV3 1.4.9; a9 uses vLLM 0.28. See [earlier installations](docs/public-install.md) for those versions. Mach's BF16 interface is maintained downstream; the official ExLlamaV3 wheel does not include it.

## Usage

After building the image and preparing the three model directories in the installation guide:

```bash
docker run --rm --name mach --gpus '"device=0,1"' \
  --ipc=host --network=host \
  -v "$PWD/models:/models:ro" \
  -v mach-kernel-cache:/root/.cache \
  vllm-mach:local \
  --model /models/exl3 --mxfp6-checkpoint /models/mxfp6 \
  --rank64-bundle /models/rank64 --owner-prefill \
  --max-num-seqs 48 --host 127.0.0.1 --port 8000
```

The launcher sets the full profile, including BF16 I/O, Temporal M24, GDN decode, fused collectives, BA overlap and FP16 recurrent state. `--dry-run` prints the resolved flags. Omit `--rank64-bundle` and `--owner-prefill` to use the previous checkpoint-hybrid execution paths.

The new decode path uses selected NVFP4 gate/up weights with rank64 compensation at physical M32. Owner-prefill partitions rows between the two ranks and replicates MLP weights at 32 layers to reduce communication. Replication costs 3.11 GiB per GPU; the launcher keeps the original 8.22 GB KV allocation. These additions are optional and do not replace the EXL3 provider.

For standalone EXL3 or individual feature switches, see the [base EXL3 setup](docs/public-install.md), [checkpoint-hybrid routing](docs/checkpoint-hybrid.md), [Temporal M24](native/exl3_temporal_m24/README.md), and [GDN integration](profiles/flashinfer-0.6.18-gdn/README.md).

## MXFP6 integration

[`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120) owns MXFP6 packing, MXFP8 activation quantization, W6A8 GEMM, and workspace management. vLLM Mach handles vLLM registration, checkpoint metadata, tensor-parallel slices, projection routing, CUDA Graph lifecycle, and the optional FlashInfer AllReduce/RMSNorm/MXFP8 boundary. The hybrid profile requires both packages.

## Validation

The [validation record](docs/validation.md) lists package tests, real-weight kernel checks, changing-input CUDA Graph checks, and TP2 task regressions for each release. The [rank64 and owner-prefill integration](docs/champion-port.md) records the clean-image checks and C4/C16/C24/C32 short regression for the current source. The [installation guide](docs/installation.md) records the dependency contract and test commands.

## Limitations

Routed MoE execution, `lm_head` conversion and additional model/GPU configurations remain unsupported. The optional GDN source profile is not installed by the Python wheel.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Performance results must identify the model, checkpoint format, runtime versions, topology, request shape, graph mode, correctness criterion, and baseline.

## License

vLLM Mach is licensed under [Apache-2.0](LICENSE). Derived notices are listed in [NOTICE](NOTICE). External runtimes retain their own licenses. This project is independent of the vLLM project.
