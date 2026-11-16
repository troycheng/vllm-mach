<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/vllm-mach-horizontal-dark.png">
    <img src="assets/logo/vllm-mach-horizontal-light.png" alt="vLLM Mach" width="720">
  </picture>
</p>

<p align="center">MXFP6, MXFP8 and block-FP8 inference for vLLM on NVIDIA RTX 5090</p>

<p align="center">
  <a href="https://github.com/troycheng/vllm-mach/releases"><img alt="Release" src="https://img.shields.io/github/v/release/troycheng/vllm-mach?sort=semver"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.33.0-6C5CE7">
</p>

vLLM Mach packages model-specific serving optimizations for **vLLM 0.33.0 on RTX 5090 (SM120)**. The [MXFP6 SM120 library](https://github.com/Nekofish-L/mxfp6_sm120) supplies native MXFP6/MXFP8 kernels; Mach integrates model loading, fusion, recurrent state, attention, graph policies and serving launchers. Existing block-FP8 checkpoints have a separate optimization path.

## Serving performance

**vLLM 0.33.0 · 3,000 input / 1,000 output tokens · output tokens/s · RTX 5090.**

MXFP8 uses BF16 KV on 2B and FP8 KV on 4B, with FP32 recurrent state by default; BF16 is optional on 2B and 4B. Native MXFP6 uses FP32 state by default; `--fp16-ssm` selects FP16 state. The hybrid head is enabled by default; `--no-hybrid-head` selects the complete head.

### Qwen3.5-2B · MXFP8

![Qwen3.5-2B · MXFP8 throughput](docs/assets/qwen35-2b-vllm033-throughput.svg)

### Qwen3.5-4B · MXFP8

![Qwen3.5-4B · MXFP8 throughput](docs/assets/qwen35-4b-vllm033-throughput.svg)

### Qwen3.5-4B · block-FP8

See the [block-FP8 migration results](docs/performance.md#block-fp8-migration-status).

### Qwen3.8-27B Dense · MXFP6

![Qwen3.8-27B Dense · MXFP6 throughput](docs/assets/qwen38-27b-vllm033-throughput.svg)

### Qwen3.5-35B-A3B MoE · MXFP6

![Qwen3.5-35B-A3B MoE · MXFP6 throughput](docs/assets/qwen35-35b-a3b-vllm033-throughput.svg)

[Performance and precision](docs/performance.md) · [Historical vLLM 0.29 FP8/NVFP4 comparisons](docs/performance-vllm029.md)

Dense retains FP32/head ON and FP16 FULL. MoE recommends FP32/head ON with headgraph OFF. [Published aggregate evidence](docs/data/evidence-vllm033/README.md) records the configurations, measurements and source hashes.


## Choose and run a profile

Use Linux x86-64, Python 3.12, Docker Buildx and the NVIDIA container runtime. See [runtime versions](docs/installation.md#runtime-versions).

```bash
git clone https://github.com/troycheng/vllm-mach.git
cd vllm-mach
```

| Model / checkpoint | Build file | Serving command | Setup |
|---|---|---|---|
| Qwen3.5-2B MXFP8 | `deploy/Dockerfile.mxfp8` | `vllm-mach-mxfp8-2b-serve` | [Prepare and serve](docs/mxfp8-2b-champion.md) |
| Qwen3.5-4B MXFP8 | `deploy/Dockerfile.mxfp8` | `vllm-mach-mxfp8-serve` | [Prepare and serve](docs/mxfp8-champion.md) |
| Qwen3.5-4B block-FP8 | `deploy/Dockerfile.fp8` | `vllm-mach-fp8-serve` | [Existing checkpoint](docs/fp8-profile.md) |
| Qwen3.8-27B MXFP6 | `deploy/Dockerfile` | `vllm-mach-serve` | [Install and launch](docs/installation.md) |
| Qwen3.5-35B-A3B MXFP6 | `deploy/Dockerfile` | `vllm-mach-serve` | [MoE setup](docs/qwen35-moe.md) |

For example, build the shared MXFP8 image and run a [prepared 2B checkpoint](docs/mxfp8-2b-model.md):

```bash
docker buildx build --load --build-arg MAX_JOBS=2 \
  -f deploy/Dockerfile.mxfp8 -t vllm-mach:mxfp8-champion .
mkdir -p "$PWD/runs"
docker run --rm --gpus '"device=0"' --ipc=host -p 8000:8000 \
  --entrypoint vllm-mach-mxfp8-2b-serve \
  -e VLLM_HYBRID_NVFP4_LM_HEAD_CUDAGRAPH=1 \
  -v /absolute/path/to/prepared-2b:/model:ro -v "$PWD/runs:/runs" \
  vllm-mach:mxfp8-champion /model --run-dir /runs/serve --host 0.0.0.0 \
  --attention-backend TRITON_ATTN --ssm-dtype float32 --hybrid-head
```

`--attention-backend TRITON_ATTN` selects the measured 2B path. Use the 2B entrypoint shown above; the shared image defaults to 4B. The server provides an OpenAI-compatible API at `/v1`; `/v1/models` lists the served model name.

This example selects the 2B FP32 default and headgraph ON. Optional BF16 state uses `--ssm-dtype bfloat16`; the headgraph environment variable is independent of the state dtype. Native MXFP6 selects Dense or MoE from `config.json`; append `--fp16-ssm` for FP16 recurrent state. For block-FP8, use `--features n64 ordered silu fa2`. The linked guides include checkpoint preparation and benchmark commands.

`--hybrid-head` supports greedy and eligible unseeded top-k sampling, with optional top-p and presence-only penalties. Other sampling requests use the original sampler. See [sampling support and numerical behavior](docs/lm-head-sampling.md).

## What the selected profiles include

| Profile | Implemented optimizations | Details |
|---|---|---|
| 2B MXFP8 | Native projections; BF16 BA M1–8/M16; ordered FP32 GDN with mixed D1–160; pinned reset-ID uploads; FULL16/FULL24 PDL; exact2048 piecewise graphs; Triton attention; K128 hybrid greedy head | [Champion components](docs/mxfp8-2b-champion.md#champion-components) |
| 4B MXFP8 | Dual-activation MLP/QKVZ; M64 pipeline variant; BF16 BA GEMV; ordered GDN; parallel QKVZ/BA; FP8 KV; coarse/refined head; fixed compilation and graph routing | [Complete profile](docs/mxfp8-champion.md#required-parts) |
| 4B block-FP8 | N64 GEMM tiles; ordered low-M GEMM; fused SiLU quantization; mixed FA2; matched RMS compilation choices | [Block-FP8 profile](docs/fp8-profile.md) |
| 27B Dense MXFP6 | Native W6A8; compact greedy sampling; TP2 AllReduce/residual/RMSNorm and activation fusion; persistent GDN; BA overlap; lossless and owner prefill | [Configuration breakdown](docs/dense-gains-aligned-baseline-20260920.md) |
| 35B-A3B MoE MXFP6 | Grouped experts; prefill/decode graphs; compact greedy sampling; fused TP2 communication; GDN integration; projection dispatch for selected physical rows32–160 | [MoE breakdown](docs/moe-gains-fp8-baseline-20260920.md) |

## Documentation

- [Performance](docs/performance.md): throughput, latency, configurations and precision.
- [Guide index](docs/README.md): installation, model preparation, operator details and development.
- [MXFP8 benchmark](docs/mxfp8-benchmark.md): per-concurrency results, SSM/KV configurations and block-FP8 comparisons.
- [Shared GDN/Mamba runtime](docs/gdn-runtime.md): quantization-independent optimizations, model/path admission and NVFP4 validation.
- [Releases](https://github.com/troycheng/vllm-mach/releases): packages and release notes.

The block-FP8 linear work is also submitted in [vLLM PR #60385](https://github.com/vllm-project/vllm/pull/60385). Mixed FA2 builds on [LiRunGuo's PR #58013](https://github.com/vllm-project/vllm/pull/58013); SiLU quantization work relates to [PR #45055](https://github.com/vllm-project/vllm/pull/45055). Mach includes its integrations directly.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for performance and correctness evidence expected with changes. [Apache-2.0](LICENSE); derived notices are in [NOTICE](NOTICE). External components retain their own licenses. This project is independent of the vLLM project.
