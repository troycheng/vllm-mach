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

vLLM Mach adds accelerated native MXFP6 execution paths to vLLM. Its validated model-specific profile targets Qwen3.8-27B Dense on vLLM 0.29.0. Mach integrates checkpoint loading, tensor-parallel execution, fused communication and CUDA Graph workspace management. Native MXFP6 kernels are provided by [`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120). The current wheel no longer registers or ships the EXL3 provider; earlier results remain in the [benchmark archive](docs/benchmarks.md).

## Why Mach

Mach targets fast serving at small and medium batch sizes, with most tuning focused on 4 to 32 concurrent requests. Its native MXFP6 profile combines packed weights with fused tensor-parallel communication and CUDA Graph support. Optional owner-prefill and LM head paths further accelerate prefill and eligible greedy decode while retaining vLLM's serving interface.

The aim is higher throughput with controlled numerical error. FP16 recurrent state and NVFP4 LM head candidate search remain explicit options; the default profile retains FP32 recurrent state and the BF16 LM head.

## Support

| Path | Validated configuration |
|---|---|
| Native MXFP6 | vLLM `0.29.0`; Qwen3.8-27B-MXFP6; `mxfp6-sm120==0.2.1`; TP2/PP1; SM120; BF16 activations/KV; text-only, non-speculative decoding |
| MXFP6 CUDA Graph | The native configuration above with the V2 runner and `FULL_DECODE_ONLY` capture sizes `1, 2, 4, 8, 16, 24, 32` |
| Fused FlashInfer collective | The native configuration above with FlashInfer `0.6.18` and the matching runtime/local IPC patches |
| Lossless / owner prefill | Optional `--lossless-prefill` / `--owner-prefill`; matching [lossless](native/lossless_prefill/README.md) and [owner](native/owner_prefill/README.md) extensions |
| FP16 SSM / NVFP4 LM head | Optional `--fp16-ssm` / `--nvfp4-lm-head`; head uses FlashInfer's built-in B12X backend, without the standalone `b12x` package; changes numerical behavior |

See the [current source installation](docs/installation.md) for native dependencies and [native MXFP6 integration](docs/native-mxfp6.md) for request eligibility and fallback behavior. Earlier EXL3 and checkpoint-hybrid releases are documented in [earlier installations](docs/public-install.md) and the [0.29 dependency upgrade](docs/dependency-upgrade.md).

## Performance

### 3k/1k reference comparison

Qwen3.8-27B, two RTX 5090 GPUs, TP2, 3000 input / 1000 output tokens, measured on September 15, 2026. The four profiles share frozen ShareGPT-derived token prompts and use 20/80/120/160 requests at c4/c16/c24/c32, with up to 32 warmup requests per point. Throughput counts generated tokens only; these are short, single-run measurements.

![Serving throughput across four configurations](docs/images/throughput-comparison.png)

| Configuration | c32 output token/s |
|---|---:|
| FP8 · stock vLLM 0.29 | 1160.4 |
| MXFP6 · Mach default | 1422.8 |
| MXFP6 · Mach full options | 1646.6 |
| NVFP4 · stock vLLM 0.29 | 1607.6 |

Weighting c4/c16/c24/c32 equally, Mach default improves throughput over stock FP8 by **22.3%**, and the full profile by **37.5%**. The full profile is slightly faster than NVFP4 at c32, but not at every concurrency.

Stock FP8/NVFP4 use unpatched vLLM 0.29 with default compilation and **FlashInfer AllReduce disabled**. Stock NVFP4 reaches **1607.6 token/s at c32**; a separate reproduction using the original benchmark client reached **1590.95**, consistent with the reported 1578.67. The previous 1268.8 result disabled compilation and is excluded.

Mach default enables fused AR/Norm and compact BF16 greedy sampling. The full profile additionally enables FP16 SSM, lossless/owner prefill and NVFP4 head search. Both use equal KV bytes and the native profile's decode graphs; stock profiles retain their own compiler and memory defaults. This compares deployable profiles, not quantization alone. [Configuration, results and reproduction](docs/native-fidelity.md).

### Numerical fidelity

Gold-token logprob MAE against a freshly measured BF16 reference over the original 256 queries and 10,479 target tokens, using physical-M32 teacher-forced decode. Lower is better; whiskers show 95% query-bootstrap intervals (20,000 resamples).

![Gold-token logprob MAE against BF16](docs/images/accuracy-comparison.png)

Mach default records **0.0906 MAE** and the full optional profile **0.0896**, compared with **0.0614 for stock FP8** and **0.1709 for stock NVFP4**, with stock compilation enabled. The two MXFP6 intervals overlap; the small difference does not establish a fidelity improvement. MAE measures numerical deviation, not task accuracy.

Logprob requests use the BF16 head. A separate test of NVFP4 candidate search retained **100% of the global BF16 top-20 tokens** across 11,998 eligible rows (239,960 tokens), with **100% final top-1 agreement**. This is an observed result on these inputs, not a guarantee for all prompts or bitwise equality of logits. [Protocol, raw results and reproduction](docs/native-fidelity.md).

### Fidelity and throughput

Upper-left is better: less logprob deviation from BF16 and higher throughput. Each point combines the M32 MAE above with the mean throughput gain over official vLLM 0.29 FP8, weighting c4/c16/c24/c32 equally. Horizontal bars show the MAE's 95% interval; vertical bars show the lowest and highest gains across those four concurrency levels.

![Logprob MAE and throughput gain against official FP8](docs/images/quality-throughput-tradeoff.png)

[Result tables, test setup and raw data](docs/native-fidelity.md) are available. Earlier EXL3 measurements remain in the [archive](docs/benchmarks.md).

### Deployment tradeoffs

The fastest profiles require matching native extensions and pinned vLLM/FlashInfer patches, so deployment takes more setup than a stock vLLM installation. The default native profile requires no EXL3 runtime or optional prefill extensions. FP16 recurrent state and NVFP4 LM head search change numerical behavior and remain opt-in. Current validation covers Qwen3.8-27B Dense, TP2 and SM120. Other models, MoE and GPU architectures need their own integration and validation.

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

To build the image with the optional prefill extensions:

```bash
docker buildx build --load \
  --build-context cuda132=/usr/local/cuda-13.2 \
  --build-arg MAX_JOBS=8 \
  -f deploy/Dockerfile -t vllm-mach:local .
```

This requires Linux x86-64, Docker Buildx and the CUDA 13.2 toolkit at the supplied path. The image supplies the separate CUDA 13.0 build for the lossless collective. Only the MXFP6 model checkpoint is required; no EXL3 checkpoint or rank64 assets are needed. Mach does not distribute model weights.

Release `0.1.0a10` remains documented with ExLlamaV3 1.4.9; a9 uses vLLM 0.28. See [earlier installations](docs/public-install.md) for those versions.

## Usage

For the default native MXFP6 profile:

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
  --fp16-ssm --lossless-prefill --owner-prefill --nvfp4-lm-head \
  --kv-cache-memory-bytes 8218214400 --host 127.0.0.1 --port 8000
```

The default launcher enables native MXFP6, fused AllReduce/residual/RMSNorm, compact BF16 greedy argmax communication and CUDA Graphs. `--dry-run` prints the resolved flags. FP16 SSM, lossless/owner prefill and NVFP4 LM head search are individually opt-in; `--verify-prefill` enables diagnostic prefill comparisons.

Owner-prefill partitions rows between the two ranks and replicates MLP weights at 32 layers, costing approximately 3.11 GiB per GPU. The optional NVFP4 head adds approximately 341 MiB per GPU for candidate search followed by BF16 refinement. Keep KV bytes fixed when comparing throughput.

For individual feature switches and request fallbacks, see [native MXFP6 integration](docs/native-mxfp6.md) and the [installation guide](docs/installation.md).

## MXFP6 integration

[`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120) owns MXFP6 packing, MXFP8 activation quantization, W6A8 GEMM, and workspace management. vLLM Mach handles vLLM registration, checkpoint metadata, tensor-parallel slices, CUDA Graph lifecycle, fused AllReduce/RMSNorm and optional prefill/LM head routing. The native profile requires both packages.

The version-pinned runtime patch and file manifest ship in the Mach wheel. The optional prefill CUDA extensions are built separately from this repository.

## Validation

The [native MXFP6 validation record](docs/native-mxfp6.md) covers official-wheel patch installation and idempotency, changing-input CUDA Graphs, TP2 codec/owner transport, request fallbacks and real-model serving acceptance. The [fidelity and serving experiments](docs/native-fidelity.md) include per-query and per-request regression checks. The [installation guide](docs/installation.md) records the dependency contract and test commands.

The earlier [validation record](docs/validation.md) and [rank64/owner-prefill integration](docs/champion-port.md) retain the historical EXL3 release results.

## Limitations

Routed MoE execution and additional model/GPU configurations remain unsupported by this profile. The optional NVFP4 LM head path accelerates eligible greedy decode only; requests requiring full logits retain the original BF16 head and sampler. FP16 SSM and approximate candidate search do not promise bitwise equivalence. Broad model-quality and long-context validation remain outstanding; the revised Dockerfile has not been built in the current validation environment.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Performance results must identify the model, checkpoint format, runtime versions, topology, request shape, graph mode, correctness criterion, and baseline.

## License

vLLM Mach is licensed under [Apache-2.0](LICENSE). Derived notices are listed in [NOTICE](NOTICE). External runtimes retain their own licenses. This project is independent of the vLLM project.
