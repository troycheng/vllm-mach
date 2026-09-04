<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/vllm-mach-horizontal-dark.png">
    <img src="assets/logo/vllm-mach-horizontal-light.png" alt="vLLM Mach" width="720">
  </picture>
</p>

<p align="center">High-performance runtime extensions for vLLM</p>

<p align="center">
  <a href="https://github.com/troycheng/vllm-mach/releases"><img alt="Release" src="https://img.shields.io/github/v/release/troycheng/vllm-mach?include_prereleases&sort=semver"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.28.0-6C5CE7">
</p>

vLLM Mach adds an EXL3 provider and optional MXFP6 execution paths to vLLM 0.28. Its first validated model-specific profile targets Qwen3.8-27B Dense. This repository contains the vLLM loading, tensor-parallel mapping, dispatch, and CUDA Graph integration. Native MXFP6 kernels are provided by [`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120).

## Support

| Path | Validated configuration |
|---|---|
| EXL3 | vLLM `0.28.0`; Qwen3.8-27B Dense with hydrated K5/K6 weights; TP2/PP1; SM120; BF16 KV cache; non-speculative decoding |
| EXL3 CUDA Graph | The EXL3 configuration above with `FULL_DECODE_ONLY` capture sizes `1, 2, 4, 8, 16, 24, 32` |
| EXL3/MXFP6 | The EXL3 configuration above with `VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b` and `mxfp6-sm120==0.2.1` |
| Fused FlashInfer collective | The EXL3/MXFP6 profile with `flashinfer-python==0.6.16.post3` or `0.6.18` and the matching runtime patches |

The EXL3 provider does not require MXFP6. This table records completed validation for release `0.1.0a2`; configurations outside it remain unverified. See [compatibility](docs/compatibility.md) for native dependencies, fallback behavior, and unsupported configurations.

## Installation

Install vLLM and the release wheel in the same environment:

```bash
python -m pip install "vllm==0.28.0"
python -m pip install \
  https://github.com/troycheng/vllm-mach/releases/download/v0.1.0a2/vllm_mach-0.1.0a2-py3-none-any.whl
```

EXL3 execution requires an [ExLlamaV3](https://github.com/turboderp-org/exllamav3) build that exports `exl3_gemm` and `exl3_mgemm`. [B12X](https://github.com/local-inference-lab/b12x) is an optional prefill backend.

The EXL3/MXFP6 profile also requires [`mxfp6-sm120==0.2.1`](https://github.com/Nekofish-L/mxfp6_sm120#build), built against the same PyTorch and CUDA environment. Stream-K graph execution and the optional FlashInfer collective require the version-locked patches under [`profiles/vllm-0.28.0`](profiles/vllm-0.28.0/README.md). Apply the FlashInfer patch before JIT compilation.

To build vLLM Mach from source:

```bash
git clone https://github.com/troycheng/vllm-mach.git
cd vllm-mach
python -m pip install build
python -m build --wheel
```

## Usage

### EXL3

Select the `mach` plugin explicitly when other vLLM plugins are installed:

```bash
export VLLM_PLUGINS=mach

vllm serve /path/to/hydrated-exl3-model \
  --quantization exl3 \
  --tensor-parallel-size 2
```

The validated CUDA Graph configuration also enables QKV MGEMM and primes EXL3 kernels before capture:

```bash
export VLLM_PLUGINS=mach
export EXL3_QKV_MGEMM=1
export VLLM_EXL3_GRAPH_DECODE=1

vllm serve /path/to/hydrated-exl3-model \
  --quantization exl3 \
  --tensor-parallel-size 2 \
  --compilation-config \
  '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
```

### EXL3/MXFP6

Enable the Qwen3.8-27B profile in the same environment as the EXL3 serve command:

```bash
export VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b
```

The following optional settings enable B12X dispatch and fused EXL3 reconstruction. Install B12X before enabling its route:

```bash
export VLLM_EXL3_B12X_MIN_M=128
export VLLM_EXL3_B12X_N_RANGE=5120-36864
export VLLM_EXL3_B12X_ANY_BITS=1
export VLLM_EXL3_PREFILL_FUSED_RECONSTRUCT_MIN_M=128
```

After applying the matching vLLM and FlashInfer patches, enable the fused TP2 collective with:

```bash
export VLLM_MACH_EXL3_MXFP6_FUSED_AR_NORM_MXFP8=1
```

The hybrid profile keeps `lm_head` and unmatched projections on EXL3. It routes MLP and attention output projections to MXFP6 for all row counts. QKV and QKVZ projections use MXFP6 for prefill calls with at least 128 rows and EXL3 for decode.

## MXFP6 integration

[`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120) owns MXFP6 packing, MXFP8 activation quantization, W6A8 GEMM, and workspace management. vLLM Mach handles vLLM registration, checkpoint metadata, tensor-parallel slices, projection routing, CUDA Graph lifecycle, and the optional FlashInfer AllReduce/RMSNorm/MXFP8 boundary. The hybrid profile requires both packages.

## Validation

Release `0.1.0a2` passed its package and clean-install checks. The Qwen3.8-27B TP2 service gate loaded the model, captured all configured graph sizes, selected the fused FlashInfer path on both ranks, reached the health check, and returned a completion response.

The detailed test matrix and historical fused-collective measurements are in [the validation record](docs/validation.md). They are integration evidence, not a general performance claim.

## Limitations

The current release does not provide routed MoE execution, a GDN kernel replacement, `lm_head` conversion, or validated support for additional models and GPU architectures.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Performance results must identify the model, checkpoint format, runtime versions, topology, request shape, graph mode, correctness criterion, and baseline.

## License

vLLM Mach is licensed under [Apache-2.0](LICENSE). Derived notices are listed in [NOTICE](NOTICE). External runtimes retain their own licenses. This project is independent of the vLLM project.
