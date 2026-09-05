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
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.28.0-6C5CE7">
</p>

vLLM Mach adds an EXL3 provider and optional MXFP6 execution paths to vLLM 0.28. Its first validated model-specific profile targets Qwen3.8-27B Dense. The EXL3 path validates checkpoint metadata, loads tensor-parallel slices through vLLM's packed-module mapping, groups compatible QKV and QKVZ projections, and primes kernels before CUDA Graph capture. BF16 I/O and fused prefill reconstruction are optional. Native MXFP6 kernels are provided by [`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120).

## Support

| Path | Validated configuration |
|---|---|
| EXL3 | vLLM `0.28.0`; [Qwen3.8-27B Dense K5/K6 EXL3 checkpoint](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated/tree/ab3a91a13813df8096cb4c1d560ed3669035d0cf); TP2/PP1; SM120; BF16 KV cache; non-speculative decoding |
| EXL3 CUDA Graph | The EXL3 configuration above with `FULL_DECODE_ONLY` capture sizes `1, 2, 4, 8, 16, 24, 32` |
| EXL3/MXFP6 | The EXL3 configuration above with `VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b` and `mxfp6-sm120==0.2.1` |
| Fused FlashInfer collective | The EXL3/MXFP6 profile with `flashinfer-python==0.6.16.post3` or `0.6.18` and the matching runtime patches |

The EXL3 provider does not require MXFP6. This table records the validated base configurations. Release `0.1.0a3` adds separately validated opt-in decode paths described below; configurations outside the documented checks remain unverified. See [compatibility](docs/compatibility.md) for native dependencies, fallback behavior, and unsupported configurations.

## Installation

Install vLLM and the release wheel in the same environment:

```bash
python -m pip install "vllm==0.28.0"
python -m pip install \
  https://github.com/troycheng/vllm-mach/releases/download/v0.1.0a3/vllm_mach-0.1.0a3-py3-none-any.whl
```

The base EXL3 path was validated with [ExLlamaV3 `v1.4.6`](https://github.com/turboderp-org/exllamav3/tree/v1.4.6) at commit `499890c75d20d8e7c9d061f37189ae611a5c9f0b`. Build it in the environment where vLLM is installed:

```bash
git clone --branch v1.4.6 --depth 1 https://github.com/turboderp-org/exllamav3.git
cd exllamav3
python -m pip install -r requirements.txt
MAX_JOBS=4 python -m pip install .
```

Native BF16 I/O is not part of the `v1.4.6` tag. It requires [ExLlamaV3 Draft PR #330](https://github.com/turboderp-org/exllamav3/pull/330), currently published at commit [`d0094bc`](https://github.com/troycheng/exllamav3/tree/d0094bc922bcf2d6cf5e948ba35f347adda3a6ca). To build that revision instead:

```bash
git fetch origin pull/330/head:pr-330
git checkout d0094bc922bcf2d6cf5e948ba35f347adda3a6ca
MAX_JOBS=4 python -m pip install .
```

Release validation used the same BF16 API in a `v1.4.6`-based wheel. PR #330 was rebased afterward, so the public commit above is not byte-identical to the tested wheel. [B12X](https://github.com/local-inference-lab/b12x) is an optional prefill backend.

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

vllm serve malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --revision ab3a91a13813df8096cb4c1d560ed3669035d0cf \
  --quantization exl3 \
  --tensor-parallel-size 2
```

The validated CUDA Graph configuration also enables QKV MGEMM and primes EXL3 kernels before capture:

```bash
export VLLM_PLUGINS=mach
export EXL3_QKV_MGEMM=1
export EXL3_BF16_IO=1
export VLLM_EXL3_GRAPH_DECODE=1

vllm serve malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --revision ab3a91a13813df8096cb4c1d560ed3669035d0cf \
  --quantization exl3 \
  --tensor-parallel-size 2 \
  --compilation-config \
  '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
```

`EXL3_BF16_IO=1` requires the PR #330 build above. Leave it unset when using the official `v1.4.6` tag.

Release `0.1.0a3` also offers opt-in M24/M32 decode paths and a sampling metadata patch. The true-M32 kernel requires a separate native build, and the sampling patch must be applied to vLLM. See [experimental decode paths](docs/experimental-decode.md) for configuration and validation limits.

For prefill, vLLM Mach can dispatch eligible K6 matrices to B12X and use ExLlamaV3's fused reconstruction plus Hadamard path:

```bash
export VLLM_EXL3_B12X_MIN_M=128
export VLLM_EXL3_B12X_N_RANGE=5120-36864
export VLLM_EXL3_B12X_ANY_BITS=1
export VLLM_EXL3_PREFILL_FUSED_RECONSTRUCT_MIN_M=128
```

Install B12X before enabling its route. Fused prefill reconstruction uses `reconstruct_had_slice` from ExLlamaV3 and falls back to the regular reconstruction path when the symbol is unavailable.

### EXL3/MXFP6

Enable the Qwen3.8-27B profile in the same environment as the EXL3 serve command:

```bash
export VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b
```

After applying the matching vLLM and FlashInfer patches, enable the fused TP2 collective with:

```bash
export VLLM_MACH_EXL3_MXFP6_FUSED_AR_NORM_MXFP8=1
```

The hybrid profile keeps `lm_head` and unmatched projections on EXL3. It routes MLP and attention output projections to MXFP6 for all row counts. QKV and QKVZ projections use MXFP6 for prefill calls with at least 128 rows and EXL3 for decode.

## MXFP6 integration

[`mxfp6_sm120`](https://github.com/Nekofish-L/mxfp6_sm120) owns MXFP6 packing, MXFP8 activation quantization, W6A8 GEMM, and workspace management. vLLM Mach handles vLLM registration, checkpoint metadata, tensor-parallel slices, projection routing, CUDA Graph lifecycle, and the optional FlashInfer AllReduce/RMSNorm/MXFP8 boundary. The hybrid profile requires both packages.

## Validation

Release `0.1.0a3` passed 60 package tests with 3 skipped. Its Qwen3.8-27B TP2 service check enabled the new decode options alongside the EXL3/MXFP6 profile and fused FlashInfer collective, captured all seven graph sizes, and passed the existing 40-task regression suite with no new failures. Exact output agreement with the stored reference was 33/40.

The detailed test matrix and historical fused-collective measurements are in [the validation record](docs/validation.md). They are integration evidence, not a general performance claim.

## Limitations

The current release does not provide routed MoE execution, a GDN kernel replacement, `lm_head` conversion, or validated support for additional models and GPU architectures.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Performance results must identify the model, checkpoint format, runtime versions, topology, request shape, graph mode, correctness criterion, and baseline.

## License

vLLM Mach is licensed under [Apache-2.0](LICENSE). Derived notices are listed in [NOTICE](NOTICE). External runtimes retain their own licenses. This project is independent of the vLLM project.
