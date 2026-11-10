# Setup and usage

Mach 0.3 runs on vLLM 0.33.0 and RTX 5090 (SM120). The three Dockerfiles build the native extensions and install the matching runtime profile.

## Build an image

```bash
git clone https://github.com/troycheng/vllm-mach.git
cd vllm-mach
docker buildx build --load --build-arg MAX_JOBS=2 \
  -f deploy/Dockerfile -t vllm-mach:mxfp6 .
```

Use `deploy/Dockerfile.mxfp8` for 2B/4B MXFP8 or `deploy/Dockerfile.fp8` for an existing 4B block-FP8 checkpoint. The build pins the official Linux amd64 base image, checks dependency hashes, and compiles against its existing PyTorch and CUDA toolkit. Increase `MAX_JOBS` to use more build memory and CPU cores.

## Runtime versions

[`vllm_mach.stack`](../src/vllm_mach/stack.py) is the version source used by the package, build scripts and startup checks.

| Component | Version |
|---|---|
| Python | 3.12 in the serving image |
| vLLM | 0.33.0 |
| PyTorch | 2.14.0+cu130 |
| Triton | 3.8.0 |
| FlashInfer Python / cubin | 0.7.2.post1 |
| CUTLASS Python DSL | 4.8.0 |
| CUDA Python bindings | 13.4.4 |
| b12x | 1.5.1 |
| MXFP6/MXFP8 native library | 0.3.0 |
| Mach native extensions | 0.2.0 |
| CUDA compiler | 13.0.88 |
| NCCL | 2.30.7 |

The native library is built from [`1aa0955`](https://github.com/Nekofish-L/mxfp6_sm120/commit/1aa0955aa0bedf6f2bbc87be367301fe5f7c4c8b). The wheel records the actual Torch/CUDA ABI, Python/toolchain identity and both native library hashes in `mxfp6/build_info.json`. The accompanying external build receipt, `mxfp8-build.json`, records source and patched CUTLASS fingerprints, build options and the wheel hash. Mach's Dockerfiles also restore the Torch-required NCCL version and install the pinned build dependencies.

## Install into an existing container

Start from the image digest in [deploy/Dockerfile](../deploy/Dockerfile). From the Mach checkout inside that container:

```bash
python -m pip uninstall -y \
  opentelemetry-exporter-otlp-common opentelemetry-exporter-http-transport
python -m pip install --no-deps --require-hashes \
  -r deploy/requirements-mxfp8.txt
python deploy/build-mxfp6.py
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2 \
  python deploy/install.py --cuda-home /usr/local/cuda-13.0
python -m pip check
```

The installer builds the Dense owner-prefill, lossless-prefill and AR/Norm extensions, installs Mach, and applies the native runtime profile. For a runtime-only reinstall, use `python deploy/install.py --runtime-only`. For one extension, use `--native-only --native-part ar_norm --cuda-home /usr/local/cuda-13.0`.

The MXFP8 and block-FP8 image builds install their own dependencies and source profiles. See [MXFP8 installation](mxfp8-installation.md) and [block-FP8 setup](fp8-profile.md).

## Launch a model

The following examples are alternatives, not commands to run simultaneously.
Use a local checkpoint directory with the correct `config.json`; the launcher
uses it to select the model-specific profile. Mach does not ship model weights.
The Dense checkpoint is available as
[nekofish/Qwen3.8-27B-MXFP6](https://huggingface.co/nekofish/Qwen3.8-27B-MXFP6).

### Qwen3.8-27B Dense

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm-mach-serve \
  --model /models/Qwen3.8-27B-MXFP6 \
  --served-model-name mach \
  --host 127.0.0.1 --port 8000
```

Dense enables both prefill paths and retains FP32 recurrent state and the BF16
head parameter by default, with hybrid candidate search enabled. With the pinned MXFP6 build, eligible decode also uses fused
SwiGLU/MXFP8, GDN norm/MXFP8 and [AR/GemmaRMSNorm/MXFP8](ar-norm-mxfp8.md) producers. Add `--fp16-ssm` for Dense full. If the prefill
extensions are not installed, add `--no-lossless-prefill --no-owner-prefill`;
this changes the profile used for the Dense benchmark results.

### Qwen3.5-35B-A3B MoE

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm-mach-serve \
  --model /models/Qwen3.5-35B-A3B-MXFP6 \
  --served-model-name mach \
  --host 127.0.0.1 --port 8000
```

MoE is detected automatically and disables Dense prefill paths. Add
`--fp16-ssm` for MoE full. It uses the same pinned MXFP6
source build as Dense. See the [MoE guide](qwen35-moe.md) for validation and
benchmark settings.

## Configuration and memory

Standard vLLM options can be appended to `vllm-mach-serve`. The optimized
profile is validated at TP2/PP1.

| Setting | Dense default | MoE default |
|---|---|---|
| Tensor / pipeline parallelism | TP2 / PP1 | TP2 / PP1 |
| Maximum sequences | 32 | 32 |
| Maximum model length | 16,384 | 16,384 |
| Scheduled-token budget | 4,096 | 2,048 |
| Compilation / graphs | `NONE` / `FULL_DECODE_ONLY` | `VLLM_COMPILE` / `FULL_AND_PIECEWISE` |
| Graph capture sizes | 1, 2, 4, 8, 16, 24, 32 | vLLM default |
| Prefix caching | Disabled | Disabled |
| Attention backend | `TRITON_ATTN` | `TRITON_ATTN` |

The [MoE benchmark](qwen35-moe.md) uses explicit graph capture sizes and
`--max-num-seqs 64`; see that guide to reproduce its configuration.

| Switch | Default | Effect |
|---|---|---|
| `--fp16-ssm` | off | Use FP16 recurrent state |
| `--hybrid-head` / `--no-hybrid-head` | on | One switch for compact greedy hybrid and eligible top-k/presence sampling. The 4B K2048 complete head remains installed when off. `--nvfp4-lm-head` and `--lm-head-sampling` (and their `--no-` forms) are aliases. Unfiltered and top-p-only sampling retain the original sampler. See [admission and validation](lm-head-sampling.md) |
| `--no-lossless-prefill` / `--no-owner-prefill` | on for Dense; off for MoE | Disable the corresponding Dense prefill path |
| `--no-gdn-persistent` / `--no-gdn-ba-overlap` | on | Disable the corresponding GDN decode optimization |
| `--no-fused-ar-norm` | on | Disable TP2 AllReduce/residual/RMSNorm and its MXFP8 producer |
| `--no-fused-ar-quant` | on for Dense; off for MoE | Disable only the AR/GemmaRMSNorm/MXFP8 producer |
| `--verify-prefill` | off | Run diagnostic comparisons; exclude from throughput measurements |
| `--dry-run` | off | Print the resolved command/environment without loading or validating the runtime |

Full changes numerical behavior: FP16 reduces recurrent-state precision, and
NVFP4 candidate search may change the BF16 head's argmax. The NVFP4 path supports
eligible greedy decode with at most 32 rows; unsupported requests use the
ordinary head/sampling path. See [NVFP4 head details](native-mxfp6.md) and
[GDN coverage](gdn-decode.md) for restrictions and fallbacks.

The optional Dense producer controls `VLLM_MACH_FUSED_SWIGLU_QUANT`,
`VLLM_MACH_FUSED_GDN_QUANT` and `VLLM_MACH_FUSED_AR_QUANT` accept `auto` (default), `0` (disabled), or `1`
(require the operation when the model geometry is eligible). See
[admission and fallback behavior](dense-producer-fusion.md#admission). The launcher sets `VLLM_MACH_FUSED_AR_QUANT` explicitly from `--[no-]fused-ar-quant`; direct Python users may set the environment variable. Larger request concurrency remains supported by the original paths; the AR/MXFP8 optimization does not change KV capacity or the scheduler limit.

Additional memory per GPU for the validated models:

| Feature | Dense | MoE |
|---|---|---|
| Owner prefill | 3.11 GiB | Not used |
| NVFP4 head (BF16 head retained) | 341 MiB | 136.4 MiB |

Allow for these approximate allocations when sizing the KV cache. Set the same
`--kv-cache-memory-bytes` when comparing profiles.


## MXFP8 single-GPU serving

The same Mach plugin registers `Mxfp8Sm120LinearKernel` for compressed-tensors
MXFP8 checkpoints. It loads E4M3 weights and E8M0/32 scales directly, repacks only
the scales, and dynamically quantizes BF16 activations with the shared native
quantizer. W8A8 uses its own planned workspace and the existing pre-capture
warmup hooks. The W6A8 selector and producer fusions keep their existing path.

After installing the runtime and applying the Mach profile above:

```bash
CUDA_VISIBLE_DEVICES=0 vllm-mach-serve \
  --model /data1/models/Qwen3.5-4B-MXFP8 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 --port 8000
```

For a local `mxfp8-quantized` checkpoint, the launcher defaults to
`--quantization compressed-tensors`, BF16 and TP1. MXFP6 retains Quark and TP2
as defaults. The MXFP8 profile disables the default TP2 all-reduce and dense
prefill fusions and uses vLLM's default compilation and CUDA Graph policy.
Look for `Using Mxfp8Sm120LinearKernel for MXFP8 GEMM` in the worker log.

The native W8A8 kernel requires SM120, BF16 output, and positive weight N/K
dimensions divisible by 128. The tested checkpoint is dense Qwen3.5-4B with
TP1. For the complete 2B/4B serving configurations, use the dedicated [MXFP8 launchers](mxfp8-champion.md).
