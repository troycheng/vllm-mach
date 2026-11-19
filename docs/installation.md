# Setup and usage

Mach builds on the official vLLM 0.34.0 image for RTX 5090 (SM120). The same source checkout also supports vLLM 0.33.0.

## Build an image

```bash
git clone https://github.com/troycheng/vllm-mach.git
cd vllm-mach
docker buildx build --load --build-arg MAX_JOBS=2 \
  -f deploy/Dockerfile -t vllm-mach:mxfp6 .
```

Use `deploy/Dockerfile.mxfp8` for 2B/4B MXFP8 or `deploy/Dockerfile.fp8` for an existing 4B block-FP8 checkpoint. The build pins the official Linux amd64 base image, checks dependency hashes, and compiles against its existing PyTorch and CUDA toolkit. Increase `MAX_JOBS` to use more build memory and CPU cores.

To build on 0.33.0, add `--build-arg VLLM_IMAGE=vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3`. The installer selects the source patches for the installed vLLM version.

## Runtime versions

[`vllm_mach.stack`](../src/vllm_mach/stack.py) is the version source used by the package, build scripts and startup checks.

| Component | Version |
|---|---|
| Python | 3.12 in the serving image |
| vLLM | 0.34.0 or 0.33.0 |
| PyTorch | 2.14.0+cu130 |
| Triton | 3.8.0 |
| FlashInfer Python / cubin | 0.7.3 on vLLM 0.34, 0.7.2.post1 on 0.33 |
| CUTLASS Python DSL | 4.8.0 |
| CUDA Python bindings | 13.5.0 on vLLM 0.34, 13.4.4 on 0.33 |
| b12x | 1.5.1 |
| MXFP6/MXFP8 native library | 0.3.2 |
| Mach native extensions | 0.2.0 |
| CUDA compiler | 13.0.88 |
| NCCL | 2.30.7 |

The Docker images and source installer build the native library from [`dd9752f`](https://github.com/Nekofish-L/mxfp6_sm120/commit/dd9752fb7743012136e32e49aecd4eb9b7dd6d94).

The wheel records the actual Torch/CUDA ABI, Python/toolchain identity and both native library hashes in `mxfp6/build_info.json`. The accompanying external build receipt, `mxfp8-build.json`, records source and patched CUTLASS fingerprints, build options and the wheel hash. The shared [native build entry](../deploy/build-mxfp8.py) produces both libraries and the external receipt. Mach's Dockerfiles also restore the Torch-required NCCL version and install the pinned build dependencies.

The 0.34 build installs OpenTelemetry 1.45.2. The bundled external LMCache 0.5.7.post1 connector requires OpenTelemetry 1.40 or earlier, so the Mach image omits that connector. The local vLLM KV cache is built into vLLM.

## vLLM 0.34 qualification

RTX 5090 serving throughput in output tokens/s with 3,000 input tokens and 1,000 output tokens per request. The 2B and 4B models use one GPU. The 27B and 35B models use two GPUs with tensor parallelism. Prefix caching is disabled.

| Model and profile | Concurrency | vLLM 0.33 | vLLM 0.34 | Change |
|---|---:|---:|---:|---:|
| 2B MXFP8, FP32 state | 16 | 5,157.45 | 5,161.78 | +0.08% |
| 2B MXFP8, FP32 state | 64 | 9,840.45 | 9,865.12 | +0.25% |
| 4B MXFP8, FP32 state | 16 | 2,684.51 | 2,777.18 | +3.45% |
| 4B MXFP8, FP32 state | 64 | 4,796.72 | 4,933.82 | +2.86% |
| 4B block-FP8, FP32 state | 32 | 3,008.32 | 3,064.28 | +1.86% |
| 27B Dense MXFP6 FULL, FP16 state | 32 | 1,727.78 | 1,726.29 | −0.09% |
| 35B-A3B MXFP6, FP16 state | 32 | 3,362.91 | 3,354.01 | −0.26% |
| 35B-A3B MXFP6 compact, FP32 state | 128 | 4,902.56 | 4,915.76 | +0.27% |

The 2B results use the geometric mean of two runs per version on the same GPU, in 0.33 / 0.34 / 0.34 / 0.33 order. Both versions used BF16 KV, Triton attention and the hybrid head. Each c16 run completed 160 requests and each c64 run completed 320. The 4B MXFP8 results use the geometric mean of two 0.34 runs against the published 0.33 champion. The TP2 references use the geometric mean of two 0.33 runs.

With the same KV memory budget, 2B capacity increased from 1,470,223 to 1,475,584 tokens. The 4B profiles retain 878,033 tokens for MXFP8 and 508,335 for block-FP8. The TP2 profiles retain 233,472 for Dense, 743,062 for standard MoE and 941,648 for compact MoE.

Fixed-gold tests cover 256 queries and 10,479 token log probabilities per decode shape. For 2B with FP32 state, mean per-query error against BF16 changed from 0.054753 to 0.054145 at M32 and from 0.056550 to 0.056711 at M64. Both 4B profiles match their 0.33 records at M32 and M64. The 27B Dense FULL profile matches all 10,479 gold log probabilities at both M4 and M32 with the production 400-token attention blocks on both TP ranks. The 35B compact profile matches at M128 and M160, including every top-1 token and all 128 free generations tested at M128.

The FP32-state long-output test uses frozen BF16 token trajectories in 16-row batches. On 4B, all 190,251 gold log probabilities and top-1 IDs match the current 0.33 build. On 2B, mean per-query absolute log-probability error against BF16 changes from 0.044164 to 0.044113 across 330,328 tokens. The paired 95% bootstrap interval for the change is −0.000209 to +0.000110.

The final 2B image uses an 8-row tile for the M16 GDN gated norm and reuses FlashInfer's compiled prefill launches for matching calls. It matched the initial 0.34 build bit for bit on 10,479 continuation log probabilities, 256 prefill log probabilities and all 512 greedy token IDs. Decode replayed the physical M16 CUDA Graph. A separate 3,000-token prompt produced the same eight greedy token IDs and exercised the prefill launch reuse.

[27B fixed-gold results with 400-token blocks](data/vllm034-dense-400-gold.json) · [Final 2B throughput, TTFT and precision](data/vllm034-2b-followup.json) · [Initial migration data, including the original −1.32% at c16](data/vllm034-qualification.json) · [Full vLLM 0.33 concurrency curves](performance.md)

## Install into an existing container

Start from the image digest in [deploy/Dockerfile](../deploy/Dockerfile). From the Mach checkout inside that container:

```bash
python deploy/install-build-deps.py
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

The [MoE FULL benchmark command](qwen35-moe.md) specifies the capture sizes,
64-sequence limit, 8 GiB KV cache per rank and FP16 recurrent state.

| Switch | Default | Effect |
|---|---|---|
| `--fp16-ssm` | off | Use FP16 recurrent state |
| `--hybrid-head` / `--no-hybrid-head` | on | One switch for compact greedy hybrid and eligible top-k/presence sampling. The 4B K2048 complete head remains installed when off. `--nvfp4-lm-head` and `--lm-head-sampling` (and their `--no-` forms) are aliases. Unfiltered and top-p-only sampling retain the original sampler. See [admission and validation](lm-head-sampling.md) |
| `--lossless-prefill` / `--owner-prefill` | on for Dense; off for MoE | Enable the corresponding Dense prefill path; use `--no-` to disable it |
| `--gdn-persistent` / `--gdn-ba-overlap` | on | Enable the corresponding GDN decode optimization; use `--no-` to disable it |
| `--fused-ar-norm` | on at TP2 | Fuse AllReduce/residual/RMSNorm; `--no-fused-ar-norm` also disables its MXFP8 producer |
| `--fused-ar-quant` | on for Dense; off for MoE | Fuse AR/GemmaRMSNorm/MXFP8; use `--no-fused-ar-quant` to disable only this producer |
| `--verify-prefill` | off | Run diagnostic comparisons; exclude from throughput measurements |
| `--dry-run` | off | Print the resolved command/environment without loading or validating the runtime |

The FULL profile stores recurrent state in FP16 instead of FP32. Hybrid head
search is enabled in both profiles and refines candidates with the BF16 head.
See [head behavior and coverage](native-mxfp6.md) and
[GDN coverage](gdn-decode.md) for supported shapes and numerical behavior.

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
