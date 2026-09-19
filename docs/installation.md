# Setup and usage

Start from `vllm/vllm-openai:v0.29.0`. Run the following steps inside the
container, using its existing Python environment.

## 1. Check CUDA Toolkit

The official image includes CUDA 13.0 compilation tools. Check the compiler:

```bash
/usr/local/cuda/bin/nvcc --version
```

If CUDA 13.0 compilation tools or required development headers are missing,
install the toolkit:

```bash
apt update
apt install -y cuda-toolkit-13-0
```

## 2. Install MXFP6 kernels

Build MXFP6 from the official project's pinned revision
[`cd4e964c391fcb8aaf1a27d28a63d778e3a38ece`](https://github.com/Nekofish-L/mxfp6_sm120/commit/cd4e964c391fcb8aaf1a27d28a63d778e3a38ece).
It retains package version **0.2.1** and includes the scale-initialization,
SwiGLU and GDN producer operations used by the current Dense profile.
The original 0.2.1 release does not contain these operations.

```bash
git clone https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
git checkout cd4e964c391fcb8aaf1a27d28a63d778e3a38ece
git submodule update --init --depth 1 third_party/cutlass
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/build_wheel.sh
pip install --no-deps --force-reinstall dist/mxfp6_sm120-0.2.1-*.whl
cd ..
```

Build against the container's existing PyTorch with CUDA 13.0. The Mach
Docker build uses this same source revision and checks the required native
operations after installation. Both `vllm-mach-install` and `vllm-mach-serve`
reject an older extension that lacks them, even if its package version is
0.2.1. Upgrade both projects together; see the
[producer-fusion results](dense-producer-fusion.md).

## 3. Build and install Mach

```bash
git clone https://github.com/troycheng/vllm-mach.git
cd vllm-mach

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

# Compile and install the Dense prefill extensions.
MAX_JOBS=8 pip install --no-build-isolation --no-deps \
  ./native/lossless_prefill ./native/owner_prefill

# Install Mach and apply its runtime patches.
pip install --no-build-isolation --no-deps .
vllm-mach-install --apply
```

Both prefill extensions are required for Dense default/full. MoE-only
installations can skip the prefill build. To run Dense without these extensions,
pass `--no-lossless-prefill --no-owner-prefill` when serving.

## 4. Launch a model

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
head by default. With the pinned MXFP6 build, eligible decode also uses fused
SwiGLU/MXFP8 and GDN norm/MXFP8 producers. Add `--fp16-ssm --nvfp4-lm-head`
for Dense full. If the prefill
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
`--fp16-ssm --nvfp4-lm-head` for MoE full. No additional kernels are needed
beyond MXFP6 0.2.1. See the [MoE guide](qwen35-moe.md) for validation and
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
| `--nvfp4-lm-head` | off | Use NVFP4 candidate search with BF16 refinement for eligible greedy decode |
| `--no-lossless-prefill` / `--no-owner-prefill` | on for Dense; off for MoE | Disable the corresponding Dense prefill path |
| `--no-gdn-persistent` / `--no-gdn-ba-overlap` | on | Disable the corresponding GDN decode optimization |
| `--no-fused-ar-norm` | on | Disable TP2 AllReduce/residual/RMSNorm fusion |
| `--verify-prefill` | off | Run diagnostic comparisons; exclude from throughput measurements |
| `--dry-run` | off | Print the resolved command/environment without loading or validating the runtime |

Full changes numerical behavior: FP16 reduces recurrent-state precision, and
NVFP4 candidate search may change the BF16 head's argmax. The NVFP4 path supports
eligible greedy decode with at most 32 rows; unsupported requests use the
ordinary head/sampling path. See [NVFP4 head details](native-mxfp6.md) and
[GDN coverage](gdn-decode.md) for restrictions and fallbacks.

The optional Dense producer controls `VLLM_MACH_FUSED_SWIGLU_QUANT` and
`VLLM_MACH_FUSED_GDN_QUANT` accept `auto` (default), `0` (disabled), or `1`
(require the operation when the model geometry is eligible). See
[admission and fallback behavior](dense-producer-fusion.md#admission).

Additional memory per GPU for the validated models:

| Feature | Dense | MoE |
|---|---|---|
| Owner prefill | 3.11 GiB | Not used |
| NVFP4 head (BF16 head retained) | 341 MiB | 136.4 MiB |

Allow for these approximate allocations when sizing the KV cache. Set the same
`--kv-cache-memory-bytes` when comparing profiles.
