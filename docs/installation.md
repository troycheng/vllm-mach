# Native MXFP6 installation

This profile starts from official vLLM 0.29.0. It requires no modified vLLM
checkout, EXL3 checkpoint, ExLlamaV3, rank64 assets or GDN source overlay.
The Python wheel excludes the historical EXL3 provider.

## Dependencies

Use Linux x86-64, Python 3.12 and two RTX 5090 GPUs (SM120, 32 GiB each).
The runtime contract is vLLM 0.29.0, PyTorch 2.13.0, FlashInfer Python/cubin
0.6.18, CUTLASS DSL 4.6.2, and mxfp6-sm120 0.2.1.
The NVFP4 head uses FlashInfer's built-in `b12x` backend; the standalone
`b12x` package and the `vllm[b12x]` extra are not required.
Native extension binaries must match the installed PyTorch/CUDA ABI.

A source installation in a dedicated environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install vllm==0.29.0
# Follow mxfp6_sm120's build prerequisites, including its patched CUTLASS.
# Build against the torch installed above, not against an isolated older torch.
uv pip install --no-build-isolation /path/to/mxfp6_sm120
uv pip install .
vllm-mach-install
vllm-mach-install --apply
```

Use the [mxfp6_sm120 build instructions](https://github.com/Nekofish-L/mxfp6_sm120)
and its 0.2.1 release. A prebuilt wheel is acceptable only when its native ABI
matches. `python -c 'import torch, mxfp6; mxfp6.load_library()'` checks loading.
The provided [image build helper](../deploy/build-mxfp6.py) pins both MXFP6 and
CUTLASS revisions and applies the required CUTLASS patches.

The first installer command is a dry run. It validates dependency versions,
patch applicability for all 13 vLLM files, and the local FlashInfer IPC patch. The second applies
the staged changes. Repeat installation is a no-op. Stop services before
patching and restart them afterwards. Incompatible source edits or partial
installations are rejected; unrelated edits outside patch hunks may remain.
Use a fresh environment when migrating from EXL3.
No source patch is applied at Python import time.

The profile and its manifest are included in the Mach wheel under
`vllm_mach/mxfp6/profile`. Neither a source checkout of Mach nor the optimization
checkout of vLLM is needed after installation.

Model checkpoint: [nekofish/Qwen3.8-27B-MXFP6 on Hugging Face](https://huggingface.co/nekofish/Qwen3.8-27B-MXFP6).

## Base acceleration

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm-mach-serve \
  --model /models/Qwen3.8-27B-MXFP6 \
  --host 127.0.0.1 --port 8000
```

The launcher selects Quark, BF16, TP2, the V2 runner, TRITON_ATTN, no prefix
caching, a 4096 scheduled-token budget, and full decode graphs up to 32 rows.
It enables native MXFP6, fused AllReduce/residual/RMSNorm, and compact BF16
greedy argmax communication, plus small-batch persistent and large-batch BA
overlap GDN decode. The checkpoint's original recurrent-state dtype
and BF16 LM head are preserved by default.

Use `--dry-run` to print flags without loading a model. Standard vLLM flags,
such as `--max-model-len`, `--max-num-batched-tokens`, host and port, can be
appended. The supported optimization geometry remains TP2/PP1 and 32 sequences.

## Optional acceleration

Only build the prefill extensions if you enable their corresponding flags:

```bash
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=8 \
  uv pip install --no-build-isolation --no-deps ./native/lossless_prefill
CUDA_HOME=/usr/local/cuda-13.2 MAX_JOBS=8 \
  uv pip install --no-build-isolation --no-deps ./native/owner_prefill
```

These extensions retain their existing compiler contracts: CUDA 13.0 for
lossless prefill and CUDA 13.2 for owner prefill, using FlashInfer 0.6.18 headers.
They do not depend on EXL3.

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm-mach-serve \
  --model /models/Qwen3.8-27B-MXFP6 \
  --fp16-ssm --lossless-prefill --owner-prefill --nvfp4-lm-head \
  --kv-cache-memory-bytes 8218214400 \
  --host 127.0.0.1 --port 8000
```

FP16 SSM changes recurrent-state precision. NVFP4 uses a 128-candidate coarse
search followed by BF16-weight refinement and is not guaranteed to preserve
the full BF16 head's argmax. Greedy decode without processors/logprobs uses this
path; stochastic sampling, penalties, masks, structured output, prefill/mixed
batches and logprob requests use the ordinary sampling path.

Owner prefill supports 512–4096 rows, with shape-dependent communication and
MLP dispatch. M1024/M1052 retain the lossless codec when enabled. The selected
32 MLP replicas add about 3.11 GiB per GPU; the NVFP4 head adds about 341 MiB.
Use explicit equal KV bytes for comparisons.

`--verify-prefill` enables the imported diagnostic checks. The normal launcher
keeps lossless-prefill graph capture and breakable graphs disabled, matching
the final serving profile. The warmed lossless graph implementation remains
available through the imported environment switches for separate experiments.

## Docker

The image builds MXFP6 and both optional prefill extensions, without EXL3:

```bash
docker buildx build --load \
  --build-context cuda132=/usr/local/cuda-13.2 \
  --build-arg MAX_JOBS=8 -f deploy/Dockerfile -t vllm-mach:local .
docker run --rm --gpus '"device=0,1"' --ipc=host --network=host \
  -v /path/to/model:/models/mxfp6:ro \
  -v mach-kernel-cache:/root/.cache vllm-mach:local \
  --model /models/mxfp6 --fp16-ssm --lossless-prefill \
  --owner-prefill --nvfp4-lm-head --kv-cache-memory-bytes 8218214400
```

Docker Buildx and a local CUDA 13.2 toolkit are required. The vLLM image supplies
the CUDA 13.0 toolkit used by the lossless extension.

See [integration and validation](native-mxfp6.md) before interpreting benchmark
results. Historical EXL3 guides apply only to earlier releases.

## Default GDN decode

These paths ship in the Mach wheel; update the wheel and keep the current native
runtime profile. Do not install the old FlashInfer/EXL3 overlay.

```bash
# FP32 state: persistent M1/2/4/8 and BA overlap M16/24/32
vllm-mach-serve --model /models/Qwen3.8-27B-MXFP6

# FP16 state: persistent M1/2/4/8 and BA overlap M16/24/32
vllm-mach-serve --model /models/Qwen3.8-27B-MXFP6 --fp16-ssm
```

Persistent requires a CUDA development toolkit for its first JIT build. The
source is packaged in `vllm_mach/mxfp6/gdn`; generated files use FlashInfer's
normal cache. Native warmup compiles and allocates scratch before profiling
and graph capture. Both flags default on and support FP32 or FP16 recurrent state.
Persistent specializes its state loads/stores to the allocated cache dtype,
retaining FP32 arithmetic; no state pool conversion occurs at batch boundaries. See [coverage and validation](gdn-decode.md).

Disable the routes individually with `--no-gdn-persistent` and
`--no-gdn-ba-overlap`.
