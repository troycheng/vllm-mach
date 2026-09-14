# Installation

The current source profile builds on vLLM 0.29.0, PyTorch 2.13.0, ExLlamaV3 1.5.0, FlashInfer 0.6.18, B12X 1.3.0 and MXFP6 SM120 0.2.1. The Python wheel alone does not include native kernels, runtime patches or model assets. Use the complete image build below.

## Build the image

Requirements: Linux x86-64, Docker with Buildx and NVIDIA Container Toolkit, and the CUDA 13.2 toolkit at `/usr/local/cuda-13.2`. The build targets SM120. CUDA 13.0 for the older lossless collective is already in the official vLLM base image; the EXL3 and owner extensions use 13.2.

```bash
git clone https://github.com/troycheng/vllm-mach.git
cd vllm-mach
git rev-parse HEAD  # record the source revision with your results
docker buildx build --load \
  --build-context cuda132=/usr/local/cuda-13.2 \
  --build-arg MAX_JOBS=8 \
  -f deploy/Dockerfile -t vllm-mach:local .
```

The Dockerfile installs the pinned optional packages, applies Mach's EXL3 BF16 patch, builds the native extensions, then installs the matching vLLM/FlashInfer patches. It also rebuilds MXFP6 0.2.1 from commit `7c891d07b65ce2f4e5e8e10a6934c1a298755b8d` with its pinned CUTLASS revision: the published PyPI wheel does not match PyTorch 2.13's dispatcher ABI. It starts from the official vLLM image, not an existing Mach installation. Do not apply the old vLLM 0.28 instructions on top. To use a local mirror of the same official image, pass `--build-arg VLLM_IMAGE=your-mirror/vllm-openai:v0.29.0`.

Build identities are recorded at `/opt/mach-build/installed.json` in the image. The native wheels are retained in `/opt/mach-build/wheels`. Runtime libraries are compiled during the build; FlashInfer/B12X also specialize some kernels on the first service start.

## Prepare model assets

The service uses three directories:

| Directory | Content |
| --- | --- |
| `models/exl3` | Qwen3.8-27B K5/K6 EXL3 checkpoint, including tokenizer and config |
| `models/mxfp6` | Matching original MXFP6 checkpoint; not a W6 cache reconstructed from EXL3 |
| `models/rank64` | Selected 48-layer NVFP4 weights, aware64 coefficients, static scales and manifest |

Get the EXL3 checkpoint at the tested revision:

```bash
mkdir -p models
docker run --rm --entrypoint hf -v "$PWD/models:/models" vllm-mach:local \
  download malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --revision ab3a91a13813df8096cb4c1d560ed3669035d0cf \
  --local-dir /models/exl3
```

Mach does not distribute the matching MXFP6 and rank64 assets. The [quantization guide](quantization.md) provides two generation commands, using the existing converter and calibration method. The assembled generation workflow has not been rerun end to end. Operators with existing assets can use the [export tools](#export-existing-assets), which package tensors without quantizing weights or fitting compensation parameters.

The runtime validates each rank64 tensor, its layer/TP rank, the originating weight and static RMSNorm scale. It does not need the full official BF16 checkpoint at serving time. The selected route uses NVFP4 only for the 48 gate/up projections at physical M32; other rows and projections retain their EXL3/MXFP6 paths.

## Start the complete profile

Choose two idle RTX 5090 GPUs with working P2P access. This example uses devices 0 and 1. Models are read-only; the separate cache volume keeps startup compilations across restarts.

```bash
docker run --rm --name mach --gpus '"device=0,1"' \
  --ipc=host --network=host \
  -v "$PWD/models:/models:ro" \
  -v mach-kernel-cache:/root/.cache \
  vllm-mach:local \
  --model /models/exl3 \
  --mxfp6-checkpoint /models/mxfp6 \
  --rank64-bundle /models/rank64 \
  --owner-prefill --max-num-seqs 48 \
  --host 127.0.0.1 --port 8000
```

This command selects TP2, BF16 activations and attention KV, FP16 recurrent state, 8192 context, 4096 chunked prefill, 8,218,214,400 KV bytes per GPU, and FULL_DECODE_ONLY graph sizes 1/2/4/8/16/24/32/48. Prefix caching is off. Temporal M24, existing BA overlap and the lossless-prefill fallback are enabled. The model is registered as `Qwen3.8-27B`, matching the benchmark client.

Owner-local MLP replicates the other TP shard at 32 even-numbered layers, adding 3,342,336,000 weight bytes per GPU without reducing the KV budget. The selected owner route handles physical M3000–4096; other shapes retain the existing path. The separate M32 P1 candidate is not enabled.

For the earlier checkpoint-hybrid path, omit `--rank64-bundle` and `--owner-prefill`, and use `--max-num-seqs 32`. Native dependencies and vLLM/FlashInfer patches remain the same. Add `--dry-run` to print the complete resolved command and profile flags without loading the model.

## Check and measure

Wait for service readiness:

```bash
curl --fail http://127.0.0.1:8000/health
```

Then run a short 3k/1k installation check:

```bash
python3 -m pip install aiohttp
python3 docs/data/benchmark_fixed_token_contract.py \
  --base-url http://127.0.0.1:8000 --model Qwen3.8-27B \
  --input-tokens 3000 --output-tokens 1000 \
  --max-concurrency 32 --num-prompts 64 \
  --warmup-requests 32 --warmup-output-tokens 128 \
  --contract-seed 20260910 --request-seed-base 2026091000 \
  --json-out mach-c32-short.json
```

This is a short installation check. The README's reference curves use the longer [benchmark protocol](benchmarks.md#reproduce-and-verify); keep those request counts and conditioning when comparing against them.

For local acceptance, `--diagnostics --host 127.0.0.1` enables explicit status and rank64-check endpoints. `--verify-owner-forwards 2` additionally compares two real owner prefill forwards against the original operations. Verification adds synchronization and must finish before timing. These options are not needed for normal serving.

Take rank64 snapshots while all 32 requests are decoding, using two different input batches. Confirm that each snapshot falls after every request's first token and before any request completes. Do not sample after the requests drain: smaller graphs can reuse the M32 graph's intermediate buffers.

## Export existing assets

These commands package the already selected tensors; they do not fit new parameters or upload anything. Output directories must be new.

```bash
python3 tools/export_model_assets.py \
  --model /path/to/paired-mxfp6 --output models/mxfp6 \
  --source-model https://modelscope.cn/models/Qwen/Qwen3.8-27B \
  --source-revision e823e888ae179eb3be02c1a48899c4f828371376
python3 tools/import_rank64_bundle.py \
  --weights /path/to/saved_weights/mask48_manifest.json \
  --residuals /path/to/aware64/manifest.json \
  --scales /path/to/rmsnorm_static_nvfp4_scales_v1.json \
  --output "$PWD/models/rank64"
python3 tools/export_model_assets.py --model models/mxfp6 --verify
```

The rank64 importer requires CPU PyTorch. It preserves NVFP4 bytes and aware64 tensor values, removes unselected cols32 coefficients and writes a portable manifest. Keep the original model license with any redistributed model assets.
