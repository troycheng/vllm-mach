# Qwen3.5-2B MXFP8 champion

The [single MXFP8 benchmark](mxfp8-benchmark.md) contains the current per-concurrency SSM/KV matrix.

The **Triton attention + K128 hybrid head** profile runs on one RTX 5090, TP1. See the [per-concurrency comparisons with block FP8](mxfp8-benchmark.md#qwen35-2b).

The public checkpoint builder reconstructs all 438 tensors from pinned BF16 weights. Native MXFP8, BF16 KV, FP32 GDN and the complete BF16 embedding/head weights are retained; eligible greedy requests use the hybrid head described below.

## Build and prepare

Build the shared MXFP8 image from this repository on a Linux host with Docker Buildx and the NVIDIA container runtime. `Dockerfile.mxfp8` pins the vLLM base, installs the hashed build dependencies and builds the official MXFP8 kernel source. It also installs Mach and its MXFP8 runtime source profile; there is no separate 2B Dockerfile or experimental wheel to install.

```bash
docker buildx build --load --build-arg MAX_JOBS=2 \
  -f deploy/Dockerfile.mxfp8 -t vllm-mach:mxfp8-champion .
```

Download the complete pinned BF16 snapshot to a real local directory, including its tokenizer and chat template. For example, with `huggingface_hub` installed on the host:

```bash
python3 -m pip install huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-2B",
    revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    local_dir="qwen35-2b-bf16",
)
PY
sha256sum qwen35-2b-bf16/model.safetensors-00001-of-00001.safetensors
find qwen35-2b-bf16 -type l -print
```

The BF16 shard must hash to `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`, and the `find` command must print nothing. The builder independently verifies the pinned source files and rejects symlinks. Run preparation on CPU; create the output parent first, and choose a new `/output/model` directory:

```bash
mkdir -p "$PWD/mxfp8-2b-output" "$PWD/mxfp8-2b-runs"
docker run --rm --entrypoint vllm-mach-mxfp8-2b-prepare \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v "$PWD/qwen35-2b-bf16:/input/bf16:ro" \
  -v "$PWD/mxfp8-2b-output:/output" \
  vllm-mach:mxfp8-champion \
  --bf16 /input/bf16 --output /output/model
sha256sum mxfp8-2b-output/model/model.safetensors
```

The output is one self-contained, 438-tensor checkpoint with its own tokenizer. Its complete shard SHA256 must be `115948c161e2cc8d02a4e7e580cefcf3c02513410a2af56e80dd63f9f48a1671`. The builder verifies every tensor and the whole file before publishing the directory; `mach_2b_model.json` records the result. A full CPU reconstruction from the pinned public BF16 snapshot has reproduced every tensor and the whole-file hash. The package can also validate a copied checkpoint at server startup.

## Serve

Override the shared image's default 4B entrypoint to select the 2B launcher. Give the run directory a writable mount; it receives `launch.json`, worker receipts and fresh compilation caches.

```bash
docker run --rm --gpus '"device=0"' --ipc=host -p 8000:8000 \
  --entrypoint vllm-mach-mxfp8-2b-serve \
  -v "$PWD/mxfp8-2b-output/model:/model:ro" \
  -v "$PWD/mxfp8-2b-runs:/runs" \
  vllm-mach:mxfp8-champion /model --run-dir /runs/serve \
  --host 0.0.0.0 --served-model-name q35-2b-study \
  --attention-backend TRITON_ATTN --hybrid-head
```

After startup, `http://localhost:8000/v1/models` lists `q35-2b-study`. The launcher checks the installed package versions and runtime sources, hashes the full checkpoint, and checks the worker's graph and operator state before serving. `--print-command` displays JSON containing the underlying vLLM `argv` and a reproducible Mach `launch_command`, including the head selection, without starting a worker. `--quality-rows 4`, `32` or `64` selects a separate fixed-row numerical check mode; use the default production mode for serving and throughput measurements.

The production contract is TP1, text-only, non-speculative inference with BF16 activations, BF16 KV and complete BF16 vocabulary-head weights, FP32 SSM, Triton attention, `max_num_seqs=160`, `max_model_len=16384`, `max_num_batched_tokens=2048` and a 19 GiB KV pool. Eligible greedy head calls use the candidate/refinement path described below. Native MXFP8 projections, ordered FP32 GDN with mixed D1–160 decode, BF16 BA M1–8/M16 and pinned asynchronous GDN reset-ID uploads run under the 2B profile. Exact FULL16/FULL24 captures also use the complete PDL dependency chain. The graph policy preserves FULL decode captures at 4/8/16/24/32/48/64/96/128/160 rows and routes exact 2048-token prefill or mixed batches to PIECEWISE mode. Batches above 160 tokens with a size other than 2048 use NONE without padding. Prefix caching and LoRA are outside this profile.

Add `--ssm-dtype bfloat16` to select BF16 state with ordered W4 GDN, now the default backend for both state dtypes. FP32 remains the default storage dtype. Use `--no-ordered-bf16-ssm` to select stock GDN/FLA for BF16 state. Grouped BF16 GDN and GDN PDL remain disabled by default.
See [BF16 SSM measurements and configuration](mxfp8-benchmark.md).

## Champion components

| Component | Selected behavior |
|---|---|
| Projection | Native MXFP8 W8A8 with prepared workspaces; unchanged checkpoint |
| BA projection | BF16 GEMV for M1–8 and native BF16 M16; original linear fallback elsewhere |
| GDN | Ordered FP32 W4 recurrence; mixed D1–160 coverage; explicit state materialization and slot lifecycle |
| CPU submission | Pinned asynchronous reset-ID uploads |
| Dependency scheduling | Full norm → quantization → GEMM → SwiGLU PDL chains for actual FULL16/FULL24 captures |
| Graph routing | FULL decode at 4/8/16/24/32/48/64/96/128/160; exact2048 PIECEWISE; other large tails use NONE without padding |
| Attention | `TRITON_ATTN` with BF16 KV |
| Head | NVFP4 sidecar proposes K128 candidates; original BF16 weights refine scores; eligible greedy decode M≤64 |

These mechanisms are already in the public package. The October 11 code qualification matched all 101 installed package source hashes.

## Hybrid head and Triton attention

`--attention-backend TRITON_ATTN` selects attention independently. Hybrid LM-head is now enabled by default; use `--no-hybrid-head` with the default attention backend to reproduce the preceding FA2/full-head profile. This one switch controls both greedy and admitted unseeded top-k decode, including optional top-p and presence-only penalties. The NVFP4 sidecar is built after graph capture and remains separate from the tied BF16 embedding/head parameter. Prefill, unsupported rows, unfiltered/top-p-only sampling, logprobs, frequency/repetition penalties, grammar and other unsupported processing use the complete head. Worker receipts and `vllm_mach.mxfp8.two_b.hybrid_head.inspect_worker` record setup and successful dispatches.

## Precision

The [current precision table](performance.md#qwen35-2b-precision) reports October 20 M32/M64 measurements for FP32 SSM and optional BF16 SSM, using Mach `afcb4cf` and native `ce0cd05`. Each configuration covers 256 queries and 10,479 fixed gold tokens against the frozen BF16 M32 anchor. Logprob requests use the complete BF16 head. The [data](data/mxfp8-fidelity-20261020.json) include query-level errors, confidence intervals, repeated-cohort checks and a compressed bundle of raw logprobs and runtime receipts.

The October 11 six-shape results used production-geometry replay; this update uses the packaged fixed-M quality mode. Their original configurations and measurements remain in `historical_validation.two_b` in the [benchmark data](data/mxfp8-benchmark.json).

The October 11 generation check covered 1,536 requests and 196,608 generated tokens. Across 129,878 hybrid decisions, no BF16 winner was omitted and no lower-scored token was selected; one M24 decision selected a different token tied at BF16 score 20.375. M48/M64 full batches then used the complete BF16 head. The current greedy head supports eligible rows through M64; its newer generation and sampling checks are in the [head validation](lm-head-sampling.md).

The [fixed token manifest](data/fidelity-samples.json), checkpoint reconstruction and validation tool are public. See [reproducing the precision measurements](performance.md#reproduce-mxfp8-fidelity).

## Benchmark and revisions

In a second terminal after readiness, run the shared client against the 2B service. Use a fresh output directory:

```bash
docker run --rm --network host --entrypoint vllm-mach-mxfp8-bench \
  -v "$PWD/mxfp8-2b-runs:/runs" vllm-mach:mxfp8-champion \
  --base-url http://127.0.0.1:8000 --model q35-2b-study \
  --outdir /runs/sixpoint-01
```

The [benchmark](mxfp8-benchmark.md) defines generated requests, warmup and 1,040 scored requests per configuration. Its JSON retains per-concurrency results and historical numerical qualification.

| Identity | Revision / record |
|---|---|
| Precision runtime | Mach `afcb4cf6bbab0dfa615fe6a2096e958623219ef6`; native `ce0cd05a007504853d2bf3963a183e3c76f3c75b` |
| Native MXFP8 runtime | Official decode/PDL source pinned in `deploy/Dockerfile.mxfp8`; binary SHA recorded in the precision data |
| Current configuration matrix | [Per-concurrency benchmark](mxfp8-benchmark.md) and [consolidated data](data/mxfp8-benchmark.json) |

Current throughput and precision appear together in the [model performance report](performance.md#qwen35-2b).
