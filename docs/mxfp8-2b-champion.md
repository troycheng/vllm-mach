# Qwen3.5-2B MXFP8 champion

The selected **Triton attention + K128 hybrid head** profile runs on one RTX 5090, TP1. Across c4/c16/c24/c32/c48/c64 it measures **+17.53–25.89%** against the historical community FP8 baseline and **+1.77–13.83%** against the preceding FA2/full-head champion on the same GPU. See the [six-point data, latency and charts](performance.md#qwen35-2b).

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
| Head | NVFP4 sidecar proposes K128 candidates; original BF16 weights refine scores; eligible greedy decode M≤32 |

These mechanisms are already in the public package. The October 11 code qualification matched all 101 installed package source hashes.

## Hybrid head and Triton attention

`--attention-backend TRITON_ATTN --hybrid-head` independently selects the two components. Omitting both reproduces the preceding FA2/full-head profile. The NVFP4 sidecar is built after graph capture and remains separate from the tied BF16 embedding/head parameter. Prefill, batches above 32, stochastic sampling, logprobs, penalties, grammar and other logits processing use the complete head. Worker receipts and `vllm_mach.mxfp8.two_b.hybrid_head.inspect_worker` record setup and successful dispatches.

## Precision

Each of M4/M16/M24/M32/M48/M64 covers the original 256 queries and 10,479 gold tokens, with zero first-cohort repeat difference. M4/M32/M48/M64 use the packaged production path; M16/M24 reuse the qualified diagnostic replay and packaged-generation parity checks. All use the same frozen BF16 M32 anchor. Logprob requests use the complete BF16 head.

| Physical rows | Previous FA2/full-head MAE | Selected MAE | Change |
|---|---:|---:|---:|
| M4 | 0.04904216 | 0.04897897 | -0.13% |
| M16 | 0.04915289 | 0.04928728 | +0.27% |
| M24 | 0.04941809 | 0.04782973 | -3.21% |
| M32 | 0.04822663 | 0.04946948 | +2.58% |
| M48 | 0.04865848 | 0.04899862 | +0.70% |
| M64 | 0.04845331 | 0.04875407 | +0.62% |

The M32 increase is 0.00124284 (+2.58%), attributed to attention: restoring FA2 with hybrid enabled reproduced every previous gold logprob. M64's English subset increased about 0.00294. These are numerical fidelity measurements; the [source data](data/mxfp8-2b-triton-hybrid-sixpoint-20261011.json) retain per-domain intervals and coverage.

Actual generation used 256 real-text prompts per cohort and 128 output tokens: 1,536 scored requests and 196,608 generated tokens. Across 129,878 actual hybrid decisions, no BF16 winner was omitted and no lower-scored token was selected; one M24 decision chose a different token tied at BF16 score 20.375. M48/M64 full batches use BF16, while smaller tails can use hybrid. The fixed-gold corpus is private; checkpoint reconstruction and the serving benchmark are public.

## Benchmark and revisions

In a second terminal after readiness, run the shared client against the 2B service. Use a fresh output directory:

```bash
docker run --rm --network host --entrypoint vllm-mach-mxfp8-bench \
  -v "$PWD/mxfp8-2b-runs:/runs" vllm-mach:mxfp8-champion \
  --base-url http://127.0.0.1:8000 --model q35-2b-study \
  --outdir /runs/sixpoint-01
```

The [protocol](mxfp8-benchmark.md) defines generated requests, warmup and 1,040 scored requests per complete screen. The selected snapshot additionally includes the reverse c48 pair and earlier qualified c16/c24 package audits; all 2,560 scored requests succeeded. A later c32 observation of the same runtime was 7,216.58 tok/s, retained separately from the primary paired table.

| Identity | Revision / record |
|---|---|
| Mach runtime | `6e390a1a559d4464fb2b67fe09d855f1b442f763`, merged by `b8d55d86e5f10670cfdfc3d9cdcb6cda020e1ceb` |
| Native MXFP8 runtime | `a4cdc1b23a72cdedecb29707534fa4e3a595933f` |
| Current selected result | [Triton + hybrid six-point receipt](data/mxfp8-2b-triton-hybrid-sixpoint-20261011.json) |
| Base changes | [BA M16 / low-D / FULL PDL qualification](data/mxfp8-2b-full-pdl-20261011.json) |
| Earlier studies | [October 9 package](data/mxfp8-2b-champion-20261009.json), [October 10 mixed GDN](data/mxfp8-2b-mixed-gdn-20261010.json), [preceding FA2 snapshot](data/mxfp8-2b-release-snapshot-20261011.json) |

Historical incremental studies use their own GPU and baseline. Their results remain in these receipts; current complete-profile gains are calculated directly from the [performance table](performance.md#qwen35-2b).
