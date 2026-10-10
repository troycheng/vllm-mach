# Qwen3.5-2B MXFP8 champion

`qwen35-2b-mxfp8-champion-v1` serves Qwen3.5-2B on one RTX 5090 (SM120), Linux x86-64 and Python 3.12. It has its own checkpoint builder, launcher and worker profile. Use a self-contained checkpoint reconstructed from the [public BF16 model](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc); no private quantized checkpoint, calibration fixture or replacement-code asset is needed. The [model guide](mxfp8-2b-model.md) describes the exact conversion and file identities.

The current update includes `mixed-lean-stockmath-w4-v1` in the package. The same launcher enables it automatically; no experimental startup hook or extra model asset is needed.

## Build and prepare

Build the shared MXFP8 image from this repository on a Linux host with Docker Buildx and the NVIDIA container runtime. `Dockerfile.mxfp8` pins the vLLM base, installs the hashed build dependencies and builds the official MXFP8 kernel source. It also installs Mach and its MXFP8 runtime source profile; there is no separate 2B Dockerfile or experimental wheel to install.

```bash
docker buildx build --load --build-arg MAX_JOBS=2 \
  --build-arg MXFP6_REVISION=15a56aa2774552d0584d8fc6b621b41dc173f30b \
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
  --host 0.0.0.0 --served-model-name q35-2b-study
```

After startup, `http://localhost:8000/v1/models` lists `q35-2b-study`. The launcher checks the installed package versions and runtime sources, hashes the full checkpoint, and checks the worker's graph and operator state before serving. `--print-command` displays its underlying vLLM arguments without starting a worker. `--quality-rows 4`, `32` or `64` selects a separate fixed-row numerical check mode; use the default production mode for serving and throughput measurements.

The production contract is TP1, text-only, non-speculative inference with BF16 activations, BF16 KV and the complete BF16 vocabulary head, FP32 SSM, FlashAttention 2, `max_num_seqs=160`, `max_model_len=16384`, `max_num_batched_tokens=2048` and a 19 GiB KV pool. Native MXFP8 projections, ordered FP32 GDN, small-row BF16 BA and pinned asynchronous GDN reset-ID uploads run under the 2B profile. The graph policy preserves FULL decode captures at 4/8/16/24/32/48/64/96/128/160 rows and routes exact 2048-token prefill or mixed batches to PIECEWISE mode. Batches above 160 tokens with a size other than 2048 use NONE without padding. Prefix caching and LoRA are outside this profile.

## Measurement boundary

The shared benchmark client supports the 2B service. In another terminal, run the fixed 3000-input / 1000-output, six-concurrency protocol:

```bash
docker run --rm --network host \
  --entrypoint vllm-mach-mxfp8-bench \
  -v "$PWD/mxfp8-2b-runs:/runs" \
  vllm-mach:mxfp8-champion \
  --base-url http://127.0.0.1:8000 --model q35-2b-study \
  --outdir /runs/sixpoint-01
```

Choose a new output directory for each run. The [benchmark protocol](mxfp8-benchmark.md) describes the 1,040 scored requests, warmup, generated token IDs and recorded metrics.

The first 2B study measured a **+9.05%** median paired six-point throughput gain against community block-FP8 on GPU1. A second, independent study on GPU7 measured a further **+2.56%** median paired gain for pinned asynchronous GDN reset-ID uploads against the *first 2B champion*. Each study used 3000 input / 1000 output tokens at c4, c16, c24, c32, c48 and c64, with four controlled pairs. These are different comparisons on different GPUs; their percentages are not a directly measured combined gain over community FP8. The pinned-ID change kept the checkpoint and floating-point kernels unchanged, and its fresh M32 real-text logprobs matched the first champion token for token. Those logprob checks measure numerical fidelity, not task accuracy.

The packaged release was checked on October 9 against the selected champion on the same RTX 5090. Each arm completed all 1,040 requests and 1,040,000 output tokens. Pooled throughput was **5,676.98 vs 5,655.59 output tok/s (+0.38%)** in this one-pair packaging regression, preserving the selected performance.

| Concurrency | Selected champion tok/s | Packaged profile tok/s | Difference |
|---|---:|---:|---:|
| 4 | 1389.63 | 1390.27 | +0.05% |
| 16 | 4001.18 | 4030.85 | +0.74% |
| 24 | 5457.49 | 5420.44 | -0.68% |
| 32 | 6525.35 | 6540.77 | +0.24% |
| 48 | 7713.55 | 7827.72 | +1.48% |
| 64 | 8534.71 | 8540.10 | +0.06% |

Fresh fixed-token replay at M4, M32 and M64 matched the selected implementation's logprobs exactly: 256 queries and 10,479 scored tokens at each shape, with zero repeat difference. M4/M32 used the frozen records; M64 used a fresh same-GPU reference. BF16 MAE was 0.048942 / 0.048984 / 0.048233 respectively. The free-running HTTP comparison produced identical text in 937 of 1,040 requests. Numerical fidelity is assessed by the fixed-token replay with controlled physical rows.

The CPU suite passed 200 tests and 60 subtests, with 10 optional-dependency skips on macOS. All nine builder tests also passed with PyTorch on Linux. The [machine-readable results](data/mxfp8-2b-champion-20261009.json) include point metrics, input/output counts, source result hashes, GPU clocks and the fidelity record hashes.

## Mixed decode update (October 10)

Mixed prefill/decode batches now retain ordered W4 state for 24–160 decode requests. Only the prefill slots are materialized before stock convolution and prefill; the mixed recurrent kernel uses the stock mixed path's FP32 normalization and update order. Pure decode, slot allocation/reset and the BF16 head/KV stay on their existing paths. The runtime receipt identifies this implementation as `gdn.mixed_decode.protocol = mixed-lean-stockmath-w4-v1` and includes source hashes and eager/capture-construction counters.

The following comparison uses the previous packaged release (PR #9, `574f704`) as its reference on the same RTX 5090, with 3000 input / 1000 output tokens and four independent pairs per point. Throughput is total output tokens divided by total scored duration for each arm.

| Concurrency | Previous release tok/s | Mixed decode tok/s | Throughput change | Mean TTFT change |
|---|---:|---:|---:|---:|
| 4 | 1407.84 | 1405.65 | −0.15% | +0.55% |
| 16 | 4071.66 | 4065.49 | −0.15% | +1.75% |
| 24 | 5515.67 | 5491.55 | −0.44% | +0.82% |
| 32 | 6546.32 | 6655.77 | +1.67% | −5.42% |
| 48 | 7888.47 | 7990.93 | +1.30% | −2.59% |
| 64 | 8680.30 | 8813.15 | +1.53% | −3.43% |

c4/c16/c24/c48 use 40/160/120/240 requests per arm, in AB/BA/BA/AB order. c32/c64 use three waves (96/192 requests per arm), from the earlier target and independent audit. Every scored window is retained. c48 improved in all four pairs (+0.56% to +2.12%); c24 ranged from −1.47% to −0.016%. These runs establish a c32–c64 improvement and no new low-concurrency gain; their different request counts are kept separate in the [machine-readable results](data/mxfp8-2b-mixed-gdn-20261010.json).

With matched observers, fixed-token replay at cohort32 preserved all 10,479 gold logprobs across 256 queries, with zero repeat difference. Same-call checks covered 18 layers and 72 mixed calls with bitwise-equal BF16 output and FP32 state; a 20-step arithmetic handoff checked pure/mixed W4 transitions. This update does not add full-model fidelity measurements for other cohort sizes. The JIT body is unchanged from that qualified implementation, and CPU regression tests cover the mixed boundary, prefill-slot isolation and fallback behavior.

The installed-wheel regression completed 984 scored requests: c4 −0.34%, c32 +0.04% and c64 −0.46% against the selected experimental implementation. c4/c32 have one pair; c64 has two opposing-order pairs (−0.97% and +0.06%). All runs used the same checkpoint, native binary, graph sizes and KV capacity. Fresh cohort32 replay matched the earlier released reference on all 10,479 gold logprobs, with zero repeat difference. Its 32 tiny differences from the observed experimental record exactly reproduced the previously recorded reference/observer difference (MAE 2.80e-8); the result file retains both comparisons. The packaged kernel also passed eight-step bitwise output/state replay, and the MXFP8 CPU suite completed 147 tests with five optional-dependency skips.

See [runtime installation](mxfp8-installation.md) for the shared source profile and build pins, and [model preparation](mxfp8-2b-model.md) for source assets, conversion details and validation.
