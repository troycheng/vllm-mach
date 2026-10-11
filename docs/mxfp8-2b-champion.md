# Qwen3.5-2B MXFP8 champion

`qwen35-2b-mxfp8-champion-v1` serves Qwen3.5-2B on one RTX 5090 (SM120), Linux x86-64 and Python 3.12. It has its own checkpoint builder, launcher and worker profile. Use a self-contained checkpoint reconstructed from the [public BF16 model](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc); no private quantized checkpoint, calibration fixture or replacement-code asset is needed. The [model guide](mxfp8-2b-model.md) describes the exact conversion and file identities.

The selected champion combines **Triton attention and the K128 hybrid greedy head** with native BF16 BA M16, mixed GDN D1–160 coverage and the complete FULL16/FULL24 PDL dependency chain. The serving example below selects this configuration explicitly. Its [six-point comparison](#current-six-point-snapshot-october-11) improves output throughput by **1.77–13.83%** over the preceding FA2/full-head champion.

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

## Hybrid head and Triton attention

The serving example enables `--attention-backend TRITON_ATTN --hybrid-head`. These switches independently select attention and head behavior; omitting both reproduces the preceding FA2/full-head profile. The checkpoint, complete BF16 head weights, BF16 KV, FP32 SSM, cache capacity and scheduler geometry stay the same.

For eligible greedy decode batches of at most 32 rows, the head uses an NVFP4 sidecar to propose 128 vocabulary candidates, refines their scores with the original BF16 weights and selects the result. The sidecar is built after model graph capture and kept separate from the tied BF16 embedding/head parameter. Prefill, larger batches, stochastic sampling, logprob requests, penalties, grammar and other logits processing use the original full head. Worker receipts record the selected backend and head setup; `vllm_mach.mxfp8.two_b.hybrid_head.inspect_worker` also returns actual successful dispatch counts.

Fixed-token replay covers the original 256 queries / 10,479 gold tokens at each of M4/M16/M24/M32/M48/M64, with zero first-cohort repeat difference. M4/M32/M48/M64 use the packaged production path; M16/M24 reuse the qualified diagnostics and packaged-generation parity checks. The following MAEs use the same frozen BF16 M32 anchor across all cohorts.

| Physical rows | Previous FA2/full-head MAE | Triton + hybrid MAE | MAE change |
|---|---:|---:|---:|
| M4 | 0.04904216 | 0.04897897 | −0.13% |
| M16 | 0.04915289 | 0.04928728 | +0.27% |
| M24 | 0.04941809 | 0.04782973 | −3.21% |
| M32 | 0.04822663 | 0.04946948 | +2.58% |
| M48 | 0.04865848 | 0.04899862 | +0.70% |
| M64 | 0.04845331 | 0.04875407 | +0.62% |

The packaged head was also checked on actual free-running generation: 256 real-text prompts per cohort, 128 output tokens per request, totaling 1,536 scored requests and 196,608 generated tokens. M48 uses another 32 unscored padding requests. Across **129,878 actual hybrid decisions**, no BF16 winner was omitted and no lower-scored token was chosen. One M24 decision selected a different token tied at BF16 score 20.375. Actual M48/M64 batches above 32 rows used the complete BF16 head; smaller tails could use hybrid. Random sampling, logprobs and penalty requests also exercised the original fallback.

The M32 MAE increase is **0.00124284 (+2.58%)**, caused by the attention backend: fixed-token logprob requests use the complete BF16 head, and restoring FA2 while retaining hybrid reproduced every previous M32 gold logprob exactly. Its query-paired 95% interval is [−0.00002244, +0.00251540]. The M64 English subset increased by about 0.00294; its exploratory, unadjusted interval is above zero. These measurements quantify numerical fidelity, not task accuracy or bitwise equivalence.

The [six-point result data](data/mxfp8-2b-triton-hybrid-sixpoint-20261011.json) include per-shape precision, domain intervals, source hashes, generation counters and all paired throughput observations. The [earlier c16/c24 qualification](data/mxfp8-2b-hybrid-triton-20261011.json) retains the prototype and independent package comparisons. All 101 package source hashes match the tested implementation.

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

The October 9 packaged release was checked against its selected champion on the same RTX 5090. Each arm completed all 1,040 requests and 1,040,000 output tokens. Pooled throughput was **5,676.98 vs 5,655.59 output tok/s (+0.38%)** in that one-pair packaging regression.

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

## Base optimizations: BA M16, low-D and FULL PDL

The current integration contains three accepted changes:

- BF16 BA uses the native `mxfp8_sm120::bf16_ba_m16` operator for aligned, contiguous CUDA BF16 X[16,2048] and W[32,2048]. It preserves the original weight and returns a fresh output; M1–8 retain the audited GEMV, and other shapes retain `F.linear`.
- Mixed GDN eligibility expands from D24–160 to D1–160. Stock convolution and prefill, ordered FP32 W4 recurrence, prefill-slot materialization and state ownership use the same arithmetic and lifecycle.
- The complete PDL dependency chain applies only while constructing actual FULL16/FULL24 graphs, with exact source/launch contracts. Eager calls, other FULL rows, PIECEWISE and NONE retain the standard path. The selected native GEMM ABI uses `bool use_pdl`; this change does not alter quantization or the checkpoint.

Startup receipts require native BA M16 support and captured BA launches, and verify the PDL captures. The M16/M24 fidelity experiments use production geometry with teacher cohorts 16/24; the separate `--quality-rows` mode retains only 4/32/64. The official native build pins commit `a4cdc1b23a72cdedecb29707534fa4e3a595933f` (based on `29c48018`) and applies CUTLASS patches 0001/0003/0004/0005/0006 and runs dispatcher/schema checks without a GPU, including BA M16 and the GEMM `bool use_pdl` ABI.

The accepted experimental champion `ba-m16-mixed-lowd-fullpdl-m16m24-v1` was compared with `ba-m16-mixed-lowd-v1` on the same RTX 5090 GPU0. The 3000-input / 1000-output protocol used five request waves, two target pairs and two independent audit pairs at each point. Every one of the 1,600 scored requests and 1,600,000 output tokens completed.

| Concurrency | Previous experimental champion tok/s | Current experimental champion tok/s | Throughput change | Mean TTFT previous → current ms |
|---|---:|---:|---:|---:|
| 16 | 4243.236814 | 4290.796126 | +1.120826% | 143.092509 → 144.064691 |
| 24 | 5531.482205 | 5612.540083 | +1.465392% | 175.356289 → 176.469588 |

All four pairs improved at both points. Target and audit gains were +1.158940%/+1.082738% at c16 and +1.422104%/+1.508886% at c24. This measures the combined new native library plus complete PDL change against a baseline that already contained BA M16 and low-D. That qualification measured c16/c24. The current six-point snapshot below completes coverage of the installed version.

Before this comparison, GPU3 independently measured the old D24 guard against D1 at c24: 5524.694995 → 5646.827819 tok/s (+2.210671%), with 480 requests completed. That experiment used a different card and baseline; its gain must not be multiplied into the GPU0 comparison.

M16/M24 fixed-token replay each covered 256 queries and 10,479 gold logprobs, with identical ordered records to the preceding champion and zero repeat differences. M16 final qualification ran on GPU0; M24 and reference records include GPU3, with migration identities recorded in the experiment audit. Component checks covered 100 cases and 96 changing/restored-input and graph checks per row. The packaged implementation also passed production-geometry M16/M24 replay on GPU0: each cohort size reproduced all 256 query records and 10,479 gold logprobs exactly, with zero repeat differences. The official BA kernel matched the prototype across 72 real-weight eager/graph cases spanning all 18 BA layers.

The October 11 installed-wheel check compared the accepted prototype and package on the same GPU0, using 3000 input / 1000 output tokens, three waves per point and AB/BA order. All **480 requests and 480,000 output tokens** completed with no run-monitor errors.

| Concurrency | Accepted prototype tok/s | Installed package tok/s | Throughput change | Mean TTFT prototype → package ms |
|---|---:|---:|---:|---:|
| 16 | 4294.184229 | 4294.860551 | +0.015750% | 170.377469 → 169.455197 |
| 24 | 5547.555227 | 5601.402037 | +0.970640% | 218.870209 → 218.125896 |

The individual AB/BA changes were −0.125690%/+0.157222% at c16 and +1.369248%/+0.574028% at c24. Throughput pools output tokens over scored durations; mean TTFT is weighted by request count, and per-run p99 values are retained without averaging. This is a packaging recheck against the accepted prototype, separate from the historical optimization comparison above; the percentages are not multiplied.

The Mach CPU suite passed **224 tests and 72 subtests**, with 12 optional-dependency skips. Native GPU tests passed **15 tests**, with two skips, alongside the 72 real-weight BA eager/graph cases. Packaged M16/M24 precision each reproduced **256 queries and 10,479 gold logprobs exactly, with zero repeat differences**. The [machine-readable results](data/mxfp8-2b-full-pdl-20261011.json) record the historical experiment and package recheck separately, including paired windows, scored counts, durations and source evidence hashes.

## Current six-point snapshot (October 11)

The selected **Triton + hybrid** champion uses Mach runtime `6e390a1a559d4464fb2b67fe09d855f1b442f763`, merged in `b8d55d86e5f10670cfdfc3d9cdcb6cda020e1ceb`, and native runtime `a4cdc1b23a72cdedecb29707534fa4e3a595933f`. The preceding FA2/full-head configuration and selected configuration were paired on GPU0 with 3000 input / 1000 output tokens. Community block-FP8 values are the original GPU1 historical reference.

| Concurrency | Community FP8 tok/s | Previous FA2/full-head tok/s | Triton + hybrid tok/s | vs. community FP8 | vs. previous |
|---|---:|---:|---:|---:|---:|
| c4 | 1261.89 | 1395.31 | 1588.27 | +25.86% | +13.83% |
| c16 | 3814.33 | 4295.10 | 4801.96 | +25.89% | +11.80% |
| c24 | 5069.68 | 5594.33 | 6209.52 | +22.48% | +11.00% |
| c32 | 5870.22 | 6595.49 | 7259.99 | +23.67% | +10.08% |
| c48 | 6837.60 | 7896.46 | 8036.47 | +17.53% | +1.77% |
| c64 | 7578.86 | 8702.23 | 9008.34 | +18.86% | +3.52% |

Each arm uses 40/160/120/160/240/320 scored requests at c4/c16/c24/c32/c48/c64. c16/c24 reuse the independent packaged-code audit; c4/c32/c48/c64 complete coverage of that same version. c48 also has a reverse-order pair, with gains of +1.91% and +1.63% before pooling. All **2,560 requests and 2,560,000 output tokens** completed. Throughput is total output tokens divided by total scored duration; per-run p99 values remain separate. The native binary, checkpoint, package sources and serving geometry match across these runs, with no monitor errors. Ratios against community FP8 compare separate cards and studies.

| Concurrency | Mean TTFT previous → selected ms | Mean TPOT previous → selected ms |
|---|---:|---:|
| c4 | 92.57 → 98.90 | 2.776 → 2.421 |
| c16 | 124.03 → 134.09 | 3.601 → 3.198 |
| c24 | 174.79 → 192.81 | 4.108 → 3.666 |
| c32 | 205.22 → 224.49 | 4.636 → 4.174 |
| c48 | 273.69 → 277.45 | 5.785 → 5.678 |
| c64 | 349.98 → 333.26 | 6.974 → 6.743 |

A later c32 check of the same runtime measured **7216.58 tok/s for Triton + hybrid** and 7210.06 tok/s for FA2 + hybrid, with 160 requests per arm. That separate observation is retained alongside the six-point paired comparison above. The [current result JSON](data/mxfp8-2b-triton-hybrid-sixpoint-20261011.json) records both studies, precision coverage and evidence hashes; the [preceding FA2/full-head snapshot](data/mxfp8-2b-release-snapshot-20261011.json) remains available for comparison.

## Earlier mixed decode qualification (October 10)

The earlier qualified mixed prefill/decode implementation retained ordered W4 state for 24–160 decode requests. Only the prefill slots are materialized before stock convolution and prefill; the mixed recurrent kernel uses the stock mixed path's FP32 normalization and update order. Pure decode, slot allocation/reset and the BF16 head/KV stay on their existing paths. The runtime receipt identifies this implementation as `gdn.mixed_decode.protocol = mixed-lean-stockmath-w4-v1` and includes source hashes and eager/capture-construction counters.

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
