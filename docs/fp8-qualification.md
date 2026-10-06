# Block-FP8 qualification — October 6, 2026

The all-feature `qwen35-4b-block-fp8-v1` profile preserves the stock fixed-gold records at M4/M32/M64 and improves measured output throughput by 5.97–9.03% across c4/c16/c32/c64. Each arm completes all 680 scored requests. These are one stock/all pair per point, with no statistical significance or separate feature-gain claim. See the [machine-readable receipt](data/fp8-qualification-20261006.json) and [build/launch guide](fp8-profile.md).

The measurements use one RTX 5090 / SM120, TP1, vLLM 0.29.0, Torch 2.13.0+cu130, BF16 activations/KV/full head and FP32 SSM. Production retains maxlen8192, maxseq128, scheduler2048, 19 GiB KV, no prefix cache and the framework's FULL/PW graph policy. The candidate enables n64, ordered, SiLU and mixed FA2 together, with its source-matched `production` RMS compilation recipe.

## Serving

Both arms use the same public uniform-token-ID workload, 3000 input and 1000 output tokens, deterministic sampling, prompt/request seeds, prewarm and internal warmup. Protocol SHA-256 is `6e08e63c957a73508abd5d96ce181b1ed42c2b5de06a9e4bc37d776f4a798190`. Scored duration excludes warmup, session setup and file writes; output throughput is completed output tokens divided by that duration. TTFT runs from semaphore admission to the first text-bearing streaming event; p99 uses linear interpolation.

| Concurrency | Scored requests per arm | Stock output tok/s | All features output tok/s | Change |
|---|---:|---:|---:|---:|
| c4 | 40 | 678.538 | 739.843 | +9.03% |
| c16 | 160 | 2018.890 | 2143.596 | +6.18% |
| c32 | 160 | 2834.840 | 3004.000 | +5.97% |
| c64 | 320 | 3363.733 | 3598.838 | +6.99% |

| Concurrency | Mean TTFT stock → all (ms) | Change | p99 TTFT stock → all (ms) | Change |
|---|---:|---:|---:|---:|
| c4 | 184.98 → 192.46 | +4.05% | 303.63 → 307.45 | +1.26% |
| c16 | 261.81 → 260.16 | -0.63% | 1187.41 → 1186.63 | -0.07% |
| c32 | 469.07 → 451.17 | -3.82% | 2558.88 → 2476.31 | -3.23% |
| c64 | 830.50 → 771.87 | -7.06% | 5493.98 → 5156.29 | -6.15% |

c4 throughput improves while mean/p99 TTFT rise slightly; the other three points reduce both TTFT measures. No claim is made for c24/c48. The raw client summaries say `complete:false` because `selected_points` contains four points of its original six-point protocol. Both say `all_selected_points_succeeded:true`; this flag is not a failure or a claim that all six points ran. Historical prototype gains are not combined with these results.

Start the candidate with the [public launcher](fp8-profile.md), then run the same attached client independently against each arm, using a new output directory:

```bash
vllm-mach-fp8-bench --base-url http://127.0.0.1:8000 \
  --model q35-fp8-study --outdir ./all-fourpoint --points 4 16 32 64
```

For the stock arm, run the same server settings and checkpoint/tokenizer with `VLLM_PLUGINS=''` and isolated vLLM/Inductor/Triton caches, then repeat the client into `./stock-fourpoint`. The benchmark does not start or reconfigure services. Its [public protocol](mxfp8-benchmark.md) and generated prompts are independent of the optimization profile.

## Precision

Each fixed physical-row check at M4, M32 and M64 covers 256 queries and 10,479 forced-gold raw decode logprobs. All three final checks have zero changed values, MAE/max absolute difference zero, repeat maximum difference zero, and record files byte-identical to their matching stock reference. Their record SHA-256 identities are in the receipt. This validates the tested short-gold contract, not general task accuracy or arbitrary models/contexts. The evaluation corpus is private and is not bundled.

The initial all-feature M4 check failed because two unchanged RMS sources selected different reduction configurations: 5405/10479 values differed, MAE 0.013750718. Exact recipes for `quality4`, `quality32`, `quality64` and `production` now match source AST, geometry, signature/constants, specialization, FP flags and SM120 before pinning the corresponding stock launch choices. Related drift fails closed; unrelated kernels retain normal heuristics. No frozen compiled cache is distributed. The passing records above follow this repair.

The fixed-M driver uses maxlen1024, maxseq equal to M, token budgets 2048/8192/16384 at M4/32/64, 4/4/19 GiB KV and FULL_AND_PIECEWISE capture containing the requested physical row count. These settings differ from dynamic production serving. Component byte/graph tests and the [upstream FA2 source diagnostic](fp8-fa2-source-check.md) remain separate evidence; the original upstream head is not qualified by Mach results.

## Identity

The tested image ID is `sha256:9eabff48de8a713589928824aef0b353dd92cab4e8e5fa05810b804441fe33e9`; the final source build is `sha256:f1423984dfd1d79d1de4bcb72a6ec7bc7e51ea3c29d8d1a5cc218d66a93fc55e`. The rebuild after three lines of trailing-whitespace cleanup has identical SHA-256 hashes for all 85 compared installed Python, JSON, patch and native-library files, including both native libraries. The sole native build-metadata change is its source hash, which now records the cleaned source. All other native metadata fields match. The receipt retains both image identities, the compared file hashes and the comparison receipt hash; no new GPU result is claimed for the rebuild.

Checkpoint identities below come from the previously frozen environment receipt. This run reused the same read-only model mount; it does not claim a new full-checkpoint hash pass or infer a download URL. The complete previously recorded file list is in the machine-readable receipt.

| Checkpoint file | Previously frozen SHA-256 |
|---|---|
| `config.json` | `6b687fd37cac954fc6c009f5e9e003ce449bc4320c7a2ffc383e5a73080ee2f3` |
| `model-00001-of-00002.safetensors` | `c058f0646ce4f6344e84a93ba29a6ad628bca9b7d1bd043a37bd26b815dd0e9b` |
| `model-00002-of-00002.safetensors` | `a56f6ed8658d05e172b78d19217a83ff78d825a51137924dd808f497cef770db` |
| `model.safetensors.index.json` | `ec8e03795756adc5445122a210b9b9c018169570fdb51a219ea29daf10b7f2e8` |
