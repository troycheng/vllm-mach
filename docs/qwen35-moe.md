# Qwen3.5-35B-A3B MXFP6

Mach supports the Quark packed E3M2 checkpoint with dynamic per-32 E4M3
activations on two SM120 GPUs. The default launcher uses TP2, BF16 activations,
FP32 recurrent state and full decode CUDA graphs at 1/2/4/8/16/24/32 tokens.

Build `mxfp6-sm120` from revision
`7c891d07b65ce2f4e5e8e10a6934c1a298755b8d` (v0.2.1) against the serving
environment's PyTorch, following [installation](installation.md). The existing
`deploy/build-mxfp6.py` pins this source and CUTLASS; this release already
contains the required MoE kernels. The framework adapters were imported from
the local checkout's newer vLLM example at revision
`cd4e964c391fcb8aaf1a27d28a63d778e3a38ece` and ported to vLLM 0.29.0.

For the local kernel checkout and model:

```bash
# Inside the vLLM 0.29.0 / torch 2.13.0 environment:
python -m pip install --no-deps --no-build-isolation /data/lxy/detailed_benchmark/mxfp6_sm120
python -m pip install --no-deps .
vllm-mach-install --apply
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 vllm-mach-serve \
  --model /data1/models/Qwen3.5-35B-A3B-MXFP6/ \
  --host 127.0.0.1 --port 8000
```

Choose two free GPUs in `CUDA_VISIBLE_DEVICES`. No TP argument is necessary.
Existing dense-profile installations can be upgraded with `vllm-mach-install
--apply`; the separate MoE patch is staged and checked before any writes.

The adapter retains packed expert weights, packs checkpoint scales to the
native layout and uses the package's routed grouped GEMMs for prefill. Eligible
Qwen35 TP2 layers combine the shared expert with the routed weights for the
1–4 token fused router/expert schedule and use the indirect grouped schedule
for 5–96 tokens. Larger batches use the generic routed path. The standard vLLM
runner performs the final TP reduction.

The launcher recognizes `qwen3_5_moe` in the local model configuration and
turns off dense-specific AR/Norm and persistent/BA GDN optimizations. Dense-only
FP16 SSM, owner/lossless prefill and NVFP4 head switches are rejected for this
model. This profile targets text-only, non-speculative TP2/PP1 inference;
dense-model benchmark numbers do not describe its performance.

Startup should report `Using native mxfp6-sm120 grouped MoE backend` and,
when weight geometry matches, `Enabled Qwen3.5-35B TP2 MXFP6 fused small-batch
router/shared-expert schedule`.

## FP8 baseline

Use the sibling checkpoint `/data1/models/Qwen3.5-35B-A3B-FP8` in an
**unpatched official vLLM 0.29.0 environment**. FP8 keeps the official compiler,
CUDA Graph, model-runner and attention-backend defaults. Disable FlashInfer
AllReduce explicitly. All Mach optimizations belong to the MXFP6 arm.

Use the same GPU pair, TP2, BF16 activations, FP32 SSM, request lengths, maximum
context, batch limits and fixed KV budget. Run each server sequentially:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 VLLM_PLUGINS='' \
VLLM_ALLREDUCE_USE_FLASHINFER=0 \
python -m vllm.entrypoints.cli.main serve /data1/models/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 2 --dtype bfloat16 --max-num-seqs 32 \
  --max-model-len 16384 --max-num-batched-tokens 4096 \
  --kv-cache-memory-bytes 8589934592 --no-enable-prefix-caching \
  --generation-config vllm --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 127.0.0.1 --port 8000
```

Pass `--kv-cache-memory-bytes 8589934592` to the MXFP6 launcher as well.
This compares serving profiles: Mach uses its native MoE schedules, V2 runner,
decode graphs and BF16 compact greedy head; FP8 keeps official execution and
sampling defaults. Results do not isolate the MoE kernel alone.

For each server, run the existing benchmark tool with its corresponding model
path. The smoke benchmark uses uniform token IDs, so it measures throughput,
not language quality or the dense profile's ShareGPT workload:

```bash
python tools/benchmark_native_mxfp6.py \
  --base-url http://127.0.0.1:8000 \
  --model /data1/models/Qwen3.5-35B-A3B-MXFP6 \
  --num-prompts 64 --max-concurrency 32 \
  --input-tokens 3000 --output-tokens 1000 \
  --warmup-requests 32 --warmup-output-tokens 128 \
  --contract-seed 20260917 --request-seed-base 2026091700 \
  --json-out mxfp6-c32.json
```

## Validation on September 17, 2026

Both checkpoints completed 35/35 functional checks: 32 concurrent Chinese
chat probes (arithmetic, capital, translation, chemistry), plus full-logprob,
stochastic and token-bias completions. These checks establish basic serving
behavior, not broad language quality or equivalence to BF16.

The default suite and follow-up groups passed all 173 tests. Four TP2 tests
initially skipped in the single-GPU suite were rerun on two GPUs. One existing
owner-prefill test failed once during NCCL communicator initialization and
passed on its isolated retry. MoE routing was compared with independent
per-expert dense GEMMs, including graph replay with changed inputs and expert
IDs. Installer tests cover clean installation, dense-profile upgrade,
idempotency and rejection without writes.

The Python wheel was built and its MoE adapters and patch contents checked.
The installed native kernel binary matches the local kernel checkout's built
wheel; the MoE Python API matches the source checkout. Native CUDA sources and
the Docker image were not rebuilt in this validation.

The retained performance runs use physical GPUs 6/7 sequentially, first FP8,
then MXFP6. Shared settings and exact per-request results are in the
[results manifest](data/qwen35-moe-20260917.json). The baseline imports the
unpatched official vLLM/FlashInfer runtime, disables Mach plugins and
FlashInfer AllReduce, and otherwise keeps official execution defaults:
`VLLM_COMPILE`, FlashAttention 2 and `FULL_AND_PIECEWISE` graphs. Native FP8 MoE
uses the automatically selected DeepGEMM backend. MXFP6 uses the launch profile
described above. Each measurement has one repetition.

| Concurrency | Requests/arm | FP8 output tok/s | MXFP6 output tok/s | FP8 mean TPOT (ms) | MXFP6 mean TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 256.9 | 255.0 | 3.77 | 3.81 |
| 4 | 16 | 694.1 | 725.9 | 5.50 | 5.28 |
| 16 | 32 | 1627.7 | 1825.0 | 9.19 | 8.18 |
| 24 | 48 | 2046.9 | 2291.1 | 10.87 | 9.70 |
| 32 | 64 | 2335.0 | 2622.5 | 12.65 | 11.23 |
