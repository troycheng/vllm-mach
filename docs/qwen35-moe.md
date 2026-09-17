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
for 5–96 tokens. Larger batches use the generic routed path. With AR/Norm
fusion disabled, the standard vLLM runner performs the final TP reduction;
with fusion enabled, the following norm owns it.

The launcher recognizes `qwen3_5_moe` in the local model configuration and
enables native TP2 AllReduce/residual/RMSNorm fusion. The routed and shared
expert output stays local until the following fused norm performs the TP sum;
this preserves the small-batch expert schedules. GDN decode reuses the dense
profile's
persistent route at physical M1/2/4/8 and BA projection overlap at M16/24/32.
The adapter checks the 35B TP2 geometry (hidden 2048, 8 local key heads, 16 local
value heads, packed QKV width 4096) and warms each geometry/state dtype before
CUDA Graph capture. Prefill, mixed batches, speculative decode and unsupported
layouts retain the original vLLM path. Use `--no-gdn-persistent` and
`--no-gdn-ba-overlap` to disable the routes independently.

`--nvfp4-lm-head` enables NVFP4 candidate search followed by BF16 refinement
for eligible greedy requests (at most 32 rows). The original BF16 head remains
available for full logits, logprobs and unsupported sampling options. This is
an approximate candidate search, not a replacement of the stored BF16 head.
It adds approximately 136.4 MiB per GPU at this model's TP2 vocabulary size.
Use `--no-fused-ar-norm` to independently disable communication fusion.

FP16 SSM and owner/lossless prefill switches remain rejected for this model;
serving retains FP32 recurrent state. This profile targets text-only,
non-speculative TP2/PP1 inference;
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
To reproduce the original FP8/MXFP6 comparison below with the current launcher,
add `--no-gdn-persistent --no-gdn-ba-overlap --no-fused-ar-norm` to MXFP6 and
leave `--nvfp4-lm-head` off.
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

## GDN decode migration

The 27B GDN adapter now supports the 35B TP2 geometry. Both routes are enabled
by default; this does not change the MoE expert kernels. This first migration
was measured with AR/Norm fusion and NVFP4 head disabled. Their subsequent
migration is documented below. The September 17 FP8 comparison above was
measured before either change.

Validation for the migration:

- 90 focused routing, native MoE, workspace, installer and deployment tests passed.
- GPU adapter comparisons cover both 27B and 35B at M4/16/24/32. BA overlap
  outputs and updated states match the native composition bitwise.
- Each geometry passed 16 persistent-kernel cases (M1/2/4/8, two state layouts,
  FP32/FP16), including changing-input graph replay, null slots and canaries.
  For 35B the maximum relative output L2 error against the native composition
  was 0.002230; maximum state error was 0.000124. Graph/eager results match
  bitwise. FP16 is a kernel test here, not an enabled 35B serving option.
- The TP2 server prepared 30 GDN layers per rank and captured all seven decode
  sizes. It passed 35/35 concurrent chat and request-fallback smoke checks.
  These checks do not establish broad model quality or BF16 fidelity.

Reproduce the kernel acceptance with:

```bash
python tools/verify_gdn_gpu.py --model 35b --output gdn-35b.json
```

The serving ablation uses the same MXFP6 checkpoint for both arms, two RTX 5090s,
TP2/PP1, vLLM 0.29.0, Torch 2.13.0+cu130, FlashInfer 0.6.18 and MXFP6 SM120
0.2.1. BF16 activations, FP32 SSM, 8 GiB/rank KV allocation, TRITON_ATTN,
FULL_DECODE_ONLY graphs (1/2/4/8/16/24/32), 4096 batched-token limit and all
other launcher settings are held fixed. Baseline adds only
`--no-gdn-persistent --no-gdn-ba-overlap`; the optimized arm uses the GDN
routes. To reproduce these historical arms with the current launcher, add
`--no-fused-ar-norm` to both and leave `--nvfp4-lm-head` off. The optimized
server ran first, followed by a fresh baseline server
on the same GPU pair. Each point has one run, using the same uniform-token
3000-input/1000-output protocol, seeds, request counts and warmups as above.

Output throughput (tokens/s):

| GDN configuration | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| Disabled | 726.6 | 1821.7 | 2284.6 | 2609.9 |
| Persistent + BA overlap | 791.3 | 1841.5 | 2315.9 | 2659.3 |
| Change | +8.9% | +1.1% | +1.4% | +1.9% |

These single-run measurements show the largest gain at c4; the smaller
large-batch changes need repeated runs to distinguish from noise.
[Aggregate results and per-request raw data](data/qwen35-gdn-20260917.json)
also retain the kernel acceptance measurements.

## AllReduce and NVFP4 head

The README uses the same default/full naming as the 27B profile:

| Profile | Enabled optimizations | Launch options |
|---|---|---|
| Mach default | Native MoE schedules, decode graphs, GDN, fused AR/Norm, compact BF16 greedy sampling | No extra optimization flags |
| Mach full | Default plus NVFP4 head candidate search with BF16 refinement | `--nvfp4-lm-head` |

Both 35B profiles retain FP32 SSM. The 27B full profile's FP16 SSM and
owner/lossless prefill options remain unsupported on 35B. The tables below
retain individual ablations; the README chart shows only default and full
alongside the official FP8 baseline.

The current default also fuses TP2 AllReduce, residual addition and RMSNorm.
Attention projections and the combined routed/shared MoE output return local
partial sums. Each following norm owns exactly one reduction, including the
model's final norm. The native small-batch and grouped MoE schedules remain
active. Fusion is restricted to native MXFP6 Qwen35 TP2/PP1 with no EP, DP,
context/sequence parallelism, microbatching, LoRA or speculative decoding.
Unsupported configurations retain their existing model path. FlashInfer's
`trtllm` IPC workspace supplies the fused operation directly, independently of
the generic `VLLM_ALLREDUCE_USE_FLASHINFER=0` setting. Unsupported tensor shapes
retain explicit AllReduce plus norm. Fusion changes reduction/rounding
order and does not promise bitwise equivalence to the separate operations.

The optional LM head uses FlashInfer's built-in B12X backend to search an NVFP4
copy of the head for 128 candidates per rank, then recomputes their logits
against the original BF16 weights. Only eligible greedy decode requests use
this path. The BF16 head is retained for full-logit and sampling fallbacks.
The candidate search is approximate; keeping BF16 refinement does not guarantee
that every BF16 winner enters the candidate set on arbitrary workloads.

Upgrade the installed runtime and launch with both optimizations:

```bash
vllm-mach-install --apply
CUDA_VISIBLE_DEVICES=0,1 vllm-mach-serve \
  --model /models/Qwen3.5-35B-A3B-MXFP6 --nvfp4-lm-head \
  --kv-cache-memory-bytes 8589934592
```

Omit `--nvfp4-lm-head` for the default BF16 head; add `--no-fused-ar-norm` to
disable communication fusion independently. The additive MoE fusion patch is
staged together with the existing runtime and IPC patches. Installer tests
cover clean installation, upgrades from either prior profile, repeat installs
and rejection of incompatible/partial source without writes.

For diagnostics on real hidden states (eager mode, with shadow BF16 head and
communication work; never use these timings as throughput results):

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/verify_qwen35_optimizations.py \
  --model /models/Qwen3.5-35B-A3B-MXFP6 --output qwen35-diagnostics.json
```

The new serving ablation keeps GDN enabled in every arm. Other settings and
the 3000/1000 fixed-token workload match the GDN experiment. Three fresh server
lifecycles run sequentially on the same GPU pair: AR + head, AR only, then GDN
only. Each point is a single run; the baseline is MXFP6 with GDN, not FP8.

| Configuration (output tokens/s) | c4 | c16 | c24 | c32 |
|---|---:|---:|---:|---:|
| GDN only | 785.8 | 1849.5 | 2333.2 | 2684.8 |
| Mach default (GDN + fused AR/Norm) | 1012.1 | 2224.1 | 2807.9 | 3169.2 |
| Mach full (default + NVFP4 head) | 1068.4 | 2288.5 | 2914.5 | 3241.9 |
| AR gain over GDN | +28.8% | +20.3% | +20.3% | +18.0% |
| Head gain over GDN + AR | +5.6% | +2.9% | +3.8% | +2.3% |
| Combined gain over GDN | +36.0% | +23.7% | +24.9% | +20.8% |

Validation passed 139 focused tests, including two-GPU H2048 graph replay with
changing inputs at M1/4/16/24/32/3001/4096; graph and eager fused outputs agree
bitwise. The combined server passed 35/35 request smoke checks. On 1640 real
decode positions from eight short prompts at batch sizes 1/4/16/32, the NVFP4
head retained every global BF16 top-20 token and matched every BF16 top-1.
Some refined logits differed from the full BF16 GEMM (maximum absolute difference
0.125), so top-1 agreement is not a claim of bitwise logit equivalence.
The shadow communication probe confirmed FlashInfer fusion, with zero residual
error and maximum relative normalized-output L2 error 0.003487 on its sampled
shapes. All 40 MoE layers deferred exactly the final TP sum and retained their
native expert schedules. These short diagnostics do not establish broad quality.

[Results, per-request data and numerical diagnostics](data/qwen35-ar-head-20260917.json)
retain the measurements; the built wheel was checked for the new adapter and patch.
