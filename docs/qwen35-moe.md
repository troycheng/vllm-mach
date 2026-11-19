# Qwen3.5-35B-A3B MXFP6

[September 17 default/full numerical fidelity](profile-fidelity-20260917.md) covers matched BF16 references at physical M4 and M32, with FP8/NVFP4 comparisons and repeatability checks. These precision diagnostics are separate from the serving throughput measurements below.

Mach supports the Quark packed E3M2 checkpoint with dynamic per-32 E4M3 activations on two SM120 GPUs. The default launcher uses TP2, BF16 activations, FP32 recurrent state and vLLM's default compilation/CUDA Graph configuration. On vLLM 0.33.0 this resolves to `VLLM_COMPILE` with `FULL_AND_PIECEWISE`: full graphs for decode and piecewise graphs for eligible prefill batches. The launcher no longer forces `NONE` / `FULL_DECODE_ONLY` for MoE. The default MoE batched-token budget is 2048. Explicit `--cudagraph-capture-sizes`, `--max-num-seqs` and scheduler overrides are forwarded to vLLM. Dense models retain their existing decode-only configuration.

The package also installs [37 measured projection configurations](moe-projection-20260923.md)
for this model's TP2 BF16 path on SM120. They are loaded before workspace
planning and graph capture, with no external tuning file or checkpoint change.
The native config ABI is checked at startup. Set
`VLLM_MACH_MOE_PROJECTION_TUNING=0` before starting a new worker to disable them.

The [runtime versions](installation.md#runtime-versions) list the native library and dependencies. Build the native wheel against the serving environment's PyTorch, following [installation](installation.md). `deploy/build-mxfp6.py` pins the native source and CUTLASS revisions.

For a local checkout and checkpoint, install the pinned native wheel and Mach package as in [installation](installation.md). The first command uses the launcher defaults. The second reproduces the measured FP16 FULL service configuration; select two free SM120 GPUs and set `MODEL` to the same checkpoint path for both commands.

```bash
python -m pip install --no-deps --no-build-isolation /path/to/mxfp6_sm120
python -m pip install --no-deps .
vllm-mach-install --apply
MODEL=/path/to/Qwen3.5-35B-A3B-MXFP6
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 vllm-mach-serve \
  --model "$MODEL" --host 127.0.0.1 --port 8000
```

For the original FULL benchmark service, use this explicit graph and capacity configuration:

```bash
MODEL=/path/to/Qwen3.5-35B-A3B-MXFP6
GRAPH_CONFIG='{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,10,12,14,16,20,24,28,32,36,40,48,56,64,72,80,96,112,128,160,192,224,256,320,384,448,512,640,768,784,896,1024,1280,1536,1792,2048]}'
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
VLLM_HYBRID_NVFP4_LM_HEAD_CUDAGRAPH=0 vllm-mach-serve \
  --model "$MODEL" --tokenizer "$MODEL" --tensor-parallel-size 2 \
  --hybrid-head --fp16-ssm --kv-cache-memory-bytes 8589934592 \
  --kv-cache-dtype bfloat16 --max-num-seqs 64 --max-model-len 16384 \
  --max-num-batched-tokens 2048 --compilation-config "$GRAPH_CONFIG" \
  --served-model-name migration-perf --host 127.0.0.1 --port 8000
```

`VLLM_HYBRID_NVFP4_LM_HEAD_CUDAGRAPH=0` leaves the private head graph off while retaining the hybrid head. The model runner uses `FULL_AND_PIECEWISE`. The FP16 FULL worker binds the native head scale and four bounded FLA prefill graph banks after normal model graph capture. The prefill graphs replay the original six kernels for qualified layouts and metadata; other inputs use the original path. They do not change the recurrent state, KV capacity or public output ownership.

Existing dense-profile installations can be upgraded with `vllm-mach-install --apply`; the MoE overlay is checked before writes.

The adapter retains packed expert weights, packs checkpoint scales to the
native layout and uses the package's routed grouped GEMMs for prefill. Eligible
Qwen35 TP2 layers combine the shared expert with the routed weights for the
1–4 token fused router/expert schedule and use the indirect grouped schedule
for 5–96 tokens. Larger batches use the generic routed path by default. With AR/Norm
fusion disabled, the standard vLLM runner performs the final TP reduction;
with fusion enabled, the following norm owns it.

## Compact MoE at high token counts

Pass `--moe-high-m-compact` to opt in on native Qwen3.5-35B-A3B MXFP6 TP2.
Eligible layers use compact expert scheduling for physical M97–160, where M is
the number of rows submitted to the MoE layer, including graph padding. The
schedule preserves the vLLM router's FP32 summation order and rounds the routed
output to BF16 before adding the shared expert. It uses a separate compilation
cache. Startup requires a native wheel with `qwen35_compact_gemm_out`.

On two RTX 5090s, the installed native and Mach wheels served the same Quark E3M2 checkpoint with vLLM 0.33.0, TP2, FP32 SSM, the full BF16 head, and BF16 KV. Each c128 window used 128 warmup requests, then 256 timed requests with 3,000 input and 1,000 output tokens. Both settings used 11 GiB of KV per GPU and replayed full M128 decode graphs.

| c128 run order | Default, output tok/s | Compact, output tok/s | Change |
|---|---:|---:|---:|
| Default → compact | 4,811.15 | 4,888.94 | +1.62% |
| Compact → default | 4,817.09 | 4,916.22 | +2.06% |

The geometric mean throughput change was **+1.84%**. In the earlier c64 trial, the compact configuration averaged −0.87% across two pairs, with zero graph steps in the target range. Use the default schedule for c64.

At physical M128 and M160, all 10,479 fixed-token log probabilities and top-1 predictions per shape matched the released route. Mean absolute logprob error relative to BF16 was 0.07264 at M128 and 0.07144 at M160. All 128 M128 free-generation token sequences matched.

To use the measured precision and serving configuration, keep the graph sizes
from `GRAPH_CONFIG` above and select the compact schedule explicitly.

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 vllm-mach-serve \
  --model "$MODEL" --tokenizer "$MODEL" --tensor-parallel-size 2 \
  --moe-high-m-compact --no-hybrid-head --mamba-ssm-cache-dtype float32 \
  --kv-cache-dtype bfloat16 --kv-cache-memory-bytes 11811160064 \
  --max-model-len 16384 --max-num-seqs 256 --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --compilation-config "$GRAPH_CONFIG" \
  --cudagraph-metrics --host 127.0.0.1 --port 8000
```

The compact path reports `Executing Qwen3.5-35B TP2 MXFP6 compact high-M schedule`
on each worker that executes it.

## Other MoE routes

The launcher recognizes `qwen3_5_moe` and `qwen3_5_moe_text` in the local model
configuration. Native TP2 AllReduce/residual/RMSNorm fusion supports both
`NONE` and the default `VLLM_COMPILE` execution. The compiled path uses a
functional custom op with FakeTensor support: workspace checks run on real
tensors, and the normalized output and updated residual use separate buffers.
This preserves the inputs and avoids alias-driven copies during compilation.
Compiled fusion includes the TP rank in its cache namespace so rank-specific
embedding bounds and device allocations cannot be loaded by another rank.
The routed and shared expert output stays local until the following fused norm performs the TP sum;
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

The default retains FP32 recurrent state. FULL selects FP16 recurrent state with
`--fp16-ssm` and keeps the hybrid head enabled; owner/lossless prefill switches remain Dense-only. This profile targets text-only,
non-speculative TP2/PP1 inference;
dense-model benchmark numbers do not describe its performance.

Startup should report `Using native mxfp6-sm120 grouped MoE backend` and,
when weight geometry matches, `Enabled Qwen3.5-35B TP2 MXFP6 fused small-batch
router/shared-expert schedule`.

## Performance and numerical fidelity

The [vLLM 0.34 results](performance.md#vllm-034-throughput) cover the standard FP16-state profile at c32 and the compact FP32-state profile at c128. The [vLLM 0.33 curves](performance.md#qwen35-35b-a3b) cover c4 through c32. The September 28 vLLM 0.29 measurements are retained in the [historical MoE report](performance-vllm029.md#qwen35-35b-a3b).

The [configuration breakdown](moe-gains-fp8-baseline-20260920.md) records graph, sampler, communication, state and head increments. [Projection dispatch](moe-projection-20260923.md) covers physical rows 32–160. [Numerical fidelity](profile-fidelity-20260917.md) records default/full against BF16, FP8 and NVFP4 at controlled physical rows.
