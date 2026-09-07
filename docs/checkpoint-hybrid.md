# Direct-checkpoint hybrid profile

This unreleased profile ports the direct MXFP6 weight-loading and merged QKV routes from the experimental Qwen3.8-27B service. It is opt-in; the existing `qwen38-27b` profile is unchanged. The port has passed loading, native Graph, and TP2 functional checks described in [validation.md](validation.md). These checks do not establish full-model fidelity or performance.

## Configuration

Use a local MXFP6 checkpoint from the same base model as the EXL3 checkpoint:

```bash
export VLLM_PLUGINS=mach
export VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b-checkpoint
export VLLM_MACH_MXFP6_CHECKPOINT=/models/Qwen3.8-27B-MXFP6
export VLLM_MACH_EXL3_MXFP6_FUSED_AR_NORM_MXFP8=0
```

These are additional settings for the existing EXL3 serve command. The profile retains the vLLM 0.28.0, Qwen3.8-27B Dense, SM120, TP2/PP1 boundary and requires a matching `mxfp6-sm120==0.2.1` build. Source metadata, packed shapes, and TP shard mapping are checked. Matching geometry alone does not establish that two checkpoints contain weights from the same model revision: the operator must verify the paired model origins. Do not combine unrelated fine-tunes.

Keep the optional fused collective disabled for this upgrade candidate. The successful TP2 check used its fallback; the FlashInfer 0.6.18 fused-path experiment failed the task regression. This does not disable the MXFP6 weight routes or fused MLP activation path.

The experimental MXFP6 checkpoint currently has a local content manifest but no verified public repository/revision. There is therefore no public download link for an exact reproduction of that recipe. This profile accepts an operator-provided local checkpoint; the experimental throughput and quality results must not be assigned to an arbitrary compatible checkpoint.

| Projection | Weight source | Execution |
|---|---|---|
| MLP gate/up and down; attention/GDN output | Original MXFP6 packed weights and scales | MXFP6 for all nonempty row counts |
| QKV and QKVZ | Original MXFP6 packed weights and scales, merged in output order | One MXFP6 call at physical M=32 or M>=128 |
| QKV and QKVZ at other row counts | EXL3 checkpoint | Existing EXL3 path |
| LM head and unmatched projections | EXL3/model checkpoint | Unchanged |

The loader does not reconstruct EXL3 weights and requantize them. It slices the original MXFP6 codes and logical scales, joins compatible output shards, and packs the scales for the native runtime. QKV retains its EXL3 tensors for the other row counts. Workspace planning uses Mach's existing shared warmup; an additional planner is not installed.

M denotes the physical flattened input row count, including CUDA Graph padding, not the number of active requests. This MXFP6 M32 route is distinct from `EXL3_BF16_IO_TILE_M32`, which selects an EXL3 kernel. The checkpoint profile does not require that optional EXL3 M32 extension for its routed M32 call.

## Evidence and limits

The source service's direct-checkpoint recipe passed its frozen performance comparison against the strong MXFP6 deployment on dual RTX 5090, TP2, FULL decode graphs, 1024 input / 256 output tokens. This is evidence for selecting the implementation, not a measurement of the Mach port or a 3K/1K performance claim.

The later combined recipe also used a separate Temporal M24 EXL3 K6 kernel. That kernel is not included in this profile. Its measured c24 improvement and c4 regression must not be attributed to this port.

The 2026-09-07 experimental comparison used 256 fixed samples and 10,479 target tokens. For the combined recipe, the sample-weighted target-logprob MAE difference from MXFP6 was +0.003467 (95% CI [-0.000826, +0.007667]); the secondary NLL difference was +0.009464 nats/token (95% CI [+0.002434, +0.017180]). Intervals were not multiplicity-adjusted. These are prefill teacher-forced fidelity results, not business accuracy, and do not establish losslessness. The recipe also pinned FLA arithmetic configurations; Mach does not silently reproduce private rank-specific autotuner patches. Its own runtime and quality acceptance remain required.

## Validation status

- Both ranks' 512 projections match the frozen reference loader, including packed weights, logical scales, and aggregate content digests.
- Real-weight changing-input eager/Graph checks passed for EXL3 QKV/QKVZ and all six MXFP6 projection families.
- The built Mach wheel passed the TP2 40-task regression. Runtime details and fused-collective coverage are recorded in [validation.md](validation.md).
- A full-model fidelity/performance comparison and a verified public paired-checkpoint identity remain outstanding. Record the actual FLA configuration before attributing experimental quality results to another deployment.
