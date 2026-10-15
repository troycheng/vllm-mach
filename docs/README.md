# Documentation

Start with [performance by model size](performance.md), then use the guide for the profile you intend to run. Each profile guide owns its build, model inputs, runtime contract and benchmark command.

## Model setup

| Model / path | Start here | Model inputs |
|---|---|---|
| Qwen3.5-2B MXFP8 | [Champion](mxfp8-2b-champion.md) | [Public BF16 reconstruction](mxfp8-2b-model.md) |
| Qwen3.5-4B MXFP8 | [Champion](mxfp8-champion.md) | [BF16 + released L0 asset](mxfp8-model.md) |
| Qwen3.5-4B block-FP8 | [Profile](fp8-profile.md) | Existing compatible compressed-tensors checkpoint |
| Qwen3.8-27B MXFP6 | [Installation](installation.md) | [Public MXFP6 checkpoint](https://huggingface.co/nekofish/Qwen3.8-27B-MXFP6) |
| Qwen3.5-35B-A3B MXFP6 | [MoE guide](qwen35-moe.md) | [Native conversion guide](https://github.com/Nekofish-L/mxfp6_sm120/blob/main/docs/quantize-checkpoints.md) |

## Performance and precision

- [Complete performance report](performance.md): per-model FP8/NVFP4 throughput charts, all concurrency points, latency and BF16-reference fidelity.
- [Best-path and sampling recheck (2026-10-15)](best-path-recheck-20261015.md): 2B/4B pre-sampling/current comparisons, dispatch evidence and top-k/top-p results.
- [MXFP8 benchmark](mxfp8-benchmark.md): the single 2B/4B per-concurrency report, SSM/KV configurations, block-FP8 comparisons and public measurement protocol.
- [4B MXFP8 validation](mxfp8-champion.md#validation) and [block-FP8 validation](fp8-profile.md#precision-and-qualification): source identities and numerical checks.
- [Dense optimization breakdown](dense-gains-aligned-baseline-20260920.md), [MoE breakdown](moe-gains-fp8-baseline-20260920.md) and [MXFP6 fidelity](profile-fidelity-20260917.md).
- [Normalized current data](data/model-performance.json); original receipts remain alongside it in `data/`.

## Runtime and operators

| Area | Guides |
|---|---|
| Native integration and compatibility | [MXFP6](native-mxfp6.md), [MXFP8 installation](mxfp8-installation.md) |
| GDN and scheduling | [MXFP6 GDN](gdn-decode.md), [MXFP8 GDN](mxfp8-gdn.md), [FP16 SSM](fp16-ssm.md) |
| Fusion and prefill | [Producers](dense-producer-fusion.md), [AR/Norm/MXFP8](ar-norm-mxfp8.md), [lossless prefill](lossless-prefill.md), [observed-shape prefill](long-prefill.md) |
| MoE | [Projection dispatch](moe-projection-20260923.md), [model integration](qwen35-moe.md) |
| Block-FP8 | [Linear](fp8-linear.md), [activation](fp8-activation.md), [attention](fp8-attention.md) |

## Contributing

See [contribution requirements](../CONTRIBUTING.md) for source, correctness and performance evidence.
