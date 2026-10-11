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

- [Complete performance report](performance.md): charts, all concurrency points, throughput, latency and fidelity.
- [MXFP8 benchmark protocol](mxfp8-benchmark.md): public fixed requests and metric definitions.
- [4B MXFP8 qualification](mxfp8-qualification-20261006.md) and [block-FP8 qualification](fp8-qualification.md): source identities and numerical checks.
- [Dense optimization breakdown](dense-gains-aligned-baseline-20260920.md), [MoE breakdown](moe-gains-fp8-baseline-20260920.md) and [MXFP6 fidelity](profile-fidelity-20260917.md).
- [Normalized current data](data/model-performance.json); original receipts remain alongside it in `data/`.

## Runtime and operators

| Area | Guides |
|---|---|
| Native integration and compatibility | [MXFP6](native-mxfp6.md), [MXFP8 installation](mxfp8-installation.md), [compatibility](compatibility.md) |
| GDN and scheduling | [MXFP6 GDN](gdn-decode.md), [MXFP8 GDN](mxfp8-gdn.md), [FP16 SSM](fp16-ssm.md) |
| Fusion and prefill | [Producers](dense-producer-fusion.md), [AR/Norm/MXFP8](ar-norm-mxfp8.md), [lossless prefill](lossless-prefill.md), [observed-shape prefill](long-prefill.md) |
| MoE | [Projection dispatch](moe-projection-20260923.md), [model integration](qwen35-moe.md) |
| Block-FP8 | [Linear](fp8-linear.md), [activation](fp8-activation.md), [attention](fp8-attention.md), [FA2 source check](fp8-fa2-source-check.md) |
| Diagnostics | [B12X audit](b12x-audit.md); earlier EXL3 paths are in [history](archive.md) |

## History and contribution

[Historical studies and release notes](archive.md) retain the earlier EXL3 and native results without mixing them into current setup. See [contribution requirements](../CONTRIBUTING.md) for source, correctness and performance evidence.
