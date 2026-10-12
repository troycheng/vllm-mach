# Historical studies and releases

Current results live in [model performance](performance.md); current setup and operator guides are in the [documentation index](README.md). Superseded installation notes and chronological experiment logs are linked below at their last preserved commit. Original measurement JSON, CSV and benchmark scripts remain in this repository.

## EXL3 history

| Preserved document | Contents |
|---|---|
| [benchmarks.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/benchmarks.md) | EXL3-era throughput and fidelity |
| [champion-alignment.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/champion-alignment.md) | EXL3 checkpoint / Temporal alignment |
| [champion-port.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/champion-port.md) | Rank64 and owner-prefill port acceptance |
| [dependency-upgrade.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/dependency-upgrade.md) | vLLM0.29 / EXL31.4.9 migration |
| [exllamav3-1.5.0.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/exllamav3-1.5.0.md) | EXL31.5 migration |
| [public-install.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/public-install.md) | Version-pinned EXL3 installation |
| [validation.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/validation.md) | Early alpha-release acceptance |

| [quantization.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/quantization.md) | Legacy EXL3 and rank64 asset preparation |
| [ba-overlap.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/ba-overlap.md) | EXL3-era BA/QKV overlap acceptance |
| [experimental-decode.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/experimental-decode.md) | Legacy EXL3 decode experiments |
| [checkpoint-hybrid.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/checkpoint-hybrid.md) | Historical direct-checkpoint hybrid profile |

The EXL3-era NVFP4 run included patched FlashInfer AllReduce. Its values are historical local-runtime results, not the unpatched community NVFP4 baseline.

## Native MXFP6 history

| Preserved document | Contents |
|---|---|
| [native-fidelity-20260915.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/native-fidelity-20260915.md) | September15 native comparison |
| [native-fidelity.md](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/native-fidelity.md) | September16–17 native serving and fidelity snapshots |
| [MoE migration and early serving](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/qwen35-moe.md) | September17 graph, GDN and head migration; later measurements supersede the headline results |
| [2B optimization history](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/mxfp8-2b-champion.md) | October9–11 package, mixed-decode and PDL studies with their individual baselines |

## Releases

The [v0.1.1 release](releases/0.1.1.md) covers native MXFP6. MXFP8 and block-FP8 profiles described on `main` were added later. Earlier notes: [a10](releases/0.1.0a10.md), [a8](releases/0.1.0a8.md), [a7](releases/0.1.0a7.md), [a6](releases/0.1.0a6.md), [a5](releases/0.1.0a5.md), [a5.dev0](releases/0.1.0a5.dev0.md), [a4](releases/0.1.0a4.md), [a3](releases/0.1.0a3.md).
