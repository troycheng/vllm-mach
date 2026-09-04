# vLLM Mach

vLLM Mach is an independent performance engineering layer for vLLM. It focuses on end-to-end serving performance under real workload shapes while keeping model fidelity explicit and testable. Qwen is the current model focus; EXL3 and MXFP6 are execution backends, not the boundary of the project.

The first release contains an out-of-tree EXL3 provider for `vllm==0.28.0`. It recognizes hydrated EXL3 checkpoints, validates their metadata, loads tensor-parallel shards, groups compatible QKV/QKVZ projections, primes decode kernels before CUDA Graph capture, and supports optional B12X prefill and native BF16 I/O paths. The installed vLLM package does not need to be patched for provider registration.

## Current support

The initial runtime has been validated with Qwen3.8-27B Dense, K5/K6 EXL3 weights, TP2, and SM120. This is the tested configuration, not a claim that every EXL3 checkpoint or GPU architecture works. Unsupported configurations should be treated as unverified until they have their own load, graph, correctness, and serving tests.

Native execution requires an ExLlamaV3 build exporting `exl3_gemm` and `exl3_mgemm`. B12X is optional and used only when its prefill route is enabled. Native BF16 I/O is also optional and requires the corresponding ExLlamaV3 extension entry points; otherwise the provider uses the standard EXL3 boundary.

## Install

```bash
python -m pip install vllm==0.28.0
python -m pip install /path/to/exllamav3.whl
python -m pip install /path/to/b12x.whl  # optional
python -m pip install .
```

Build a wheel with:

```bash
python -m pip install build
python -m build --wheel
```

## Run

Select the plugin explicitly when other vLLM plugins are installed:

```bash
export VLLM_PLUGINS=mach_exl3
export EXL3_QKV_MGEMM=1
export VLLM_EXL3_GRAPH_DECODE=1
export VLLM_EXL3_B12X_MIN_M=128
export VLLM_EXL3_B12X_N_RANGE=5120-36864
export VLLM_EXL3_B12X_ANY_BITS=1
export VLLM_EXL3_PREFILL_FUSED_RECONSTRUCT_MIN_M=128

vllm serve /path/to/hydrated-exl3-model \
  --quantization exl3 \
  --tensor-parallel-size 2 \
  --compilation-config \
  '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
```

These environment variables describe the validated profile, not universal defaults. Start from the standard EXL3 path and enable optional routes only when their native dependencies and target shapes have been checked.

## Development status

Work in progress covers a selective EXL3/MXFP6 runtime, fused SiLU-to-MXFP8 activation output, and a FlashInfer AllReduce/RMSNorm/MXFP8 handoff. Those paths are not in the first package because they depend on separate kernel and framework changes that still need a reproducible public build and a monitored release audit. They will be added as independent integrations instead of being hidden inside the EXL3 provider.

See [compatibility](docs/compatibility.md) for the version boundary and [validation](docs/validation.md) for the completed gates. vLLM Mach is not affiliated with or endorsed by the vLLM project.
