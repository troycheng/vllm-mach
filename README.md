# vLLM Mach

vLLM Mach is an independent performance engineering layer for vLLM. It focuses on end-to-end serving performance under real workload shapes while keeping model fidelity explicit and testable. Qwen is the current model focus; EXL3 and MXFP6 are execution backends, not the boundary of the project.

The `0.1.0a1` preview contains an out-of-tree EXL3 provider for `vllm==0.28.0` and an optional Dense MXFP6 bridge. The EXL3 provider recognizes hydrated EXL3 checkpoints, validates their metadata, loads tensor-parallel shards, groups compatible QKV/QKVZ projections, primes decode kernels before CUDA Graph capture, and supports optional B12X prefill and native BF16 I/O paths. The MXFP6 bridge connects vLLM's OCP-MX Dense selector to the external `mxfp6-sm120` wheel; it does not copy the CUDA implementation into this repository.

## Current support

The initial runtime has been validated with Qwen3.8-27B Dense, K5/K6 EXL3 weights, TP2, and SM120. This is the tested configuration, not a claim that every EXL3 checkpoint or GPU architecture works. Unsupported configurations should be treated as unverified until they have their own load, graph, correctness, and serving tests.

Native execution requires an ExLlamaV3 build exporting `exl3_gemm` and `exl3_mgemm`. B12X is optional and used only when its prefill route is enabled. Native BF16 I/O is also optional and requires the corresponding ExLlamaV3 extension entry points; otherwise the provider uses the standard EXL3 boundary.

The optional MXFP6 path handles static MXFP6 E3M2 weights with dynamic MXFP8 E4M3 activations through `mxfp6-sm120==0.2.1`. The selector is restricted to SM120 and leaves vLLM's emulation kernel in place for other supported MXFP6 layouts. The first bridge covers Dense layers, the `mxfp8_e4m3` Quark mapping (the checkpoint metadata spelling is `fp8_e4m3`), and Stream-K workspace warmup through a version-locked vLLM runtime profile. It does not include routed MoE or GDN kernel changes; the complete public reproducer in [mxfp6_sm120](https://github.com/Nekofish-L/mxfp6_sm120/tree/main/examples/vllm) remains the reference for those paths.

## Install

```bash
python -m pip install vllm==0.28.0
python -m pip install /path/to/exllamav3.whl
python -m pip install /path/to/b12x.whl  # optional
python -m pip install .
```

For the MXFP6 bridge, build `mxfp6-sm120==0.2.1` against the same PyTorch and
CUDA environment as vLLM, then install that wheel before vLLM Mach. The bridge
checks its version and ABI at runtime and fails closed when the native library
is not usable.

Stream-K CUDA Graph execution also requires the version-locked
[`vLLM 0.28 runtime profile`](profiles/vllm-0.28.0/README.md). The profile adds
workspace warmup calls at the two lifecycle points that vLLM 0.28 does not
provide as plugin hooks.

Build a wheel with:

```bash
python -m pip install build
python -m build --wheel
```

## Run

Select the plugin explicitly when other vLLM plugins are installed. The
`mach` entry point registers both the EXL3 provider and the
optional MXFP6 bridge:

```bash
export VLLM_PLUGINS=mach
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

The alpha MXFP6 integration is deliberately limited to the Dense selector boundary and the workspace lifecycle needed by that kernel. It has passed a TP2 model-load, graph-capture, and inference gate, but no throughput claim is attached to this preview. Fused SiLU-to-MXFP8 output, routed MoE, GDN changes, and the latest hybrid EXL3/MXFP6 serving work remain follow-ups that need isolated correctness tests and same-image end-to-end comparisons before they are enabled by default.

See [compatibility](docs/compatibility.md) for the version boundary and [validation](docs/validation.md) for the completed gates. vLLM Mach is not affiliated with or endorsed by the vLLM project.
