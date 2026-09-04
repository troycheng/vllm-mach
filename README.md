# vLLM Mach

vLLM Mach is an independent performance engineering layer for vLLM. It focuses on end-to-end serving performance under real workload shapes while keeping model fidelity explicit and testable. Qwen is the current model focus; EXL3 and MXFP6 are execution backends, not the boundary of the project.

The `0.1.0a1` preview contains an out-of-tree EXL3 provider for `vllm==0.28.0` and two optional ways to use the external `mxfp6-sm120` runtime. The native Dense bridge connects vLLM's OCP-MX selector to that runtime. A separate, opt-in profile converts selected EXL3 projections to MXFP6 when the model is loaded. The CUDA implementation stays in `mxfp6-sm120`; it is not copied into this repository.

## Current support

The initial runtime has been validated with Qwen3.8-27B Dense, K5/K6 EXL3 weights, TP2, and SM120. This is the tested configuration, not a claim that every EXL3 checkpoint or GPU architecture works. Unsupported configurations should be treated as unverified until they have their own load, graph, correctness, and serving tests.

Native execution requires an ExLlamaV3 build exporting `exl3_gemm` and `exl3_mgemm`. B12X is optional and used only when its prefill route is enabled. Native BF16 I/O is also optional and requires the corresponding ExLlamaV3 extension entry points; otherwise the provider uses the standard EXL3 boundary.

The optional MXFP6 path handles static MXFP6 E3M2 weights with dynamic MXFP8 E4M3 activations through `mxfp6-sm120==0.2.1`. The selector is restricted to SM120 and leaves vLLM's emulation kernel in place for other supported MXFP6 layouts. The first bridge covers Dense layers, the `mxfp8_e4m3` Quark mapping (the checkpoint metadata spelling is `fp8_e4m3`), and Stream-K workspace warmup through a version-locked vLLM runtime profile. It does not include routed MoE or GDN kernel changes; the complete public reproducer in [mxfp6_sm120](https://github.com/Nekofish-L/mxfp6_sm120/tree/main/examples/vllm) remains the reference for those paths.

The selective EXL3/MXFP6 profile is disabled by default. Setting `VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b` reconstructs the selected rank-local EXL3 shards after tensor-parallel slicing and quantizes the copies during model loading. `mlp.gate_up_proj`, `mlp.down_proj`, `linear_attn.out_proj`, and `self_attn.o_proj` use MXFP6 for every row count. The Dense MLP path fuses SiLU/mul with the packed MXFP8 activation consumed by the MXFP6 down projection. `linear_attn.in_proj_qkvz` and `self_attn.qkv_proj` use MXFP6 only for prefill calls with at least 128 rows; their decode path remains EXL3. `lm_head` and unmatched projections also remain EXL3. The profile requires the validated Qwen3.8-27B Dense geometry, TP2/PP1, SM120, and `mxfp6-sm120==0.2.1`. It does not rewrite the checkpoint.

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
# Optional selective EXL3/MXFP6 profile:
export VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b
# Optional diagnostic fallback to the separate SiLU and MXFP8 operations:
# export VLLM_MACH_EXL3_MXFP6_FUSED_MLP=0
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

These environment variables describe the validated profile, not universal defaults. Leave `VLLM_MACH_EXL3_MXFP6_PROFILE` unset for the standard EXL3 path. Enable optional routes only when their native dependencies and target shapes have been checked.

## Development status

The alpha MXFP6 integration covers the native Dense selector and the explicit Qwen3.8-27B EXL3/MXFP6 profile described above. Both use the same graph-safe custom operator and workspace lifecycle. The selective profile and its fused Dense MLP activation path have passed their native operation-boundary checks; the profile has also passed TP2 eager and FULL_DECODE_ONLY graph serving gates. No throughput claim is attached to this preview. Routed MoE, fused GDN/collective changes, additional models, and additional GPU architectures still need their own correctness tests and same-image end-to-end comparisons before support is declared.

See [compatibility](docs/compatibility.md) for the version boundary and [validation](docs/validation.md) for the completed gates. vLLM Mach is not affiliated with or endorsed by the vLLM project.
