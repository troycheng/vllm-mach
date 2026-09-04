# Compatibility

The provider is pinned to vLLM 0.28.0 because it uses vLLM's quantization configuration, parameter loading, packed-module mapping, tensor-parallel linear, and CUDA Graph lifecycle APIs. A version bump needs a fresh import, registration, checkpoint metadata, model load, graph capture, and inference gate.

Supported checkpoint records declare `quant_format: exl3` and contain `trellis`, an input scale (`suh` or `su`), an output scale (`svh` or `sv`), and at most one codebook marker (`mcg` or `mul1`). Invalid metadata is rejected before weight execution.

The plugin registers the `exl3` quantization configuration and, when `mxfp6-sm120==0.2.1` is importable, prepends the native Dense MXFP6 kernel to vLLM 0.28's CUDA selector. It also adds the `mxfp8_e4m3` spelling to the Quark OCP-MX activation map; checkpoint metadata uses `fp8_e4m3` and vLLM normalizes it before lookup. These two changes are idempotent and do not modify vLLM's private online-quantization shorthand table. Native EXL3, B12X, and MXFP6 CUDA symbols are resolved only when their paths are used.

Importing and registering `vllm_mach.mxfp6` are lazy and do not import the optional module. Registration checks the installed distribution metadata for the pinned `mxfp6-sm120` version. The native selector imports the module and performs the CUDA/ABI availability probe only when vLLM selects a concrete device; a wheel built for a different CUDA runtime remains unavailable rather than being selected silently.

The runtime includes the tested EXL3 Dense path, an alpha MXFP6 Dense bridge, and one explicit selective EXL3/MXFP6 profile. The native bridge accepts only static E3M2 weights with dynamic MXFP8 E4M3 activations on SM120. The version-locked runtime profile supplies Stream-K workspace and graph-capture-stream warmup for both MXFP6 entry points.

`VLLM_MACH_EXL3_MXFP6_PROFILE=qwen38-27b` is accepted only with the existing Qwen3.8-27B Dense and TP2/PP1 checks. After tensor-parallel slicing, selected EXL3 shards are reconstructed and quantized into rank-local MXFP6 copies. Gate and up shards are joined before quantization. The MLP projections, GDN output projection, and attention output projection always use the copy. QKV and QKVZ use it only when the flattened row count is at least 128, so their EXL3 tensors are retained for decode. The conversion is transactional: a failed shard conversion leaves the original EXL3 state intact. It does not alter the checkpoint on disk.

For eligible Dense MLPs, the profile uses `mxfp6-sm120` to fuse SiLU/mul with packed MXFP8 activation quantization before the MXFP6 down projection. The hook requires BF16 two-dimensional input, bias-free Hybrid gate/up and down projections, and the standard TP-parallel reduction contract. Other modules fall back to their original vLLM forward method. Set `VLLM_MACH_EXL3_MXFP6_FUSED_MLP=0` to disable this fusion for diagnosis.

The profile does not cover `lm_head`, routed MoE, or GDN kernel replacement. Other models, tensor-parallel layouts, vLLM versions, and GPU architectures remain unverified. The full public `mxfp6_sm120` vLLM example remains the reference for its additional patches.
