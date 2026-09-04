# Compatibility

The provider is pinned to vLLM 0.28.0 because it uses vLLM's quantization configuration, parameter loading, packed-module mapping, tensor-parallel linear, and CUDA Graph lifecycle APIs. A version bump needs a fresh import, registration, checkpoint metadata, model load, graph capture, and inference gate.

Supported checkpoint records declare `quant_format: exl3` and contain `trellis`, an input scale (`suh` or `su`), an output scale (`svh` or `sv`), and at most one codebook marker (`mcg` or `mul1`). Invalid metadata is rejected before weight execution.

The plugin registers the `exl3` quantization configuration and, when `mxfp6-sm120==0.2.1` is importable, prepends the native Dense MXFP6 kernel to vLLM 0.28's CUDA selector. It also adds the `mxfp8_e4m3` spelling to the Quark OCP-MX activation map; checkpoint metadata uses `fp8_e4m3` and vLLM normalizes it before lookup. These two changes are idempotent and do not modify vLLM's private online-quantization shorthand table. Native EXL3, B12X, and MXFP6 CUDA symbols are resolved only when their paths are used.

Importing `vllm_mach.mxfp6` is lazy and does not import the optional module. The native selector performs the CUDA/ABI availability probe before it is chosen; a wheel built for a different CUDA runtime remains unavailable rather than being selected silently.

The runtime includes the tested EXL3 Dense path and an alpha MXFP6 Dense bridge. The MXFP6 bridge accepts only static E3M2 weights with dynamic MXFP8 E4M3 activations on SM120. It does not include checkpoint conversion, routed MoE, GDN, or graph-capture warmup integration. The full public `mxfp6_sm120` vLLM example remains the reference for those additional patches.
