# Compatibility

The initial provider is pinned to vLLM 0.28.0 because it uses vLLM's quantization configuration, parameter loading, packed-module mapping, tensor-parallel linear, and CUDA Graph lifecycle APIs. A version bump needs a fresh import, registration, checkpoint metadata, model load, graph capture, and inference gate.

Supported checkpoint records declare `quant_format: exl3` and contain `trellis`, an input scale (`suh` or `su`), an output scale (`svh` or `sv`), and at most one codebook marker (`mcg` or `mul1`). Invalid metadata is rejected before weight execution.

The plugin registers only the `exl3` quantization configuration. It does not modify vLLM's private online-quantization shorthand table. Native EXL3 and B12X modules are imported on worker execution rather than during controller-side plugin discovery.

The runtime currently includes the Dense path. MoE, online Trellis conversion, and the selective EXL3/MXFP6 hybrid are not registered by version 0.1.0.
