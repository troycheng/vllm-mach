# Owner prefill

In the Qwen3.8-27B Dense MXFP6 TP2 profile, the owner-prefill path assigns a
contiguous range of hidden rows to each rank between norms. It admits
BF16 prefill-sized batches from 512 to 4096 physical rows at hidden size 5120
on SM120, outside CUDA Graph capture. With [lossless prefill](../lossless_prefill/README.md)
enabled, physical M1024 and M1052 stay on that path.

The path gathers MXFP8 activations and scales for the projections and uses
the loaded BF16 parameters for GDN BA. It pads ragged communication packets
and crops them before attention. One final BF16 gather feeds the LM head.

Starting at M2048, the owner path runs the MLP locally on the 32 even-numbered
layers selected by the launcher. Each rank replicates the peer's packed MXFP6 gate/up
and down weights, preserving the two original branches and their BF16
rounding. These replicas use 3.113 GiB per rank, in addition to the owner
collective workspace and BA replicas. Smaller admitted batches keep the
original TP MLP branches.

The current [MXFP6 setup](../../docs/installation.md) builds
`vllm-mach-owner-prefill` and installs the matching vLLM overlay. The wheel
contains four CUDA 13.0 extensions and is checked for the `owner-prefill-v1`
ABI. To rebuild just this wheel in the selected runtime, use
`python deploy/install.py --native-only --native-part owner_prefill` from the
repository root with `--cuda-home` pointing to that toolkit.

`vllm-mach-serve` enables owner prefill for Qwen3.8-27B Dense MXFP6 TP2.
Use `--no-owner-prefill` to disable it. `--verify-prefill` checks the first
eligible real prefill forward against the original operations. Run throughput
measurements with verification off.

The [historical toolkit benchmark](BENCH_TOOLKIT.md) compares CUDA 13.0 and
13.2 on synthetic primitive inputs.
