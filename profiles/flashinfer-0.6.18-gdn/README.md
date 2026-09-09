# SM120 fused GDN profile

Version-locked backport for vLLM 0.28.0 and FlashInfer 0.6.18. It contains the fused GDN implementation and integration used by the Temporal checkpoint reference. The operation fuses the b/a projection, causal convolution, gates and recurrent state update. Existing paths handle unsupported geometry, prefill and speculative decoding.

Source work: [FlashInfer #4698](https://github.com/flashinfer-ai/flashinfer/pull/4698) and [vLLM #53645](https://github.com/vllm-project/vllm/pull/53645). This directory preserves the tested integration snapshot, not the current heads of those PRs. Derived files retain their Apache-2.0 notices. The framework gate defaults to off in this backport.

Apply in a dedicated environment before starting vLLM:

```bash
python profiles/flashinfer-0.6.18-gdn/install.py
python profiles/flashinfer-0.6.18-gdn/install.py --apply
export VLLM_ENABLE_QWEN_GDN_FUSED_DECODE=1
```

The installer checks all target hashes before writing, rejects unknown installed sources, and accepts an already applied profile. Do not run it against a running service. The overlay is separate from the Python wheel. A complete CUDA development toolkit is required for first-use JIT compilation; compiled artifacts go to FlashInfer's cache.

The SM120 TP2 reference uses `cuda_sm120_persistent`. Acceptance must observe that implementation in both worker logs and reject backend failures or unexpected fallback. CUDA Graph capture must complete before measuring. This profile is not a general GPU or model support claim.

The combined checkpoint/Temporal configuration passed the existing40-task regression suite on TP2 SM120, with both workers using `cuda_sm120_persistent`. See the [alignment record](../../docs/champion-alignment.md). This validates the documented integration; it does not establish general precision equivalence or isolate the GDN speedup.

## Optional M32 BA overlap

The `0.1.0a7` wheel and updated overlay add `VLLM_MACH_BA_OVERLAP=1`, default off. Install both, reapply this profile, and restart workers. The installer accepts the original source or the known previous Mach overlay; it still rejects unrelated edits.

For the Qwen3.8-27B checkpoint profile, the M32 fallback can run the original BA projection and split/contiguous work on an auxiliary stream while QKV and convolution run on the main stream. The main stream joins before packed recurrent decode. Weights and arithmetic kernels are unchanged. The guard requires TP2, contiguous BF16 input of physical shape 32×5120, active merged checkpoint MXFP6 at M32, non-speculative decode without prefill, and the unsupported-shape fallback of the FlashInfer fused entry. Other paths keep serial execution. A captured M32 Graph can also serve padded tail batches; physical M32 does not imply 32 live requests.

Version 0.1.0a8 has completed Mach GPU/Graph/service acceptance: serial-reference QKV/BA assertions passed during real replay at 48 layers per rank, followed by a separate verifier-off service test at 1024/256 and 3000/1000. The optional `qwen38-checkpoint-long.env` now enables BA overlap; base defaults remain off. The source experiment's +1.4455% c32 result remains source evidence, not an isolated Mach speedup. See [the port record](../../docs/ba-overlap.md).

For diagnostics only, `VLLM_MACH_BA_OVERLAP_VERIFY=1` adds Graph-replayed bitwise assertions against the serial QKV/BA producers. Install the matching v0.1.0a8 wheel and overlay together. Keep verification off for normal serving.
