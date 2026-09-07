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
