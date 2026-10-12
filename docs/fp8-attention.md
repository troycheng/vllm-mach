# Mixed FA2 attention

Mach provides a narrow, optional source backport of [LiRunGuo's vLLM PR #58013](https://github.com/vllm-project/vllm/pull/58013), original head `e17555d2673e632cbe85365a53490a138e4b2fae`. The algorithm and original author attribution are retained. The backport targets the pinned vLLM 0.29 `flash_attn.py` with stock SHA-256 `5d9c676d8d03ec4e583183f9f42fac33c223768fdc3d6088dff97b5ac0cd4c3a`; a different file requires a reviewed source profile.

After applying the matching FP8 source profile, set `VLLM_MACH_FP8_FA2=1` before workers construct attention metadata builders. With the flag absent, the original attention call executes. This feature is independent of block-FP8 linear and SiLU quantization and does not apply to the MXFP8 FlashInfer profile.

The builder reads the runner's existing CPU query offsets and records the verified one-token decode prefix in `FlashAttentionMetadata.num_split_decodes`. It builds `prefill_query_start_loc` with offsets rebased to zero. Normal `FlashAttentionImpl.forward` then executes the original FA2 function twice, first for decode with maximum query length one, then for prefill. It preserves the original paged K/V storage, output buffer and cache writer. Request-indexed KV lengths, block tables and descales are sliced with the same request boundary. There is no ContextVar, global FlashAttention function replacement or forward wrapper.

The supported contract matches the historical adapter: TP1/PP1/DCP1, FA2 decoder, BF16 Q/output/K/V, 16 query heads, 4 KV heads, D256 and 528-token KV pages. A mixed batch has a leading one-token decode prefix followed by one or two longer prefill requests. It has no speculative decoding, DBO, cascade, scheduler metadata, multimodal/R-SWA masks, sliding window, alibi, sinks, logits softcap or output quantization. Both calls use `num_splits=1`; metadata values other than zero or one fall back. Unsupported geometry, pure decode/prefill and CUDA graph capture preserve the original call. The patch changes neither request ordering nor graph selection.

`tests/test_fp8_attention_cpu.py` checks the selection and operand contract with real Torch CPU tensors and a recorder in place of the external FA kernel. Set `VLLM_MACH_FP8_FA2_SOURCE` to the installed patched source when running the suite. It verifies two normal-forward calls, offset rebasing, request slices, cache aliasing, output mutation and untouched padding.

## Validation

The c32/c64 attention-input checks are byte-exact, including padding, stock fallback, capture fallback and changing-input graph replay. The full profile's [M4/M32/M64 fixed-gold checks](fp8-profile.md#precision-and-qualification) also match stock byte-for-byte.

Mach uses `num_splits=1` for both mixed-batch calls. The original upstream head used zero and produced small decode differences on the tested FA2 binary; forcing one restored byte equality. The [diagnostic record](data/fp8-fa2-source-check-20261006.json) preserves both results. Reproduce with public synthetic inputs:

```bash
python tools/verify_fa2_mixed_head.py --seed 20261006 --output ./fa2-synthetic-result.json
```
