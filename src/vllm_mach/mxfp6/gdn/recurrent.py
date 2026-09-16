# SPDX-License-Identifier: Apache-2.0
"""TP2 packed recurrence schedule, retaining vLLM's arithmetic and state ABI."""


def packed_decode(*, mixed_qkv, a, b, A_log, dt_bias, scale, initial_state,
                  out, ssm_state_indices, use_qk_l2norm_in_kernel):
    # The caller has validated native TP2, M16/24/32, contiguous inner state
    # dimensions, and one unique state slot per live decode request.
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode_kernel as kernel,
    )
    kernel[(16, mixed_qkv.shape[0] * 24)](
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        o=out, h0=initial_state, ht=initial_state,
        ssm_state_indices=ssm_state_indices, scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        H=8, HV=24, K=128, V=128, BK=128, BV=8,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        SPLIT_BATCH_HEAD_GRID=False, num_warps=1, num_stages=3,
    )
    return out, initial_state
