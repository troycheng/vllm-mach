# SPDX-License-Identifier: Apache-2.0
"""Native MoE FP16 mixed GDN in the qualified legacy local8 order."""
from vllm.triton_utils import tl, triton
from vllm_mach.mxfp8.gdn.legacy_reduction import _old_dot_128_8local

@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_q: tl.constexpr,
    stride_k: tl.constexpr,
    stride_v: tl.constexpr,
    stride_a: tl.constexpr,
    stride_b: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    tl.static_assert(H == 8 and HV == 16 and K == 128 and V == 128 and BK == 128 and BV == 32)
    tl.static_assert(h0.dtype.element_ty == tl.float16)
    tl.static_assert(USE_INITIAL_STATE and INPLACE_FINAL_STATE and USE_QK_L2NORM_IN_KERNEL)
    tl.static_assert(B == 1 and IS_CONTINUOUS_BATCHING and IS_VARLEN and not IS_SPEC_DECODING and not IS_KDA)
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            # Load state index and check for invalid entries
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            # Skip if state index is invalid (NULL_BLOCK_ID=0)
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

        # compute beta_output = sigmoid(b)
        b_beta = tl.sigmoid(b_b.to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BV, BK]
        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        # [BV]
        b_v -= _old_dot_128_8local(b_h, b_k[None, :], BV)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = _old_dot_128_8local(b_h, b_q[None, :], BV)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            # Load state index and check for invalid entries
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            # Only store if state index is valid (not NULL_BLOCK_ID=0)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_o += HV * V
        p_v += stride_v
        p_b += stride_b
        p_a += stride_a
