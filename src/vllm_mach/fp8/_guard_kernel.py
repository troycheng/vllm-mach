# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Tri Dao. Derived from vLLM layernorm_guard.py.
# R128: ROW1 local4, ROW2/4 local8; actual SM120 legacy FMA order.
import triton
import triton.language as tl

@triton.jit
def _old_local8_square_sum(x, ROWS_PER_BLOCK: tl.constexpr, BLOCK_N: tl.constexpr):
    tl.static_assert((ROWS_PER_BLOCK == 1 or ROWS_PER_BLOCK == 2 or ROWS_PER_BLOCK == 4) and BLOCK_N == 128)
    if ROWS_PER_BLOCK == 1:
        q02, q13 = tl.split(tl.reshape(x, (ROWS_PER_BLOCK, BLOCK_N // 4, 2, 2)))
        q0, q2 = tl.split(q02)
        q1, q3 = tl.split(q13)
        acc = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $1;', constraints='=f,f', args=[q1], dtype=tl.float32, is_pure=True, pack=1)
        acc = _old_fma(q0, acc)
        acc = _old_fma(q2, acc)
        acc = _old_fma(q3, acc)
        return tl.sum(acc, axis=1)
    q0246, q1357 = tl.split(tl.reshape(x, (ROWS_PER_BLOCK, BLOCK_N // 8, 2, 2, 2)))
    q04, q26 = tl.split(q0246)
    q15, q37 = tl.split(q1357)
    q0, q4 = tl.split(q04)
    q2, q6 = tl.split(q26)
    q1, q5 = tl.split(q15)
    q3, q7 = tl.split(q37)
    acc = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $1;', constraints='=f,f', args=[q1], dtype=tl.float32, is_pure=True, pack=1)
    acc = _old_fma(q0, acc)
    acc = _old_fma(q2, acc)
    acc = _old_fma(q3, acc)
    acc = _old_fma(q4, acc)
    acc = _old_fma(q5, acc)
    acc = _old_fma(q6, acc)
    acc = _old_fma(q7, acc)
    return tl.sum(acc, axis=1)

@triton.jit
def _old_fma(x, acc):
    return tl.inline_asm_elementwise('fma.rn.f32 $0, $1, $1, $2;', constraints='=f,f,f', args=[x, acc], dtype=tl.float32, is_pure=True, pack=1)

@triton.heuristics({'HAS_BIAS': lambda args: args['B'] is not None, 'HAS_Z': lambda args: args['Z'] is not None})
@triton.jit
def layer_norm_fwd_kernel(X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M, N: tl.constexpr, eps, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_Z: tl.constexpr, NORM_BEFORE_GATE: tl.constexpr, IS_RMS_NORM: tl.constexpr, ACTIVATION: tl.constexpr):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)
    if HAS_Z and (not NORM_BEFORE_GATE):
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == 'sigmoid':
            x *= tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = _old_local8_square_sum(xbar, ROWS_PER_BLOCK, BLOCK_N) / N
    else:
        xbar = tl.where(mask, x, 0.0)
        var = _old_local8_square_sum(xbar, ROWS_PER_BLOCK, BLOCK_N) / N
        mean = 0.0
    rstd = tl.rsqrt(var + eps)
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]
    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]
    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == 'sigmoid':
            y *= tl.sigmoid(z)
    tl.store(Y_base, y, mask=mask)
