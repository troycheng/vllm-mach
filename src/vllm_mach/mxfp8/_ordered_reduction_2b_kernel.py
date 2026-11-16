import triton
import triton.language as tl

@triton.jit
def _old_ordered_add(a, b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[a, b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _old_ordered_mul(a, b):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[a, b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _old_local_sum(values, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, GROUP: tl.constexpr):
    tl.static_assert(GROUP == 4 or GROUP == 8)
    if GROUP == 4:
        q02, q13 = tl.split(tl.reshape(values, (XBLOCK, R0_BLOCK // 4, 2, 2)))
        q0, q2 = tl.split(q02)
        q1, q3 = tl.split(q13)
    else:
        q0246, q1357 = tl.split(tl.reshape(values, (XBLOCK, R0_BLOCK // 8, 2, 2, 2)))
        q04, q26 = tl.split(q0246)
        q15, q37 = tl.split(q1357)
        q0, q4 = tl.split(q04)
        q2, q6 = tl.split(q26)
        q1, q5 = tl.split(q15)
        q3, q7 = tl.split(q37)
    local = _old_ordered_add(q0, q1)
    local = _old_ordered_add(local, q2)
    local = _old_ordered_add(local, q3)
    if GROUP == 8:
        local = _old_ordered_add(local, q4)
        local = _old_ordered_add(local, q5)
        local = _old_ordered_add(local, q6)
        local = _old_ordered_add(local, q7)
    return tl.sum(local, 1)[:, None]

@triton.jit
def _legacy_add(a, b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[a, b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _legacy_mul(a, b):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[a, b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _legacy_fma(a, b, c):
    return tl.inline_asm_elementwise('fma.rn.f32 $0, $1, $2, $3;', constraints='=f,f,f,f', args=[a, b, c], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _legacy_gdn_square_sum(values, xmask, XBLOCK: tl.constexpr):
    q0246, q1357 = tl.split(tl.reshape(values, (XBLOCK, 16, 2, 2, 2)))
    q04, q26 = tl.split(q0246)
    q15, q37 = tl.split(q1357)
    q0, q4 = tl.split(q04)
    q2, q6 = tl.split(q26)
    q1, q5 = tl.split(q15)
    q3, q7 = tl.split(q37)
    local = _legacy_fma(q0, q0, _legacy_mul(q1, q1))
    local = _legacy_fma(q2, q2, local)
    local = _legacy_add(_legacy_mul(q3, q3), local)
    local = _legacy_fma(q4, q4, local)
    local = _legacy_add(_legacy_mul(q5, q5), local)
    local = _legacy_fma(q6, q6, local)
    local = _legacy_add(_legacy_mul(q7, q7), local)
    local = tl.where(xmask, local, 0.0)
    return tl.sum(local, 1)[:, None]
