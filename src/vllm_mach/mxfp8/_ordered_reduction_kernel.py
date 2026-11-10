"""Finite SM120 legacy local reduction expression; no loading/normalization."""
import triton
import triton.language as tl


@triton.jit
def _old_ordered_add(a, b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', constraints='=f,f,f',
                                   args=[a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _old_ordered_mul(a, b):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f',
                                   args=[a, b], dtype=tl.float32, is_pure=True, pack=1)


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
    local = _old_ordered_add(q2, local)
    local = _old_ordered_add(q3, local)
    if GROUP == 8:
        local = _old_ordered_add(q4, local)
        local = _old_ordered_add(q5, local)
        local = _old_ordered_add(q6, local)
        local = _old_ordered_add(q7, local)
    return tl.sum(local, 1)[:, None]
