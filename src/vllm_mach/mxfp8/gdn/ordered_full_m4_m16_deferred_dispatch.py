# SPDX-License-Identifier: Apache-2.0
"""Original ordered-full W4 lifecycle, with only physical M4/M16 deferred.

Other decode rows and all scratch/materialize/reset functions stay with the
frozen ordered-full component. This module introduces no new allocation.
"""
from . import ordered_full_triton as original
from . import ordered_m4_deferred_triton as m4
from . import ordered_m16_deferred_triton as m16

ROWS = (1, 2, 4, 8, 16) + tuple(range(24, 129, 8))
W = 4

DeferredScratch = original.DeferredScratch
required_shapes = original.required_shapes
validate = original.validate
materialize_slots = original.materialize_slots
reset_slots = original.reset_slots


def decode(mixed_qkv, a, b, A_log, dt_bias, base, out, indices, scratch,
           scale=128 ** -0.5):
    rows = int(mixed_qkv.shape[0])
    if rows == 4:
        return m4.decode(mixed_qkv, a, b, A_log, dt_bias, base, out, indices,
                         scratch, scale=scale)
    if rows == 16:
        return m16.decode(mixed_qkv, a, b, A_log, dt_bias, base, out, indices,
                          scratch, scale=scale)
    return original.decode(mixed_qkv, a, b, A_log, dt_bias, base, out, indices,
                           scratch, scale=scale)
