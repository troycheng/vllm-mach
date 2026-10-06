# SPDX-License-Identifier: Apache-2.0
"""W4 raw-alpha GDN dispatch across low and high decode row counts.

This module adds no kernel or allocation. M1/2/8 uses the full-store,
age-zero ordered low-M component; M24..128 step 8 uses deferred ordered all-M.
Both consume the same caller-owned raw-alpha W4 scratch layout. Other rows
are rejected so the worker hook can use its stock fallback deliberately.
"""

from . import lowm_allshape_triton as low
from . import ordered_allm_triton as high

LOW_ROWS = (1, 2, 8)
HIGH_ROWS = tuple(range(24, 129, 8))
ROWS = LOW_ROWS + HIGH_ROWS
W = 4

if (low.W != high.W or low.W != W or
        (low.HV, low.V, low.K, low.BV, low.NV) !=
        (high.HV, high.V, high.K, high.BV, high.NV) or
        low.DeferredScratch.__dataclass_fields__.keys() !=
        high.DeferredScratch.__dataclass_fields__.keys()):
    raise RuntimeError("low/high ordered W4 scratch contract differs")

DeferredScratch = high.DeferredScratch
required_shapes = high.required_shapes
validate = high.validate
materialize_slots = high.materialize_slots
reset_slots = high.reset_slots


def decode(mixed_qkv, a, b, A_log, dt_bias, base, out, indices, scratch,
           scale=high.K ** -0.5):
    m = int(mixed_qkv.shape[0])
    if m in LOW_ROWS:
        return low.decode(mixed_qkv, a, b, A_log, dt_bias, base, out,
                          indices, scratch, scale=scale)
    if m in HIGH_ROWS:
        return high.decode(mixed_qkv, a, b, A_log, dt_bias, base, out,
                           indices, scratch, scale=scale)
    raise ValueError(f"ordered full W4 has no M{m} decode; use stock fallback")
