# SPDX-License-Identifier: Apache-2.0
"""2B high-M W4 candidate. Every unlisted row deliberately uses stock."""
from . import ordered_allm_triton as high
ROWS = (32,48,64,96,128,160)
W = high.W
DeferredScratch = high.DeferredScratch
required_shapes = high.required_shapes
validate = high.validate
materialize_slots = high.materialize_slots
reset_slots = high.reset_slots

def decode(*args, **kwargs):
    return high.decode(*args, **kwargs)
