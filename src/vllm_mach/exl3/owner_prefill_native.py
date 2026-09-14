# SPDX-License-Identifier: Apache-2.0
"""Load the optional owner collective wheel without a build at service startup."""
from importlib import metadata, util

import torch

ABI = 'owner-prefill-v1'
_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    if metadata.version('flashinfer-python') != '0.6.18':
        raise RuntimeError('Owner prefill requires flashinfer-python==0.6.18')
    for name in ('mach_owner_prefill_ext', 'mach_owner_local_ext',
                 'mach_owner_ragged_ext', 'mach_owner_ragged_local_ext'):
        spec = util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError('Build and install native/owner_prefill before enabling owner prefill')
        torch.ops.load_library(spec.origin)
    if torch.ops.mach_owner_build.abi() != ABI:
        raise RuntimeError('Owner prefill native ABI mismatch; rebuild native/owner_prefill')
    required = {
        'mach_owner': ('reduce_owner', 'gather_mx8'),
        'mach_owner_local': ('ordered_sum_norm',),
        'mach_owner_ragged': ('reduce_owner', 'gather_mx8_padded'),
        'mach_owner_ragged_local': ('ordered_sum_norm',),
    }
    for namespace, operators in required.items():
        for operator in operators:
            if not hasattr(getattr(torch.ops, namespace), operator):
                raise RuntimeError(f'Owner prefill native operator missing: {namespace}::{operator}')
    _loaded = True
