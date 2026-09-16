"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path

from flashinfer.jit.core import JitSpec, gen_jit_spec, sm120a_nvcc_flags

# Read the wheel-packaged CUDA source; generated artifacts stay in FlashInfer's
# writable JIT cache rather than the installed package directory.
_GDN_FUSED_DECODE_KERNEL_DIR = Path(__file__).resolve().parent


def gen_gdn_fused_decode_module(
    hidden: int,
    n_ba: int,
    qkv_dim: int,
    h_q: int,
    hv: int,
    d: int,
    conv_width: int,
    conv_state_len: int,
    state_bits: int,
) -> JitSpec:
    """JIT spec for one registered fused-GDN-decode layer geometry (SM120).

    Single translation unit, compiled with the ``sm120a`` flags (hence the
    CUDA >= 12.8 requirement the callers gate on).  The source lives next to
    its Python module inside this package and is built only for this op.

    The layer geometry is a compile-time parameter of the translation unit
    (the block shape and warp->row mapping do not depend on it, only the
    sizes do), so it is passed as ``-D`` defines and folded into the module
    name: one module per geometry, and a serving process runs one model,
    hence one module.  The kernel stays B-dynamic -- batch size, the query
    scale and the conv-state strides remain runtime parameters.
    """
    geometry = {
        "HIDDEN": hidden,
        "N_BA": n_ba,
        "QKV_DIM": qkv_dim,
        "H_Q": h_q,
        "HV": hv,
        "D": d,
        "CONV_WIDTH": conv_width,
        "CONV_STATE_LEN": conv_state_len,
        "STATE_BITS": state_bits,
    }
    suffix = "_".join(f"{key.lower()}{value}" for key, value in geometry.items())
    return gen_jit_spec(
        f"mach_gdn_state_v2_sm120_{suffix}",
        [_GDN_FUSED_DECODE_KERNEL_DIR / "gdn_fused_decode_sm120.cu"],
        extra_cuda_cflags=sm120a_nvcc_flags
        + [f"-DFI_GDN_{key}={value}" for key, value in geometry.items()],
    )
