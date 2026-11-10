# SPDX-License-Identifier: Apache-2.0
"""Finite attention merge choice; all other descriptors keep the stock kernel."""
import hashlib
from pathlib import Path


def identity():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).with_name('_attention_reduction_kernel.py')]}


def selected(profile, segments, head_size, padded_head_size, bf16, use_fp8):
    return (profile == 'qwen35-2b-mxfp8-champion-v1' and segments == 16
            and head_size == padded_head_size == 256 and bf16 and not use_fp8)


def reduction(stock, *, profile, segments, head_size, padded_head_size, bf16, use_fp8):
    if not selected(profile, segments, head_size, padded_head_size, bf16, use_fp8):
        return stock
    from ._attention_reduction_kernel import reduce_segments_explicit_old_lane
    return reduce_segments_explicit_old_lane
