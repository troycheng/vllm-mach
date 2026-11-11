# SPDX-License-Identifier: Apache-2.0
"""The qualified Block M4 stock packed kernel with its old local4 reductions."""
import ast
import copy
import hashlib
import linecache
import inspect
import os
from pathlib import Path

from .profile import NAME

_KERNEL = None
_STATE = None
_NAME = 'fused_recurrent_gated_delta_rule_packed_decode_kernel'
_SOURCE_SHA = 'b065f160e94371192ce743bcd1ce3a9c75ddcecf31be243af4d55b39a898685e'
_JIT_SHA = '779559021a36214411ff57dce5f80baeecf83e854f7761311ba21487a41b3761'
_CLONE_SHA = '1432bdcd729fb5bfa9e3f0865498a44e4762b4373e3f8a501a244e3b35e5336b'
_ORIGINAL = None


def rewrite(source):
    tree = ast.parse(source)
    before = ast.dump(tree)
    replacements = {
        "tl.sum(b_q * b_q)": "_old_square_128(b_q)",
        "tl.sum(b_k * b_k)": "_old_square_128(b_k)",
        "tl.sum(b_h * b_k[None, :], 1)": "_old_dot_128(b_h, b_k[None, :], BV)",
        "tl.sum(b_h * b_q[None, :], 1)": "_old_dot_128(b_h, b_q[None, :], BV)",
    }
    counts = dict.fromkeys(replacements, 0)
    reverse = {ast.dump(ast.parse(v, mode="eval").body):
               ast.parse(k, mode="eval").body for k, v in replacements.items()}

    class Replace(ast.NodeTransformer):
        def visit_Call(self, node):
            key = ast.unparse(node)
            if key in replacements:
                counts[key] += 1
                return ast.copy_location(ast.parse(replacements[key], mode="eval").body, node)
            return self.generic_visit(node)

    class Undo(ast.NodeTransformer):
        def visit_Call(self, node):
            return reverse.get(ast.dump(node), self.generic_visit(node))

    tree = Replace().visit(tree)
    if any(n != 1 for n in counts.values()) or ast.dump(Undo().visit(copy.deepcopy(tree))) != before:
        raise RuntimeError("Stock packed reduction source changed")
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    return source


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from vllm.triton_utils import triton
    from ..mxfp8.gdn.legacy_reduction import _old_square_128, _old_dot_128

    original = _ORIGINAL
    if original is None or hashlib.sha256(original.src.encode()).hexdigest() != _JIT_SHA:
        raise RuntimeError("Block-FP8 packed source changed")
    source = rewrite(original.src)
    filename = "<mach-block-fp8-m4-packed>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.fn.__globals__, _old_square_128=_old_square_128,
                     _old_dot_128=_old_dot_128)
    exec(compile(source, filename, "exec"), namespace)
    candidate = triton.jit(namespace[original.__name__])
    if hashlib.sha256(candidate.src.encode()).hexdigest() != _CLONE_SHA:
        raise RuntimeError("Block-FP8 packed clone arithmetic changed")
    _KERNEL = candidate
    return _KERNEL



def admitted(bound, float32):
    """Only the actual qualified Block M4 FP32/local4 launch; others stay stock."""
    return (bound['H'] == 16 and bound['HV'] == 32
            and bound['K'] == bound['V'] == bound['BK'] == 128 and bound['BV'] == 32
            and bound['h0'].dtype == bound['ht'].dtype == float32
            and bound['mixed_qkv'].shape[0] == 4
            and not bound['SPLIT_BATCH_HEAD_GRID']
            and bound['USE_QK_L2NORM_IN_KERNEL']
            and bound['SOFTPLUS_THRESHOLD'] == 20.0 and bound['scale'] == 128**-0.5)


def install():
    global _STATE, _ORIGINAL
    if os.environ.get('VLLM_MACH_PROFILE') != NAME:
        return False
    if _STATE is not None:
        from vllm.third_party.flash_linear_attention.ops import fused_recurrent
        if getattr(fused_recurrent, _NAME) is not _STATE['proxy']:
            raise RuntimeError('Block-FP8 packed binding changed')
        return False
    import torch
    from vllm.third_party.flash_linear_attention.ops import fused_recurrent
    if hashlib.sha256(Path(fused_recurrent.__file__).read_bytes()).hexdigest() != _SOURCE_SHA:
        raise RuntimeError('Block-FP8 packed source file changed')
    original = getattr(fused_recurrent, _NAME)
    if hashlib.sha256(original.src.encode()).hexdigest() != _JIT_SHA:
        raise RuntimeError('Block-FP8 packed JIT source changed')
    _ORIGINAL = original
    candidate = _kernel()
    signature = inspect.signature(original.fn)
    state = dict(calls=0, capturing_calls=0, fallback_calls=0, compiled={})

    class Proxy:
        def __getattr__(self, key):
            return getattr(original, key)

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                bound = signature.bind(*args, **{k: v for k, v in kwargs.items()
                                                 if k in signature.parameters}).arguments
                if not admitted(bound, torch.float32):
                    state['fallback_calls'] += 1
                    return original[grid](*args, **kwargs)
                if (bound['h0'].data_ptr() != bound['ht'].data_ptr()
                        or tuple(bound['h0'].stride()) != (540672, 16384, 128, 1)
                        or bound['stride_mixed_qkv_tok'] != 12288
                        or bound['stride_init_state_token'] != 540672
                        or bound['stride_final_state_token'] != 540672
                        or kwargs.get('num_warps') != 1 or kwargs.get('num_stages') != 3):
                    raise RuntimeError('Block-FP8 packed qualified layout/config changed')
                state['calls'] += 1
                state['capturing_calls'] += int(torch.cuda.is_current_stream_capturing())
                result = candidate[grid](*args, **kwargs)
                if result is not None:
                    state['compiled'][str(result.hash)] = dict(num_warps=result.metadata.num_warps,
                        num_stages=result.metadata.num_stages, shared=result.metadata.shared)
                return result
            return launch

    state['proxy'] = Proxy()
    setattr(fused_recurrent, _NAME, state['proxy'])
    _STATE = state
    return True


def receipt():
    if _STATE is None:
        return dict(installed=False)
    from vllm.third_party.flash_linear_attention.ops import fused_recurrent
    if getattr(fused_recurrent, _NAME) is not _STATE['proxy']:
        raise RuntimeError('Block-FP8 packed binding changed')
    return dict(installed=True, original_jit_sha256=_JIT_SHA, clone_jit_sha256=_CLONE_SHA,
                reduction_order='legacy_local4_fp32',
                **{k: v for k, v in _STATE.items() if k != 'proxy'})
