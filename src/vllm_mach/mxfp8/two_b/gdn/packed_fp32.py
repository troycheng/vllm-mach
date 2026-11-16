# SPDX-License-Identifier: Apache-2.0
"""The source-pinned stock low-M packed kernel with its old local4 reductions."""
import ast
import copy
import hashlib
import linecache

_KERNEL = None
_LAUNCHES = {}


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from vllm.triton_utils import triton
    from vllm.third_party.flash_linear_attention.ops import fused_recurrent
    from ...gdn.legacy_reduction import _old_square_128, _old_dot_128

    original = fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel
    tree = ast.parse(original.src)
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
    filename = "<mach-2b-fp32-m16-packed>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.fn.__globals__, _old_square_128=_old_square_128,
                     _old_dot_128=_old_dot_128)
    exec(compile(source, filename, "exec"), namespace)
    _KERNEL = triton.jit(namespace[original.__name__])
    return _KERNEL


def decode(mixed, a, b, A_log, dt_bias, state, out, indices):
    """Called only by the prepared TP1 FP32 worker after stock convolution."""
    result = _kernel()[(4, mixed.shape[0] * 16)](
        mixed, a, b, A_log, dt_bias, out, state, state, indices, 128**-0.5,
        stride_mixed_qkv_tok=mixed.stride(0), stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0), stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0), stride_indices_seq=indices.stride(0),
        H=16, HV=16, K=128, V=128, BK=128, BV=32, SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True, SPLIT_BATCH_HEAD_GRID=False,
        num_warps=1, num_stages=3)
    if result is not None:
        _LAUNCHES[int(mixed.shape[0])] = {
            "hash": result.metadata.hash, "num_warps": result.metadata.num_warps,
            "num_stages": result.metadata.num_stages,
            "grid": [4, int(mixed.shape[0]) * 16],
            "jit_source_sha256": hashlib.sha256(_KERNEL.src.encode()).hexdigest()}
    return result


def receipt():
    return {"reduction_order": "legacy_local4_fp32", "launches": dict(_LAUNCHES)}
