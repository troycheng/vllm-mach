# SPDX-License-Identifier: Apache-2.0
"""Mixed decode with ordered FP32 state; retain stock convolution and prefill.

The source-pinned vLLM core is cloned once at worker setup. Only its mixed
recurrent branch changes. Prefill slots are materialized; decode slots keep
pending W4 terms and share the existing pure-decode lifecycle and scratch.
"""
import ast
from collections import Counter
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import textwrap

PROTOCOL = "mixed-lean-stockmath-w4-v1"
_COUNTS = {}
_VALIDATED = {}
_LAYER_CACHE = {}
_CLONE_SHA = None
_ORIGINAL_SHA = None


def _clone_core(original, namespace):
    """Change exactly the top-level 2.2 split branch, preserving every other AST node."""
    global _CLONE_SHA, _ORIGINAL_SHA
    original_source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(original_source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError('Expected one stock GDN core function')
    fn = tree.body[0]
    selected = []
    for i, node in enumerate(fn.body):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name) or node.test.id != 'split_non_spec':
            continue
        first = node.body[0]
        if isinstance(first, ast.Assign) and any(isinstance(t, ast.Tuple) and
                [e.id for e in t.elts if isinstance(e, ast.Name)] ==
                ['query_decode', 'key_decode', 'value_decode'] for t in first.targets):
            selected.append((i, node))
    if len(selected) != 1:
        raise RuntimeError('Unique stock mixed decode branch absent')
    index, old_node = selected[0]
    replacement = ast.parse('''if split_non_spec:
    core_attn_out_decode = _mach_mixed_decode(
        self, mixed_qkv_non_spec, a, b, ssm_state,
        non_spec_state_indices_tensor, num_decode_tokens, core_attn_out)
else:
    core_attn_out_decode = None
''').body[0]
    edited = ast.parse(original_source)
    edited_fn = edited.body[0]
    edited_fn.body[index] = replacement
    for i, (before, after) in enumerate(zip(fn.body, edited_fn.body, strict=True)):
        if i != index:
            assert ast.dump(before) == ast.dump(after), i
    assert ast.dump(fn.body[index]) != ast.dump(edited_fn.body[index])
    ast.fix_missing_locations(edited)
    clone_source = ast.unparse(edited) + '\n'
    _ORIGINAL_SHA = hashlib.sha256(original_source.encode()).hexdigest()
    _CLONE_SHA = hashlib.sha256(clone_source.encode()).hexdigest()
    scope = dict(namespace)
    exec(compile(clone_source, '<mach-mixed-gdn:_forward_core>', 'exec'), scope)
    clone = scope['_forward_core']
    if inspect.signature(clone) != inspect.signature(original):
        raise RuntimeError('Stock GDN core signature changed')
    return clone


def install(gw, gdn):
    """Called after the worker installs its pure-decode wrapper and SHA gate."""
    import torch
    from . import ordered_allm_triton as ordered, mixed_ordered_triton as stock_math
    GDN = gdn.QwenGatedDeltaNetAttention
    if getattr(GDN, "_mach_2b_mixed_gdn_installed", False):
        return
    if gw._READY or gw._VIEW_SCRATCH:
        raise RuntimeError("Install mixed GDN before preparing state pools")
    original_core = GDN._forward_core  # Published wrapper.
    assert gw._ORIG_CORE is not None

    def layer_cache(layer):
        base = layer.kv_cache[1]
        cached = _LAYER_CACHE.get(layer)
        if cached is not None and cached[0] is base:
            return cached
        key = gw._view_key(base)
        scratch = gw._VIEW_SCRATCH[key]
        if key not in _VALIDATED:
            _VALIDATED[key] = stock_math.validate(base, scratch)
        p, w = _VALIDATED[key]
        cached = (base, scratch, p, w)
        _LAYER_CACHE[layer] = cached
        return cached

    def lean_decode(layer, postconv, a, b, base, all_ids, d, output):
        current, scratch, p, w = layer_cache(layer)
        if current is not base:
            raise RuntimeError('Mixed GDN state binding changed')
        ids = all_ids[:d]
        target = output[:d].unsqueeze(1)
        stock_math._ordered_decode[(stock_math.NV, stock_math.HV, d)](
            postconv[:d], a[:d], b[:d], layer.A_log, layer.dt_bias,
            target, base, ids, scratch.pending_k, scratch.pending_d,
            scratch.coeff, scratch.prefix, scratch.age,
            postconv.stride(0), a.stride(0), b.stride(0), base.stride(0),
            ids.stride(0), p, w, 128**-0.5, stock_math.H, stock_math.HV,
            stock_math.V, stock_math.K, stock_math.BV, stock_math.NV,
            num_warps=4, num_stages=3)
        return target.transpose(0, 1)

    cloned_core = _clone_core(
        gw._ORIG_CORE, dict(gdn.__dict__, _mach_mixed_decode=lean_decode))

    @wraps(original_core)
    def core(self, mixed_qkv, b, a, core_attn_out, hidden_states=None):
        md = gw._metadata(self)
        eligible = (gw._READY and md is not None and self.prefix in gw._LAYERS
                    and md.spec_sequence_masks is None and md.num_spec_decodes == 0
                    and md.num_prefills > 0 and 24 <= md.num_decodes <= 160
                    and md.num_decode_tokens == md.num_decodes
                    and md.non_spec_state_indices_tensor is not None
                    and md.non_spec_state_indices_tensor.numel() >= md.num_decodes + md.num_prefills
                    and md.num_actual_tokens <= mixed_qkv.shape[0]
                    and md.num_actual_tokens > md.num_decodes)
        if not eligible:
            return original_core(self, mixed_qkv, b, a, core_attn_out, hidden_states)
        base, scratch, p, w = layer_cache(self)
        ids = md.non_spec_state_indices_tensor[md.num_decodes:md.num_decodes + md.num_prefills]
        # Same published materialize JIT/arguments, with layout validation
        # hoisted to the first encounter of each shared physical view.
        ordered._ordered_materialize[(ordered.NV, ordered.HV, ids.numel())](
            base, ids, scratch.pending_k, scratch.pending_d, scratch.coeff,
            scratch.prefix, scratch.age, base.stride(0), ids.stride(0), p, w,
            ordered.HV, ordered.V, ordered.K, ordered.BV, ordered.NV,
            num_warps=1, num_stages=3)
        result = cloned_core(self, mixed_qkv, b, a, core_attn_out, hidden_states)
        phase = 'capture' if torch.cuda.is_current_stream_capturing() else 'eager'
        counts = _COUNTS.setdefault(self.prefix, Counter())
        counts[phase + '_mixed_hit'] += 1
        return result

    GDN._forward_core = core
    GDN._mach_2b_mixed_gdn_installed = True


def snapshot():
    """Worker receipt; Python hits count eager calls and graph construction."""
    return {"protocol": PROTOCOL,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "original_core_source_sha256": _ORIGINAL_SHA,
            "clone_core_source_sha256": _CLONE_SHA,
            "scope": "non-spec mixed D24..160; stock convolution and prefill",
            "tile": {"BV": 32, "NV": 4, "decode_num_warps": 4,
                     "materialize_num_warps": 1},
            "validated_views": len(_VALIDATED),
            "python_counts_not_graph_replay": True,
            "counts_by_layer": {k: dict(v) for k, v in sorted(_COUNTS.items())}}
