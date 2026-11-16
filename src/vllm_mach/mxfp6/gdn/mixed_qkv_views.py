# SPDX-License-Identifier: Apache-2.0
"""Qualified native FP16 mixed-decode QKV views; original prefill/copy preserved."""
import ast
import __future__
import copy
import hashlib
from pathlib import Path
import types

QWEN_SHA = 'dbe8912cd418914ea1e1e19daea6d95c46379fe616e60fd8e0fbcbb6e788be0b'
MIXED_SHA = 'a5dd808134d6e7b5b51ad4af91372b4e34b8446d3d80854e838770e045f7f98b'
_STATE = None
_MARKER = '_mach_native_mixed_qkv_views'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(node):
    return ast.dump(node, include_attributes=False)

def compile_function(node, namespace, filename):
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    code = compile(tree, filename, 'exec', dont_inherit=True)
    function = next((x for x in code.co_consts if isinstance(x, types.CodeType)))
    return types.FunctionType(function, namespace)

def private_parts(mixed):
    """Retain the original admission and admitted update body; change allocation only."""
    assert digest(mixed.__file__) == MIXED_SHA
    tree = ast.parse(Path(mixed.__file__).read_text())
    original = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'update'))
    assert isinstance(original.body[0], ast.ImportFrom) and isinstance(original.body[1], ast.If)
    gate = ast.parse('def gate(layer, metadata, stock, values):\n    pass').body[0]
    gate.body = [copy.deepcopy(original.body[0]), ast.Return(value=ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(original.body[1].test)))]
    admitted = copy.deepcopy(original)
    admitted.name = 'admitted'
    admitted.body = copy.deepcopy(original.body[2:])
    assert ast.unparse(admitted.body[-1]) == "return _STATE['private'](**values)"
    admitted.body[-1:-1] = ast.parse("values['output'] = values['q'].new_empty(values['v'].shape)").body
    inverse = copy.deepcopy(admitted)
    del inverse.body[-2]
    assert ast.dump(ast.Module(body=inverse.body, type_ignores=[]), include_attributes=False) == ast.dump(ast.Module(body=original.body[2:], type_ignores=[]), include_attributes=False)
    return (compile_function(gate, vars(mixed), mixed.__file__), compile_function(admitted, vars(mixed), mixed.__file__))

def source_node(original):
    assert type(original) is types.FunctionType and original.__closure__ is None and (not hasattr(original, '__wrapped__'))
    path = Path(original.__code__.co_filename)
    tree = ast.parse(path.read_text())
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenGatedDeltaNetAttention'))
    node = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_core'))
    assert not node.decorator_list
    compiled = compile(path.read_text(), str(path), 'exec', dont_inherit=True)
    classcode = next((c for c in compiled.co_consts if isinstance(c, types.CodeType) and c.co_name == cls.name))
    expected = next((c for c in classcode.co_consts if isinstance(c, types.CodeType) and c.co_name == node.name))
    assert expected == original.__code__, 'Actual whole-module code differs'
    assert original.__globals__.get('__file__') == str(path)
    return node

def _make_can_view(torch):
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    from torch.overrides import _get_current_function_mode
    zero_tensor = getattr(torch, '_is_zerotensor', None)

    def can_view(layer, md, x, a, b):
        if _get_current_dispatch_mode() is not None or _get_current_function_mode() is not None or (not torch.is_inference_mode_enabled()) or torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting():
            return False
        if not getattr(layer, '_mach_native_mixed_fp16', False) or layer.tp_size != 2 or layer.num_k_heads != 16 or (layer.num_v_heads != 32) or (layer.head_k_dim != 128) or (layer.head_v_dim != 128) or (md.num_prefills <= 0) or (not 1 <= md.num_decodes <= 32) or (md.num_decode_tokens != md.num_decodes) or (md.num_spec_decodes != 0) or (md.spec_sequence_masks is not None):
            return False
        for value in (x, a, b):
            if type(value) is not torch.Tensor or not value.is_cuda or value.dtype != torch.bfloat16 or (value.ndim != 2) or (value.stride(-1) != 1) or value.is_neg() or value.is_conj() or (zero_tensor is not None and zero_tensor(value)):
                return False
        return x.shape[1] == 4096 and x.shape[0] >= md.num_decode_tokens and (x.stride(0) >= 4096) and (a.shape[1] == b.shape[1] == 16) and (a.shape[0] >= md.num_decode_tokens) and (b.shape[0] >= md.num_decode_tokens) and (a.stride(0) == 16) and (b.stride(0) == 16) and (a.device == b.device == x.device)
    return can_view

def transform(original_node):
    patched = copy.deepcopy(original_node)
    idx = next((i for i, n in enumerate(patched.body) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'strided_decode' for t in n.targets))))
    branch = patched.body[idx + 1]
    assert isinstance(branch, ast.If) and ast.unparse(branch.test) == 'split_non_spec'
    old = copy.deepcopy(branch.body)
    body = ast.parse('_qkv_handled, _qkv_output = _qkv_helper(self, attn_metadata, mixed_qkv_non_spec, a, b, ssm_state, non_spec_query_start_loc, non_spec_state_indices_tensor)\nif _qkv_handled:\n core_attn_out_decode = _qkv_output\nelse:\n pass').body
    body[1].orelse = copy.deepcopy(old)
    branch.body = body
    inverse = copy.deepcopy(patched)
    inverse.body[idx + 1].body = copy.deepcopy(inverse.body[idx + 1].body[1].orelse)
    assert dump(inverse) == dump(original_node), 'Full-function inverse mismatch'
    return (patched, dict(mode='ON', full_function_inverse_AST_equal=True, original_AST_sha256=hashlib.sha256(dump(original_node).encode()).hexdigest(), patched_AST_sha256=hashlib.sha256(dump(patched).encode()).hexdigest(), prefill_and_final_copy_unchanged=True))

def bind(node, original, helper):
    factory = ast.FunctionDef(name='_factory', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='_qkv_helper')], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[copy.deepcopy(node), ast.Return(value=ast.Name(id=node.name, ctx=ast.Load()))], decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    compiled = compile(tree, original.__code__.co_filename, 'exec', flags=original.__code__.co_flags & __future__.annotations.compiler_flag, dont_inherit=True)
    factorycode = next((c for c in compiled.co_consts if isinstance(c, types.CodeType)))
    fn = types.FunctionType(factorycode, original.__globals__)(helper)
    assert fn.__globals__ is original.__globals__ and fn.__code__.co_freevars == ('_qkv_helper',) and (fn.__closure__[0].cell_contents is helper)
    fn.__defaults__ = original.__defaults__
    fn.__kwdefaults__ = original.__kwdefaults__
    fn.__annotations__ = original.__annotations__
    fn.__qualname__ = original.__qualname__
    return fn

class Helpers:

    def __init__(self, qwen, mixed, torch):
        self.can_view = _make_can_view(torch)
        self.gate, self.admitted = private_parts(mixed)
        self.stock = qwen.fused_sigmoid_gating_delta_rule_update

    def values(self, layer, md, qkv, a, b, state, cu, ids):
        n = md.num_decode_tokens
        q, k, v = qkv
        return dict(A_log=layer.A_log, a=a[:n], b=b[:n], dt_bias=layer.dt_bias, q=q, k=k, v=v, initial_state=state, inplace_final_state=True, cu_seqlens=cu[:md.num_decodes + 1], ssm_state_indices=ids, use_qk_l2norm_in_kernel=True, output=None)

    def admitted_views(self, layer, md, x, a, b, state, cu, ids):
        if md.num_decode_tokens <= 1 or not self.can_view(layer, md, x, a, b):
            return None
        qkv = layer.rearrange_mixed_qkv_flashinfer(x[:md.num_decode_tokens])
        values = self.values(layer, md, qkv, a, b, state, cu, ids)
        if not self.gate(layer, md, self.stock, values):
            return None
        return values

    def direct(self, layer, md, x, a, b, state, cu, ids):
        values = self.admitted_views(layer, md, x, a, b, state, cu, ids)
        if values is None:
            return (False, None)
        output, _ = self.admitted(layer, md, self.stock, **values)
        return (True, output)


def prepare(layer):
    """Select once on prepared native MoE layers, before capture and _WARMED."""
    global _STATE
    if (getattr(layer, '_mach_native_mixed_fp16', False) is not True
            or getattr(layer, '_mach_native_legacy', False) is not True):
        return False
    import torch
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
    from . import mixed_fp16
    cls = qwen.QwenGatedDeltaNetAttention
    if (type(layer) is not cls or layer.tp_size != 2
            or layer.num_k_heads != 16 or layer.num_v_heads != 32
            or layer.head_k_dim != 128 or layer.head_v_dim != 128
            or layer.get_state_dtype()[1] != torch.float16):
        return False
    marker = vars(layer).get(_MARKER)
    if marker is not None:
        if (type(marker) is dict and vars(layer).get('_forward_core') is marker.get('owner')
                and marker.get('original') is cls._forward_core):
            return False
        raise RuntimeError('Mixed QKV views instance owner changed')
    if '_forward_core' in vars(layer):
        return False
    original = cls._forward_core
    try:
        assert digest(qwen.__file__) == QWEN_SHA, 'Qwen source changed'
        assert digest(mixed_fp16.__file__) == MIXED_SHA, 'Mixed FP16 source changed'
        assert original.__globals__ is vars(qwen), 'Core globals changed'
        if _STATE is None:
            node = source_node(original)
            node, proof = transform(node)
            helper = Helpers(qwen, mixed_fp16, torch)
            candidate = bind(node, original, helper.direct)
            _STATE = dict(original=original, original_code=original.__code__, candidate=candidate, helper=helper,
                          qwen=qwen, mixed=mixed_fp16, torch=torch, proof=proof)
        else:
            assert (_STATE['original'] is original and original.__code__ is _STATE['original_code'] and _STATE['qwen'] is qwen
                    and _STATE['mixed'] is mixed_fp16 and _STATE['torch'] is torch), 'Qualified core owner changed'
    except (AssertionError, StopIteration, OSError, TypeError) as error:
        raise RuntimeError('Mixed QKV views source/code qualification failed: ' + str(error)) from error
    owner = types.MethodType(_STATE['candidate'], layer)
    setattr(layer, '_forward_core', owner)
    setattr(layer, _MARKER, dict(owner=owner, original=original))
    return True
