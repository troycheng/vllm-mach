# SPDX-License-Identifier: Apache-2.0
"""Post-capture exact16 metadata classification for the FP32 hybrid-head profile.

Only ordinary non-spec CPU metadata classification changes. The upstream builder
class, every other statement and its spec path remain intact.
"""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import types

SOURCE_SHA = "5b35efdbe0f6842b45db52e2f4bcdec4324a088de330080090bd5fd602252d05"
BUILD_AST_SHA = "518783b679836c4c6a2ef0dc3c8e8d0c26e650b9854535a6020c12622402c292"
_STATE = "_mach_2b_metadata_classification"
_COUNTS = "num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens"
_HELPER = "_mach_exact16_classifier"


def exact_decode16(m):
    """Return counts only when all original classification predicates are proven."""
    if type(m) is not CommonAttentionMetadata:
        return None
    # Refuse modes before inspecting any Tensor property (it can be observable).
    if _get_current_dispatch_mode() is not None or _get_current_function_mode() is not None:
        return None
    if torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting():
        return None
    if type(m.num_reqs) is not int or m.num_reqs != 16:
        return None
    if type(m.num_actual_tokens) is not int or m.num_actual_tokens != 16:
        return None
    if type(m.max_query_len) is not int or m.max_query_len != 1:
        return None
    q, p, s = m.query_start_loc_cpu, m.is_prefilling, m.seq_lens_cpu_upper_bound
    for value, dtype, size in ((q, torch.int32, 17), (p, torch.bool, 16), (s, torch.int32, 16)):
        if type(value) is not torch.Tensor or value.device.type != "cpu":
            return None
        if value.dtype != dtype or value.ndim != 1 or value.shape[0] != size:
            return None
    if q.tolist() != _Q17 or any(p.tolist()):
        return None
    # q lengths are exactly 1 and no_prior_state & all-false is_prefilling is false.
    # The original split therefore has no prefills, including when seq_lens == 1.
    return _FAST_RESULT


def _bound_helper(torch, common_type, dispatch_mode, function_mode):
    # Deliberately do not copy a runtime module's globals or import the experiment.
    namespace = {"torch": torch, "CommonAttentionMetadata": common_type,
                 "_get_current_dispatch_mode": dispatch_mode,
                 "_get_current_function_mode": function_mode,
                 "_Q17": list(range(17)), "_FAST_RESULT": (16, 0, 16, 0)}
    return types.FunctionType(exact_decode16.__code__, namespace, "exact_decode16")


def _transform(node):
    original = copy.deepcopy(node)
    if (not isinstance(node, ast.FunctionDef) or node.name != "build"
            or node.decorator_list
            or hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() != BUILD_AST_SHA):
        raise RuntimeError("2B metadata builder AST differs from the qualified source")
    build = copy.deepcopy(node)
    if any(isinstance(n, ast.Name) and n.id in (_HELPER, "_mach_classification_counts")
           for n in ast.walk(build)):
        raise RuntimeError("2B metadata helper local name collision")
    branch = next(n for n in build.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "spec_sequence_masks is None")
    cut = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "num_spec_decode_tokens" for t in n.targets))
    statements = branch.body[:cut]
    if len(statements) != 6:
        raise RuntimeError("2B metadata classification boundary changed")
    inserted = ast.parse(
        f"_mach_classification_counts = {_HELPER}(m)\n"
        "if _mach_classification_counts is None:\n    pass\n"
        f"else:\n    {_COUNTS} = _mach_classification_counts\n").body
    inserted[1].body = statements
    rest = branch.body[cut:]
    branch.body = inserted + rest
    modified = copy.deepcopy(build)
    branch.body = statements + rest
    if ast.dump(build, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise RuntimeError("2B metadata inverse AST does not reconstruct the original")
    return modified


def _factory(original, modified, helper):
    """Bind the sole new closure without global mutation or re-evaluating defaults."""
    if original.__closure__ or modified.decorator_list:
        raise RuntimeError("2B metadata builder must be an undecorated non-closure")
    nested = copy.deepcopy(modified)
    nested.returns = None
    for argument in (*nested.args.posonlyargs, *nested.args.args, *nested.args.kwonlyargs):
        argument.annotation = None
    if nested.args.vararg is not None:
        nested.args.vararg.annotation = None
    if nested.args.kwarg is not None:
        nested.args.kwarg.annotation = None
    nested.args.defaults = []
    nested.args.kw_defaults = [None] * len(nested.args.kwonlyargs)
    factory = ast.parse(f"def _factory({_HELPER}):\n    pass\n").body[0]
    factory.body = [nested, ast.Return(value=ast.Name(id="build", ctx=ast.Load()))]
    tree = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    module_code = compile(tree, original.__code__.co_filename, "exec", dont_inherit=True)
    factory_code = next(c for c in module_code.co_consts if isinstance(c, types.CodeType))
    fn = types.FunctionType(factory_code, original.__globals__)(helper)
    fn.__defaults__ = original.__defaults__
    fn.__kwdefaults__ = original.__kwdefaults__
    fn.__annotations__ = original.__annotations__
    fn.__name__, fn.__qualname__, fn.__module__ = original.__name__, original.__qualname__, original.__module__
    if fn.__globals__ is not original.__globals__ or fn.__code__.co_freevars != (_HELPER,):
        raise RuntimeError("2B metadata factory changed its binding boundary")
    return fn


def _source(builder_type):
    original = builder_type.build
    if type(original) is not types.FunctionType or hasattr(original, "__wrapped__") or original.__closure__:
        raise RuntimeError("2B metadata builder callable identity changed")
    module = sys.modules.get(builder_type.__module__)
    if module is None or original.__globals__ is not vars(module):
        raise RuntimeError("2B metadata builder module globals changed")
    path = Path(original.__code__.co_filename).resolve()
    if Path(module.__file__).resolve() != path or hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("2B metadata builder source identity changed")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "GDNAttentionMetadataBuilder")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    compiled = compile(path.read_text(), original.__code__.co_filename, "exec", dont_inherit=True)
    class_code = next(c for c in compiled.co_consts if isinstance(c, types.CodeType) and c.co_name == cls.name)
    build_code = next(c for c in class_code.co_consts if isinstance(c, types.CodeType) and c.co_name == "build")
    if build_code != original.__code__:
        raise RuntimeError("2B metadata builder code differs from its source")
    return original, _transform(node)


def _selected(worker):
    contract = getattr(worker, "_mach_2b_contract", {})
    if (contract.get("mode") != "production" or contract.get("ssm_dtype") != "float32"
            or contract.get("hybrid_greedy_head") is not True):
        return False
    installed = getattr(worker, "_mach_2b_hybrid_head", None)
    state = installed.get("state") if type(installed) is dict else None
    if state is None or state.candidates != 128 or state._head_graphs is not None:
        return False
    runner = worker.model_runner
    if runner.speculative_config is not None:
        return False
    manager = runner.cudagraph_manager
    return bool(manager is not None and manager._graphs_captured and manager.graphs)


def _runtime():
    import torch
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    from torch.overrides import _get_current_function_mode
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    return GDNAttentionMetadataBuilder, _bound_helper(
        torch, CommonAttentionMetadata, _get_current_dispatch_mode, _get_current_function_mode)


def install(worker):
    if hasattr(worker, _STATE) or not _selected(worker):
        return False
    builder_type, helper = _runtime()
    builders = [builder for groups in worker.model_runner.attn_groups for group in groups
                for builder in group.metadata_builders if type(builder) is builder_type]
    if len(builders) != 3 or len({id(builder) for builder in builders}) != 3:
        raise RuntimeError("2B metadata classification requires three exact GDN builders")
    if any("build" in vars(builder) for builder in builders):
        return False
    original, modified = _source(builder_type)
    fn = _factory(original, modified, helper)
    replacements = [(builder, types.MethodType(fn, builder)) for builder in builders]
    # Preflight every owner before any instance mutation; keep the class/update intact.
    if any(builder.build.__func__ is not original or builder.build.__self__ is not builder
           or "build" in vars(builder) for builder in builders):
        return False
    applied = []
    try:
        for builder, replacement in replacements:
            builder.build = replacement
            applied.append(builder)
        setattr(worker, _STATE, {"bindings": replacements, "original": original})
    except BaseException:
        for builder in reversed(applied):
            del builder.build
        raise
    return True


def inspect_worker(worker):
    state = getattr(worker, _STATE, None)
    if state is None:
        return {"installed": False}
    if any(vars(builder).get("build") is not fn for builder, fn in state["bindings"]):
        raise RuntimeError("2B metadata classifier instance owner changed")
    return {"installed": True, "builder_count": len(state["bindings"]),
            "source_sha256": SOURCE_SHA, "build_AST_sha256": BUILD_AST_SHA,
            "selection": "production FP32 SSM, hybrid128, private headgraph OFF, non-spec exact16"}
