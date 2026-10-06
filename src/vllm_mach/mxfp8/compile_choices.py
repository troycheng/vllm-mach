# SPDX-License-Identifier: Apache-2.0

"""Rebuild the champion's Triton choices from source and geometry.

No compiled cache, best-config file, or former run directory is an input.
Only a matched CachingAutotuner gets one reviewed Config. Its reduction tree
and function body are untouched. Compilation is deliberately synchronous:
install in both parent and GPU workers before any generated module is loaded.
The profile must retain TORCHINDUCTOR_COMPILE_THREADS=1 on every cold start.
"""
from __future__ import annotations

import ast
from collections import Counter
import copy
from functools import wraps
import hashlib
import inspect as pyinspect
import json
import os
from pathlib import Path
import threading
import textwrap
import weakref
from typing import Any

_POLICY_FILE = Path(__file__).parent / "data" / "compile_choices.json"
_ACTIVE_POLICY_FILE = _POLICY_FILE
_POLICY_MODE: str | None = None
_INSTALLED = False
_ORIGINAL_INIT: Any = None
_LOCK = threading.RLock()
_BINDING_HITS: Counter = Counter()
_SEMANTIC_HITS: Counter = Counter()
_UNKNOWN_HITS: Counter = Counter()
_MISMATCHES: list[dict[str, Any]] = []
_AUTOTUNERS: list[tuple[Any, dict[str, Any]]] = []
_POLICY: dict[str, Any] | None = None


class CompileChoiceMismatch(RuntimeError):
    """An expected geometry changed source or lacks its necessary site label."""


def _json_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _triton_type_code(value):
    """Invert real Triton scalar/pointer types without stringifying objects."""
    if type(value).__module__ != "triton.language.core":
        return None
    from triton.language import core as tl, str_to_ty
    if type(value) is tl.dtype:
        code = value.mangle()
    elif type(value) is tl.pointer_type:
        # The generated signature syntax represents address-space one only.
        # mangle() omits const, so reconstruct the explicit *k spelling.
        if value.address_space != 1 or type(value.const) is not bool:
            raise TypeError("Unsupported Triton pointer address space/const qualifier")
        element = _triton_type_code(value.element_ty)
        if element is None:
            raise TypeError("Unsupported Triton pointer element type")
        code = "*" + ("k" if value.const else "") + element
    else:
        return None
    try:
        restored = str_to_ty(code, None)
    except (KeyError, ValueError, IndexError) as error:
        raise TypeError("Unsupported Triton scalar/pointer type") from error
    if type(restored) is not type(value) or restored != value:
        raise TypeError("Triton type signature roundtrip changed its type")
    return code


def _signature_encoded(signature):
    def argument(value):
        code = _triton_type_code(value)
        if code is not None:
            return code
        if type(value).__module__ == "triton.language.core":
            from triton.language import core as tl
            if type(value) is tl.constexpr_type and value.value is None:
                return "constexpr"
        return _encoded(value)
    return _encoded({name: argument(value) for name, value in signature.items()})


def _encoded(value):
    # Preserve tuple keys in Triton's divisibility specialization metadata.
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _encoded(v) for k, v in value.items()}
        pairs = [[_encoded(k), _encoded(v)] for k, v in value.items()]
        pairs.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return {"__mapping__": pairs}
    if isinstance(value, tuple):
        return {"__tuple__": [_encoded(item) for item in value]}
    if isinstance(value, list):
        return [_encoded(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    code = _triton_type_code(value)
    if code is not None:
        # Constants are typed values, unlike their signature declarations.
        return {"__triton_type__": code}
    raise TypeError(f"Unsupported Triton metadata value: {type(value).__name__}")


def _ast_value(node):
    if isinstance(node, ast.AST):
        fields = {}
        for name in node._fields:
            value = getattr(node, name, None)
            # Empty type_params is absent in Python 3.10/3.11. It has no effect
            # on these ordinary generated functions and must not vary the key.
            if name == "type_params" and not value:
                continue
            fields[name] = _ast_value(value)
        return {"node": type(node).__name__, **fields}
    if isinstance(node, list):
        return [_ast_value(item) for item in node]
    return node


def canonical_function_sha256(source: str, kernel_name: str | None = None) -> str:
    """Hash exact math/args AST, ignoring comments, name, decorators and paths."""
    tree = ast.parse(textwrap.dedent(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if kernel_name is not None:
        functions = [node for node in functions if node.name == kernel_name]
    if len(functions) != 1:
        raise ValueError("Expected one Triton function or an exact kernel_name")
    node = copy.deepcopy(functions[0])
    node.name = "kernel"
    node.decorator_list = []
    return _json_hash(_ast_value(node))


def descriptor(source: str, size_hints, triton_meta, *, kernel_name=None) -> dict:
    """Construct the public exact source/shape/compiler matching boundary."""
    device = triton_meta.get("device")
    if isinstance(device, dict):
        arch = {field: device.get(field) for field in ("type", "cc")}
    else:
        arch = {field: getattr(device, field, None) for field in ("type", "cc")}
    return {
        "function_sha256": canonical_function_sha256(source, kernel_name),
        "size_hints": _encoded(size_hints),
        "signature": _signature_encoded(triton_meta.get("signature", {})),
        "constants": _encoded(triton_meta.get("constants", {})),
        "specialization": _encoded(triton_meta.get("configs", [])),
        "compiler_flags": {name: triton_meta.get(name, default) for name, default in (
            ("enable_fp_fusion", True), ("disable_ftz", False),
            ("launch_pdl", False), ("native_matmul", False))},
        "architecture": arch,
    }


def recipe_from_source(source: str, kernel_name: str, selected_fields: dict,
                       binding_id: str) -> dict:
    """Build an inspectable recipe from generated source without executing it.

    This pure-CPU helper is for maintainers updating the versioned policy. It
    reads no cache and is not used by an installed model's startup path.
    """
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == kernel_name)
    decorator = next(node for node in function.decorator_list
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                     and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == "triton_heuristics")
    keywords = {item.arg: item.value for item in decorator.keywords}
    meta_node = keywords["triton_meta"]
    meta = {}
    for key, value in zip(meta_node.keys, meta_node.values):
        key = ast.literal_eval(key)
        if key == "device":
            meta[key] = {item.arg: ast.literal_eval(item.value) for item in value.keywords}
        else:
            meta[key] = ast.literal_eval(value)
    boundary = descriptor(source, ast.literal_eval(keywords["size_hints"]),
                          meta, kernel_name=kernel_name)
    imeta_node = keywords["inductor_meta"]
    site = next(ast.literal_eval(value) for key, value in zip(imeta_node.keys, imeta_node.values)
                if ast.literal_eval(key) == "kernel_name")
    return {"id": binding_id, "kernel_alias": kernel_name, "kernel_site": site,
            "heuristic": decorator.func.attr, "boundary": boundary,
            "semantic_key": _json_hash(boundary), "selected_fields": dict(selected_fields)}


def _policy_mode() -> str:
    mode = os.environ.get("VLLM_MACH_MXFP8_MODE", "production")
    rows = os.environ.get("VLLM_MACH_MXFP8_QUALITY_ROWS")
    if mode == "production" and rows is None:
        return "production"
    if mode == "quality" and rows in ("32", "64"):
        return "quality" + rows
    raise RuntimeError("Compile policy requires production or accepted quality32/quality64 mode")


def load_policy() -> dict:
    global _POLICY, _POLICY_MODE, _ACTIVE_POLICY_FILE
    mode = _policy_mode()
    if _POLICY_MODE is not None and _POLICY_MODE != mode:
        raise RuntimeError("Champion compile-choice mode changed after policy loading")
    if _POLICY is None:
        path = (_POLICY_FILE if mode == "production" else
                _POLICY_FILE.with_name(f"compile_choices_{mode}.json"))
        policy = json.loads(path.read_text())
        if mode != "production" and policy.get("profile_mode") != mode:
            raise RuntimeError("Quality compile-choice policy mode differs")
        expected_count = 22 if mode == "quality64" else 23
        if policy.get("format_version") != 1 or len(policy.get("bindings", [])) != expected_count:
            raise RuntimeError("Invalid champion compile-choice policy format/count")
        seen = set()
        for row in policy["bindings"]:
            if row["id"] in seen or row["semantic_key"] != _json_hash(row["boundary"]):
                raise RuntimeError("Invalid champion compile-choice identity")
            seen.add(row["id"])
            fields = row["selected_fields"]
            if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                       for name, value in fields.items() if name != "found_by_coordesc"):
                raise RuntimeError("Invalid champion compile-choice launch fields")
            if fields.get("found_by_coordesc") is not False:
                raise RuntimeError("Champion recipe cannot claim a coordinate-descent search")
        _POLICY = policy
        _POLICY_MODE = mode
        _ACTIVE_POLICY_FILE = path
    return _POLICY


def select_recipe(source: str, size_hints, triton_meta, *, kernel_alias: str,
                  kernel_site: str | None = None) -> dict | None:
    """Select a fixed recipe; leave unrelated kernels on native heuristics."""
    policy = load_policy()
    # Completely unrelated generated kernels retain native behavior, even if
    # their metadata has types outside this versioned recipe's public schema.
    # Known aliases or equivalent ASTs at a bound geometry still go through
    # the full descriptor and every source/metadata/config check below.
    function_sha = canonical_function_sha256(source)
    shape = _encoded(size_hints)
    device = triton_meta.get("device")
    arch = ({field: device.get(field) for field in ("type", "cc")} if isinstance(device, dict)
            else {field: getattr(device, field, None) for field in ("type", "cc")})
    candidates = [row for row in policy["bindings"]
                  if row["boundary"]["size_hints"] == shape
                  and row["boundary"]["architecture"] == arch
                  and (row["kernel_alias"] == kernel_alias
                       or row["boundary"]["function_sha256"] == function_sha)]
    if not candidates:
        key = _json_hash({"function_sha256": function_sha, "size_hints": shape, "architecture": arch})
        with _LOCK:
            _UNKNOWN_HITS[key] += 1
        return None
    boundary = descriptor(source, size_hints, triton_meta)
    key = _json_hash(boundary)
    group = [row for row in policy["bindings"] if row["semantic_key"] == key]
    if not group:
        changed = sorted({field for row in candidates for field in boundary
                          if row["boundary"][field] != boundary[field]})
        mismatch = {"kernel_alias": kernel_alias, "kernel_site": kernel_site,
                    "semantic_key": key, "changed_fields": changed,
                    "expected_bindings": [row["id"] for row in candidates]}
        with _LOCK:
            _MISMATCHES.append(mismatch)
        raise CompileChoiceMismatch(f"Champion Triton boundary changed: {mismatch}")
    choices = {_json_hash(row["selected_fields"]) for row in group}
    exact = [row for row in group if row["kernel_alias"] == kernel_alias
             and (kernel_site is None or row["kernel_site"] == kernel_site)]
    if len(choices) > 1 and len(exact) != 1:
        mismatch = {"kernel_alias": kernel_alias, "kernel_site": kernel_site,
                    "semantic_key": key, "reason": "ambiguous site discriminator",
                    "expected_bindings": [row["id"] for row in group]}
        with _LOCK:
            _MISMATCHES.append(mismatch)
        raise CompileChoiceMismatch(f"Champion Triton choice needs its exact site: {mismatch}")
    row = exact[0] if exact else group[0]
    with _LOCK:
        _SEMANTIC_HITS[key] += 1
        # Equivalent aliases may safely use one choice, but do not invent
        # evidence that all 23 historical bindings were constructed.
        if exact or len(group) == 1:
            _BINDING_HITS[row["id"]] += 1
    return row


def _make_config(row, Config):
    fields = row["selected_fields"]
    kwargs = {name: value for name, value in fields.items()
              if name not in ("num_warps", "num_stages", "found_by_coordesc")}
    return Config(kwargs, num_warps=fields["num_warps"], num_stages=fields["num_stages"])


def _config_fields(config) -> dict:
    return {**config.kwargs, "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "found_by_coordesc": getattr(config, "found_by_coordesc", False)}


def _require_no_compile_pool(async_compile) -> None:
    pool = getattr(async_compile.AsyncCompile, "process_pool", None)
    info = getattr(pool, "cache_info", None)
    if info is None:
        raise RuntimeError("Unsupported Torch compile-pool ABI; cannot establish synchronous compilation")
    if info().currsize or getattr(async_compile, "_pool_set", ()):
        raise RuntimeError("Champion compile policy must install before a compile process pool exists")


def install() -> bool:
    """Install in parent/GPU worker, before compilation; forbid subprocess pools."""
    global _INSTALLED, _ORIGINAL_INIT
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    import torch
    from torch._inductor import config
    config.compile_threads = 1
    from torch._inductor import async_compile
    from torch._inductor.runtime import triton_heuristics
    from triton import Config

    policy = load_policy()
    if not torch.__version__.split("+", 1)[0].startswith(policy["torch_version_prefix"]):
        raise RuntimeError(f"Champion compile policy requires {policy['torch_version_prefix']}; found {torch.__version__}")
    with _LOCK:
        _require_no_compile_pool(async_compile)
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        config.compile_threads = 1
        if _INSTALLED:
            return False
        cls = triton_heuristics.CachingAutotuner
        original = cls.__init__
        # Inspect runtime source before applying a narrow compatibility hook.
        # No code generation, reduction body, or PyTorch source file is edited.
        actual = canonical_function_sha256(pyinspect.getsource(original))
        if actual != policy["autotuner_init_sha256"]:
            raise RuntimeError("Torch CachingAutotuner constructor differs from the qualified policy ABI")
        signature = pyinspect.signature(original)
        required = {"self", "fn", "triton_meta", "configs", "size_hints", "inductor_meta"}
        if not required.issubset(signature.parameters):
            raise RuntimeError("Unsupported Torch CachingAutotuner signature")

        @wraps(original)
        def initialize(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            fn = bound.arguments["fn"]
            imeta = bound.arguments.get("inductor_meta") or {}
            row = select_recipe(fn.src, bound.arguments["size_hints"],
                                bound.arguments["triton_meta"],
                                kernel_alias=fn.__name__, kernel_site=imeta.get("kernel_name"))
            if row is None:
                return original(*bound.args, **bound.kwargs)
            selected = _make_config(row, Config)
            bound.arguments["configs"] = [selected]
            # A cold build must not mutate the reviewed launch after choosing
            # it. These per-matched-op tuner controls leave unknown ops intact.
            bound.arguments["inductor_meta"] = {
                **imeta, "dynamic_scale_rblock": False,
                "coordinate_descent_tuning": False, "incremental_autotune": False,
            }
            result = original(*bound.args, **bound.kwargs)
            # Torch's own name/hash lookup can overwrite configs in __init__.
            # The full source/meta/site match above takes final precedence.
            self.configs = [selected]
            with _LOCK:
                _AUTOTUNERS.append((weakref.ref(self), row))
            return result

        cls.__init__ = initialize
        _ORIGINAL_INIT = original
        _INSTALLED = True
        return True


def inspect(worker=None) -> dict[str, Any]:
    """Report construction coverage; it is not kernel/graph replay coverage."""
    del worker
    policy = load_policy()
    with _LOCK:
        return {"installed": _INSTALLED, "pid": os.getpid(),
                "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
                "compile_process_mode": "synchronous; no compiler subprocess",
                "policy_mode": _POLICY_MODE or _policy_mode(),
                "policy_sha256": hashlib.sha256(_ACTIVE_POLICY_FILE.read_bytes()).hexdigest(),
                "required_bindings": len(policy["bindings"]),
                "seen_bindings": sorted(_BINDING_HITS), "binding_hits": dict(_BINDING_HITS),
                "missing_bindings": [row["id"] for row in policy["bindings"]
                                     if not _BINDING_HITS[row["id"]]],
                "semantic_hits": dict(_SEMANTIC_HITS),
                "unknown_semantic_key_count": len(_UNKNOWN_HITS),
                "unknown_constructions": sum(_UNKNOWN_HITS.values()),
                "source_mismatches": list(_MISMATCHES),
                "canonical_conflicts": policy["canonical_conflicts"],
                "coverage_semantics": "CachingAutotuner construction, excludes kernel execution/graph replay"}


def verify_coverage(worker=None, *, require_complete=True) -> dict[str, Any]:
    """Check actual launch plans and all sites of the selected run recipe.

    Production and accepted quality modes require complete coverage. Explicit
    partial checks are diagnostic only and still validate every observed Config.
    """
    receipt = inspect(worker)
    if not _INSTALLED or (require_complete and receipt["missing_bindings"]) or receipt["source_mismatches"]:
        raise CompileChoiceMismatch(f"Champion compile-choice coverage incomplete: {receipt}")
    from torch._inductor import async_compile, config

    if config.compile_threads != 1 or os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") != "1":
        raise CompileChoiceMismatch("Champion synchronous compile-thread configuration changed")
    _require_no_compile_pool(async_compile)
    deviations = []
    with _LOCK:
        for reference, row in _AUTOTUNERS:
            tuner = reference()
            if tuner is None:
                continue
            # Precompile consumes configs. Compiled kernels then retain their
            # launch Config in compile_results; inspect both lifecycle states.
            configs = getattr(tuner, "configs", None)
            if configs is None:
                configs = [result.config for result in getattr(tuner, "compile_results", [])]
            if len(configs) != 1 or _config_fields(configs[0]) != row["selected_fields"]:
                deviations.append(row["id"])
    if deviations:
        raise CompileChoiceMismatch(f"Champion launch configuration changed: {sorted(set(deviations))}")
    return receipt


__all__ = ["CompileChoiceMismatch", "canonical_function_sha256", "descriptor",
           "recipe_from_source", "select_recipe", "install", "inspect", "verify_coverage"]
