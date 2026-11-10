# SPDX-License-Identifier: Apache-2.0
"""Rebuild stock RMS launch choices in the independent block-FP8 profile.

Only exact source/shape/compiler descriptors receive a fixed Config. No binary
cache is consumed and unrelated kernels retain their native heuristics. Install
in parent and worker before generated modules, then verify actual launch plans.
The MXFP8 helpers below are pure; its policy and installation are never invoked.
"""
from __future__ import annotations

from collections import Counter
from functools import wraps
import hashlib
import inspect as pyinspect
import json
import os
import re
from pathlib import Path
import threading

from ..mxfp8.compile_choices import (
    _config_fields, _json_hash, _make_config, _require_no_compile_pool, _require_compiler_versions,
    canonical_function_sha256, descriptor, recipe_from_source,
)

_DATA = Path(__file__).parent / "data"
_ENV = "VLLM_MACH_FP8_COMPILE_MODE"
_MODES = {"quality4", "quality32", "quality64", "production"}
_LOCK = threading.RLock()
_POLICY = None
_MODE = None
_INSTALLED = False
_HITS = Counter()
_UNKNOWN = 0
_MISMATCHES = []
_AUTOTUNERS = []


class CompileChoiceMismatch(RuntimeError):
    """A stock RMS boundary or its actual launch configuration drifted."""


def load_policy(mode=None):
    global _POLICY, _MODE
    mode = mode if mode is not None else os.environ.get(_ENV, "") or _MODE
    if mode not in _MODES:
        raise RuntimeError(f"Block-FP8 compile mode must be one of {sorted(_MODES)}")
    with _LOCK:
        if _MODE is not None and _MODE != mode:
            raise RuntimeError("Block-FP8 compile mode changed after loading")
        if _POLICY is None:
            policy = json.loads((_DATA / f"compile_choices_{mode}.json").read_text())
            if policy.get("format_version") != 2 or policy.get("profile_mode") != mode:
                raise RuntimeError("Invalid block-FP8 compile recipe mode/version")
            if policy.get("binding_semantics") != "exact_boundary_and_selected_config":
                raise RuntimeError("Invalid block-FP8 coverage semantics")
            ids = set()
            semantics = {}
            for row in policy["bindings"]:
                if row["id"] in ids:
                    raise RuntimeError("Duplicate block-FP8 compile binding")
                ids.add(row["id"])
                for variant in row["variants"]:
                    if variant["semantic_key"] != _json_hash(variant["boundary"]):
                        raise RuntimeError("Invalid block-FP8 source descriptor")
                    key = variant["semantic_key"]
                    if key in semantics:
                        raise RuntimeError("Duplicate or conflicting block-FP8 semantic group")
                    semantics[key] = row["selected_fields"]
                fields = row["selected_fields"]
                if fields.get("found_by_coordesc") is not False or not all(
                    type(value) is int and value > 0 for key, value in fields.items()
                    if key != "found_by_coordesc"
                ) or not {"num_warps", "num_stages"}.issubset(fields):
                    raise RuntimeError("Invalid block-FP8 launch fields")
            if not ids:
                raise RuntimeError("Empty block-FP8 recipe")
            _MODE, _POLICY = mode, policy
        return _POLICY


def select_recipe(source, size_hints, triton_meta, *, kernel_alias, kernel_site=None):
    """Exact-match bound RMS sites; refuse drift and leave other ops alone."""
    global _UNKNOWN
    policy = load_policy()
    function_sha = canonical_function_sha256(source)
    boundary = descriptor(source, size_hints, triton_meta)
    key = _json_hash(boundary)
    # Generated suffixes are provenance, not part of a numerical contract.
    # Exact descriptors include the complete arithmetic AST, shape, dtypes,
    # architecture, constants and floating/compiler flags.
    exact = [(row, variant) for row in policy["bindings"] for variant in row["variants"]
             if variant["semantic_key"] == key]
    rms_family = any(isinstance(name, str) and re.fullmatch(
        r"triton_red_fused__to_copy_add_fused_add_rms_norm(?:_per_token_group_fp8_quant_permute)?_\d+", name)
        for name in (kernel_alias, kernel_site))
    related = [row for row in policy["bindings"] if rms_family
               or kernel_alias in row["kernel_aliases"] or kernel_site in row["kernel_sites"]
               or any(v["boundary"]["function_sha256"] == function_sha
                      for v in row["variants"])]
    if not related and not exact:
        with _LOCK:
            _UNKNOWN += 1
        return None
    if len(exact) != 1:
        failure = {"kernel_alias": kernel_alias, "kernel_site": kernel_site,
                   "semantic_key": key, "expected_bindings": [r["id"] for r in related],
                   "changed_fields": sorted({field for row in related
                       for variant in row["variants"] for field in boundary
                       if variant["boundary"][field] != boundary[field]})}
        with _LOCK:
            _MISMATCHES.append(failure)
        raise CompileChoiceMismatch(f"Block-FP8 RMS boundary changed: {failure}")
    row, variant = exact[0]
    with _LOCK:
        _HITS[row["id"]] += 1
    return {**row, "matched_variant": variant["name"], "matched_semantic_key": key,
            "observed_alias": kernel_alias, "observed_site": kernel_site}


def install(mode=None):
    """Install before compilation, in both parent and worker; default is off."""
    global _INSTALLED
    requested = mode if mode is not None else os.environ.get(_ENV, "")
    if not requested:
        if _INSTALLED:
            raise RuntimeError("Cannot disable an installed block-FP8 compile policy")
        return False
    policy = load_policy(requested)
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    import torch
    from torch._inductor import async_compile, config
    from torch._inductor.runtime import triton_heuristics
    import triton
    from triton import Config

    _require_compiler_versions(policy, torch.__version__, triton.__version__)
    with _LOCK:
        _require_no_compile_pool(async_compile)
        config.compile_threads = 1
        if _INSTALLED:
            return False
        cls = triton_heuristics.CachingAutotuner
        original = cls.__init__
        if canonical_function_sha256(pyinspect.getsource(original)) != policy["autotuner_init_sha256"]:
            raise RuntimeError("Torch autotuner ABI differs or another compile policy is installed")
        signature = pyinspect.signature(original)
        if not {"self", "fn", "triton_meta", "configs", "size_hints", "inductor_meta"}.issubset(signature.parameters):
            raise RuntimeError("Unsupported Torch autotuner signature")

        @wraps(original)
        def initialize(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            fn = bound.arguments["fn"]
            imeta = bound.arguments.get("inductor_meta") or {}
            row = select_recipe(fn.src, bound.arguments["size_hints"],
                                bound.arguments["triton_meta"], kernel_alias=fn.__name__,
                                kernel_site=imeta.get("kernel_name"))
            if row is None:
                return original(*bound.args, **bound.kwargs)
            selected = _make_config(row, Config)
            bound.arguments["configs"] = [selected]
            bound.arguments["inductor_meta"] = {
                **imeta, "dynamic_scale_rblock": False,
                "coordinate_descent_tuning": False, "incremental_autotune": False,
            }
            result = original(*bound.args, **bound.kwargs)
            # Torch's cache lookup may replace the incoming configs in init.
            self.configs = [selected]
            with _LOCK:
                _AUTOTUNERS.append((self, row))
            return result

        cls.__init__ = initialize
        _INSTALLED = True
        return True


def _actual_choices():
    choices = []
    for tuner, row in _AUTOTUNERS:
        configs = getattr(tuner, "configs", None)
        state = "constructed"
        if configs is None:
            configs = [r.config for r in getattr(tuner, "compile_results", [])]
            state = "compiled"
        choices.append({"binding": row["id"], "variant": row["matched_variant"],
                        "semantic_key": row["matched_semantic_key"], "state": state,
                        "observed_alias": row["observed_alias"],
                        "observed_site": row["observed_site"],
                        "selected_config": row["selected_fields"],
                        "actual_configs": [_config_fields(c) for c in configs]})
    return choices


def inspect(worker=None):
    """Construction/config evidence, never claimed as GPU execution coverage."""
    del worker
    with _LOCK:
        if _POLICY is None:
            return {"installed": False, "pid": os.getpid(), "policy_mode": None}
        load_policy()
        path = _DATA / f"compile_choices_{_MODE}.json"
        return {"installed": _INSTALLED, "pid": os.getpid(), "policy_mode": _MODE,
                "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
                "required_bindings": len(_POLICY["bindings"]), "binding_hits": dict(_HITS),
                "missing_bindings": [r["id"] for r in _POLICY["bindings"] if not _HITS[r["id"]]],
                "unknown_constructions": _UNKNOWN, "source_mismatches": list(_MISMATCHES),
                "actual_choices": _actual_choices(),
                "coverage_semantics": "exact semantic/config group construction; excludes GPU execution and historical alias counts"}


def verify_coverage(worker=None, *, require_complete=True):
    receipt = inspect(worker)
    if not _INSTALLED or (require_complete and receipt["missing_bindings"]) or receipt["source_mismatches"]:
        raise CompileChoiceMismatch(f"Block-FP8 compile coverage incomplete: {receipt}")
    from torch._inductor import async_compile, config
    if config.compile_threads != 1 or os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") != "1":
        raise CompileChoiceMismatch("Block-FP8 synchronous compile configuration changed")
    _require_no_compile_pool(async_compile)
    for choice in receipt["actual_choices"]:
        if choice["actual_configs"] is None:
            raise CompileChoiceMismatch("Autotuner released before actual config verification")
        if choice["actual_configs"] != [choice["selected_config"]]:
            raise CompileChoiceMismatch(f"Block-FP8 actual launch config changed: {choice}")
    return receipt


__all__ = ["CompileChoiceMismatch", "load_policy", "select_recipe", "install", "inspect",
           "verify_coverage", "canonical_function_sha256", "descriptor", "recipe_from_source"]
