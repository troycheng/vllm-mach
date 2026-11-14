# SPDX-License-Identifier: Apache-2.0
"""Runner-owned immutable logit geometry for the selected 2B MXFP8 profile."""
from __future__ import annotations

import __future__
import ast
import functools
import hashlib
import inspect
import textwrap
import types

# vLLM 0.33's complete prepare_inputs function, including the general branch.
SOURCE_SHA256 = "c2d1f6d30cfd76b11cb89806953dd94601c1359cc215b5ef007d143473fcd1fa"
_BRANCH = "not draft_tokens and self.model_state.num_new_sampled_tokens_per_step == 1"
_REPLACEMENT = """total_num_draft_tokens = 0
total_num_logits = num_reqs
cu_num_logits_np, cu_num_logits, expanded_local_pos = self._mach_simple_logit_geometry[num_reqs]
expanded_idx_mapping = idx_mapping
"""


def _transform(original):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        return None
    tree = ast.parse(source)
    branches = [node for node in ast.walk(tree)
                if isinstance(node, ast.If) and ast.unparse(node.test) == _BRANCH]
    if len(branches) != 1:
        return None
    branches[0].body = ast.parse(_REPLACEMENT).body
    namespace = {}
    flags = original.__code__.co_flags & __future__.annotations.compiler_flag
    exec(compile(ast.fix_missing_locations(tree), "<mach-input-geometry>", "exec",
                 flags=flags, dont_inherit=True), original.__globals__, namespace)
    candidate = namespace[original.__name__]
    candidate.__defaults__ = original.__defaults__
    candidate.__kwdefaults__ = original.__kwdefaults__
    functools.update_wrapper(candidate, original)
    return candidate


def install(worker):
    """Allocate before the existing startup synchronization; leave other runners alone."""
    runner = getattr(worker, "model_runner", None)
    if runner is None or hasattr(runner, "_mach_simple_logit_geometry"):
        return
    cfg = worker.vllm_config
    parallel = cfg.parallel_config
    if (type(runner).__module__ != "vllm.v1.worker.gpu.model_runner"
            or cfg.speculative_config is not None
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.enable_dbo
            or cfg.cache_config.kv_sharing_fast_prefill):
        return
    original = runner.prepare_inputs
    candidate = _transform(original.__func__)
    if candidate is None:
        return
    import numpy as np
    import torch

    n = runner.max_num_reqs
    with torch.inference_mode():
        cu_cpu = np.arange(n + 1, dtype=np.int32)
        cu_gpu = torch.arange(n + 1, dtype=torch.int32, device=runner.device)
        local_zero = torch.zeros(n, dtype=torch.int32, device=runner.device)
        runner._mach_simple_logit_geometry = tuple(
            (cu_cpu[:m + 1], cu_gpu[:m + 1], local_zero[:m])
            for m in range(n + 1)
        )
    runner.prepare_inputs = types.MethodType(candidate, runner)


def inspect_worker(worker):
    runner = worker.model_runner
    geometry = getattr(runner, "_mach_simple_logit_geometry", None)
    if geometry is None:
        return {"installed": False}
    return {"installed": True,
            "max_num_reqs": runner.max_num_reqs,
            "original_function_sha256": SOURCE_SHA256 if geometry is not None else None,
            "ownership": "runner lifetime; immutable prefixes; dynamic idx_mapping unchanged"}
