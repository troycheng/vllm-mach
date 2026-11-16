# SPDX-License-Identifier: Apache-2.0
"""Keep the 4B recipe's two-dimensional GDN norm frontend.

The shared runtime uses the upstream norm frontend for other profiles.
The complete 4B profile validates that runtime before installing this hook.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap

PROFILE = "qwen35-4b-mxfp8-champion-v1"
_INSTALLED = {}
_CURRENT_BODY = """
core_attn_out = self.norm(core_attn_out, z)
output, _ = self.out_proj(core_attn_out.flatten(-2))
return output
"""


def _output_projection(
    self,
    core_attn_out,
    z,
):
    """Use the original 4B norm/quant compilation boundary."""
    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = core_attn_out.flatten(-2)
    output, _ = self.out_proj(core_attn_out)
    return output


def _method_body(method):
    node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    body = node.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def _class():
    if os.environ.get("VLLM_MACH_PROFILE") != PROFILE:
        raise RuntimeError("GDN norm frontend requires the complete 4B MXFP8 profile")
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    return QwenGatedDeltaNetAttention


def install():
    """Install after the profile's runtime-source gate, before model loading."""
    cls = _class()
    current = vars(cls).get("_output_projection")
    if cls in _INSTALLED:
        if current is not _output_projection:
            raise RuntimeError("4B GDN norm frontend changed after installation")
        return inspect_frontend()
    parameters = inspect.signature(current).parameters if inspect.isfunction(current) else {}
    if (not inspect.isfunction(current) or current.__module__ != cls.__module__
            or current.__qualname__ != cls.__qualname__ + "._output_projection"
            or tuple(parameters) != ("self", "core_attn_out", "z")
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                   or p.default is not inspect.Parameter.empty for p in parameters.values())):
        raise RuntimeError("Unexpected original GDN output-projection method identity")
    try:
        body = _method_body(current)
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError("Cannot inspect original GDN norm frontend") from exc
    if body != ast.dump(ast.parse(_CURRENT_BODY), include_attributes=False):
        raise RuntimeError("Original GDN norm frontend AST changed")
    _INSTALLED[cls] = current
    cls._output_projection = _output_projection
    return inspect_frontend()


def inspect_frontend():
    cls = _class()
    if cls not in _INSTALLED or vars(cls).get("_output_projection") is not _output_projection:
        raise RuntimeError("4B GDN norm frontend is not installed or changed")
    return {"installed": True, "frontend": "legacy_2d", "profile": PROFILE,
            "method": "QwenGatedDeltaNetAttention._output_projection",
            "body_sha256": hashlib.sha256(_method_body(_output_projection).encode()).hexdigest()}
