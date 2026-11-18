"""Version selection and ABI gates must remain independent of CUDA imports."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm_mach import stack


def test_target_stack_and_exact_torch_build():
    versions = {**stack.BASE_PACKAGES, "torch": stack.TORCH_BUILD}
    assert stack.check_versions(stack.BASE_PACKAGES, lookup=versions.__getitem__) == versions
    for wrong in ("2.14.0", "2.14.0+cpu", "2.14.0+cu128", "2.13.0+cu130"):
        with pytest.raises(RuntimeError, match="torch==2.14.0\\+cu130"):
            stack.require_version("torch", wrong, stack.BASE_PACKAGES["torch"])
    assert stack.BUILD_PACKAGES["b12x"] == "1.5.1"
    assert stack.NATIVE_VERSION == "0.3.1"


def test_plugin_unselected_mismatch_skips_and_explicit_mismatch_rejects():
    from vllm_mach import plugin
    with patch.dict(os.environ, {}, clear=True), patch.object(plugin, "version", return_value="0.29.0"), \
            patch.object(plugin, "register_dense_kernel") as dense:
        plugin.register()
        dense.assert_not_called()
        for env in ({"VLLM_PLUGINS": "mach"}, {"VLLM_MACH_PROFILE": "selected"},
                    {"VLLM_MACH_NATIVE_MXFP8": "1"}):
            with patch.dict(os.environ, env), pytest.raises(RuntimeError, match="Explicit Mach"):
                plugin.register()
        dense.assert_not_called()


def test_explicit_plugin_checks_torch_abi_before_hooks_or_cuda():
    from vllm_mach import plugin
    versions = {**stack.BASE_PACKAGES, "torch": "2.14.0+cpu"}
    with patch.dict(os.environ, {"VLLM_PLUGINS": "mach"}, clear=True), \
            patch.object(plugin, "version", side_effect=versions.__getitem__), \
            patch.object(plugin, "register_dense_kernel") as dense:
        with pytest.raises(RuntimeError, match="torch=="):
            plugin.register()
        dense.assert_not_called()


@pytest.mark.parametrize("has_python_api", [False, True])
def test_compact_moe_plugin_rejects_old_native_before_registering(monkeypatch, has_python_api):
    from vllm_mach import plugin

    versions = {**stack.BASE_PACKAGES, "torch": stack.TORCH_BUILD}
    monkeypatch.setenv("VLLM_MACH_MOE_HIGH_M_COMPACT", "1")
    monkeypatch.setattr(plugin, "version", versions.__getitem__)
    register_dense = Mock()
    monkeypatch.setattr(plugin, "register_dense_kernel", register_dense)
    load_native = Mock()
    native = SimpleNamespace(load_library=load_native)
    if has_python_api:
        native.qwen35_compact_gemm_out = Mock()
    monkeypatch.setitem(sys.modules, "mxfp6", native)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(ops=SimpleNamespace(mxfp6=object())))

    with pytest.raises(RuntimeError, match="requires the native qwen35_compact_gemm_out"):
        plugin.register()
    assert load_native.call_count == int(has_python_api)
    register_dense.assert_not_called()
