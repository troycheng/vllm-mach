"""Installer must never patch an unknown or half-installed runtime."""

import argparse
import importlib.metadata as metadata
import json
import shutil
from pathlib import Path

import pytest

from vllm_mach.mxfp6.install import PROFILE, _patch, install_profile
from vllm_mach.mxfp6.serve import build_command


@pytest.fixture
def pristine(tmp_path):
    manifest = json.loads((PROFILE / "manifest.json").read_text())
    site = Path(metadata.distribution("vllm").locate_file(""))
    flash_site = Path(metadata.distribution("flashinfer-python").locate_file(""))
    for name in manifest["files"] + manifest["moe_source"]["files"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(site / name, target)
    if (
        _patch(tmp_path, "compiled-ar-norm.patch", reverse=True, dry=True).returncode
        == 0
    ):
        assert _patch(tmp_path, "compiled-ar-norm.patch", reverse=True).returncode == 0
    if _patch(tmp_path, "moe-ar-norm.patch", reverse=True, dry=True).returncode == 0:
        assert _patch(tmp_path, "moe-ar-norm.patch", reverse=True).returncode == 0
    if _patch(tmp_path, "runtime.patch", reverse=True, dry=True).returncode == 0:
        assert _patch(tmp_path, "runtime.patch", reverse=True).returncode == 0
    if _patch(tmp_path, "moe.patch", reverse=True, dry=True).returncode == 0:
        assert _patch(tmp_path, "moe.patch", reverse=True).returncode == 0
    for name in ("flashinfer/comm/mnnvl.py", "flashinfer/comm/trtllm_ar.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(flash_site / name, target)
    if (
        _patch(
            tmp_path, "flashinfer-local-ipc.patch", reverse=True, dry=True
        ).returncode
        == 0
    ):
        assert (
            _patch(tmp_path, "flashinfer-local-ipc.patch", reverse=True).returncode == 0
        )
    return tmp_path, manifest


def test_install_preflight_apply_and_repeat(pristine):
    site, manifest = pristine
    result = install_profile(site, site)
    assert result["changed_files"]
    assert not result["applied"]
    assert _patch(site, "runtime.patch", dry=True).returncode == 0
    assert install_profile(site, site, apply=True)["changed_files"]
    assert _patch(site, "moe-ar-norm.patch", reverse=True, dry=True).returncode == 0
    assert not install_profile(site, site, apply=True)["changed_files"]


@pytest.mark.parametrize("kind", ["foreign", "partial", "flashinfer"])
def test_preflight_failure_never_changes_other_files(pristine, kind):
    site, manifest = pristine
    names = list(manifest["files"])
    if kind == "partial":
        original = (site / names[0]).read_bytes()
        assert _patch(site, "runtime.patch").returncode == 0
        (site / names[0]).write_bytes(original)
    elif kind == "foreign":
        (site / names[-1]).write_text("# incompatible source\n")
    else:
        (site / "flashinfer/comm/trtllm_ar.py").write_text("# incompatible runtime\n")
    before = {str(p): p.read_bytes() for p in site.rglob("*.py")}
    with pytest.raises(RuntimeError):
        install_profile(site, site, apply=True)
    assert {str(p): p.read_bytes() for p in site.rglob("*.py")} == before


def test_launcher_resets_stale_optimization_flags(monkeypatch):
    monkeypatch.setenv("VLLM_HYBRID_NVFP4_LM_HEAD", "1")
    args = argparse.Namespace(
        model="model",
        fp16_ssm=False,
        lossless_prefill=False,
        owner_prefill=False,
        nvfp4_lm_head=False,
        verify_prefill=False,
    )
    command, env = build_command(args, ["--port", "8123"])
    assert env["VLLM_HYBRID_NVFP4_LM_HEAD"] == "0"
    assert env["VLLM_QWEN3_5_FP16_SSM"] == "0"
    assert env["VLLM_PLUGINS"] == "mach"
    assert command[-2:] == ["--port", "8123"]
    assert "--mamba-ssm-cache-dtype" not in command


def test_plugin_registers_native_without_exl3():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from vllm_mach.plugin import register
register()
register()
from vllm.model_executor.kernels import linear
from vllm.platforms import PlatformEnum
from vllm_mach.mxfp6.dense import Mxfp6Sm120LinearKernel
assert linear._POSSIBLE_MXFP6_KERNELS[PlatformEnum.CUDA].count(Mxfp6Sm120LinearKernel) == 1
assert not any(m.startswith(('exllama', 'vllm_mach.exl3')) for m in sys.modules)
""",
        ],
        check=True,
    )


def test_existing_dense_profile_upgrade_is_atomic(pristine):
    site, manifest = pristine
    assert _patch(site, "runtime.patch").returncode == 0
    before = {str(p): p.read_bytes() for p in site.rglob("*.py")}
    result = install_profile(site, site)
    assert manifest["moe_source"]["files"][0] in result["changed_files"]
    assert {str(p): p.read_bytes() for p in site.rglob("*.py")} == before
    assert install_profile(site, site, apply=True)["changed_files"]
    assert not install_profile(site, site, apply=True)["changed_files"]


def test_incompatible_moe_source_rejected_without_writes(pristine):
    site, manifest = pristine
    (site / manifest["moe_source"]["files"][0]).write_text(
        "# foreign MoE implementation\n"
    )
    before = {str(p): p.read_bytes() for p in site.rglob("*.py")}
    with pytest.raises(RuntimeError, match="MoE patch failed"):
        install_profile(site, site, apply=True)
    assert {str(p): p.read_bytes() for p in site.rglob("*.py")} == before


def test_existing_moe_profile_upgrade_and_incompatible_fusion_are_atomic(pristine):
    site, _ = pristine
    assert _patch(site, "runtime.patch").returncode == 0
    assert _patch(site, "moe.patch").returncode == 0
    result = install_profile(site, site)
    assert "vllm/model_executor/models/qwen3_5.py" in result["changed_files"]
    install_profile(site, site, apply=True)
    assert not install_profile(site, site, apply=True)["changed_files"]
    target = site / "vllm/model_executor/models/qwen3_5.py"
    target.write_text(
        target.read_text().replace("defer_moe_allreduce(self.mlp)", "foreign(self.mlp)")
    )
    before = {str(p): p.read_bytes() for p in site.rglob("*.py")}
    with pytest.raises(RuntimeError):
        install_profile(site, site, apply=True)
    assert {str(p): p.read_bytes() for p in site.rglob("*.py")} == before


def test_existing_manual_moe_fusion_upgrades_to_compiled_support(pristine):
    site, _ = pristine
    for patch in (
        "runtime.patch",
        "moe.patch",
        "moe-ar-norm.patch",
        "flashinfer-local-ipc.patch",
    ):
        assert _patch(site, patch).returncode == 0
    before = {str(p): p.read_bytes() for p in site.rglob("*.py")}
    result = install_profile(site, site)
    assert set(result["changed_files"]) == {
        "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py",
        "vllm/model_executor/models/qwen3_5.py",
    }
    assert {str(p): p.read_bytes() for p in site.rglob("*.py")} == before
    install_profile(site, site, apply=True)
    assert not install_profile(site, site, apply=True)["changed_files"]
