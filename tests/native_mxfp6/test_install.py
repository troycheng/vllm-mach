"""CPU exact-overlay, cross-distribution staging and rollback contracts."""
from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import shutil

import pytest

from vllm_mach.mxfp6 import install as installer
from vllm_mach.mxfp6.serve import build_command


def write_profile(directory, before, after):
    manifest = {"schema_version": 1, "profile": installer.PROFILE_ID,
                "vllm": "0.33.0", "source": "fixture", "namespaces": {}}
    for namespace, patch_name in installer._PATCHES.items():
        sources = {k: v for k, v in before.items() if k.startswith(namespace + "/")}
        text = "".join("".join(difflib.unified_diff(value.decode().splitlines(True),
            after[name].decode().splitlines(True), fromfile="a/"+name, tofile="b/"+name))
            for name, value in sources.items())
        (directory / patch_name).write_text(text)
        manifest["namespaces"][namespace] = {
            "patch": patch_name, "patch_sha256": installer.sha256(text.encode()),
            "files": {name: {"upstream_sha256": installer.sha256(value),
                             "installed_sha256": installer.sha256(after[name])}
                      for name, value in sources.items()}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def staged(tmp_path):
    data = tmp_path / "profile"; data.mkdir()
    sites = {"vllm": tmp_path / "vllm-site", "flashinfer": tmp_path / "fi-site"}
    before = {"vllm/one.py": b"value = 1\n", "vllm/two.py": b"value = 2\n",
              "vllm/guard.py": b"guard = True\n",
              "flashinfer/comm/one.py": b"value = 3\n",
              "flashinfer/comm/guard.py": b"guard = True\n"}
    after = {name: value.replace(b"value = ", b"value = 10+")
             for name, value in before.items()}
    manifest = write_profile(data, before, after)
    targets = {}
    for name, value in before.items():
        target = sites[name.split("/")[0]] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value); target.chmod(0o640)
        targets[name] = target
    def run(**kwargs):
        return installer.install_profile(sites["vllm"], sites["flashinfer"],
            verify_packages=False, data_directory=data, **kwargs)
    return type("Fixture", (), dict(data=data, sites=sites, before=before, after=after,
               manifest=manifest, targets=targets, run=staticmethod(run)))


def snapshot(case):
    return {name: target.read_bytes() for name, target in case.targets.items()}


def test_preflight_apply_and_idempotent_repeat(staged):
    result = staged.run()
    assert result["previous_state"] == "upstream" and result["dry_run"]
    assert result["changed_files"] and not result["applied"]
    assert snapshot(staged) == staged.before
    assert staged.run(apply=True)["applied"]
    assert snapshot(staged) == staged.after
    assert all(target.stat().st_mode & 0o777 == 0o640 for target in staged.targets.values())
    result = staged.run(apply=True)
    assert result["previous_state"] == "installed"
    assert result["changed_files"] == [] and not result["applied"]


@pytest.mark.parametrize("kind", ["foreign", "partial", "vllm_only", "fi_only", "guard", "legacy"])
def test_reject_unknown_and_half_installed_before_any_write(staged, kind):
    if kind in ("foreign", "legacy"):
        staged.targets["vllm/one.py"].write_bytes(b"# vLLM 0.29 foreign profile\n")
    elif kind == "guard":
        staged.targets["flashinfer/comm/guard.py"].write_bytes(b"guard = False\n")
    else:
        for name, target in staged.targets.items():
            if (kind == "partial" and name == "vllm/one.py"
                or kind == "vllm_only" and name.startswith("vllm/")
                or kind == "fi_only" and name.startswith("flashinfer/")):
                target.write_bytes(staged.after[name])
    before = snapshot(staged)
    with pytest.raises(RuntimeError): staged.run(apply=True)
    assert snapshot(staged) == before


@pytest.mark.parametrize("after_replace", [False, True])
def test_cross_distribution_rollback_on_publication_failure(staged, monkeypatch, after_replace):
    original = installer._replace
    calls = []
    def fail(target, value, mode):
        calls.append(target)
        # Native files were already published before entering FI distribution.
        if target == staged.targets["flashinfer/comm/one.py"] and len(calls) == 3:
            if after_replace: original(target, value, mode)
            raise OSError("injected FI publication failure")
        original(target, value, mode)
    monkeypatch.setattr(installer, "_replace", fail)
    with pytest.raises(OSError, match="injected"): staged.run(apply=True)
    assert snapshot(staged) == staged.before
    for site in staged.sites.values():
        assert not list(site.rglob(".mach-mxfp8-*"))
        assert not (site / ".vllm-mach-mxfp8-install.lock").exists()


@pytest.mark.parametrize("corruption", ["patch_digest", "installed_hash", "syntax", "exact_context"])
def test_complete_stage_validation_precedes_publication(staged, corruption):
    if corruption == "patch_digest":
        (staged.data / "flashinfer-local-ipc.patch").write_text("corrupt\n")
    elif corruption == "installed_hash":
        staged.manifest["namespaces"]["flashinfer"]["files"]["flashinfer/comm/one.py"]["installed_sha256"] = "0"*64
        (staged.data / "manifest.json").write_text(json.dumps(staged.manifest))
    elif corruption == "syntax":
        staged.after["flashinfer/comm/one.py"] = b"invalid python !\n"
        write_profile(staged.data, staged.before, staged.after)
    else:
        name = "vllm/one.py"
        changed = b"# moved context\n" + staged.before[name]
        staged.targets[name].write_bytes(changed)
        staged.manifest["namespaces"]["vllm"]["files"][name]["upstream_sha256"] = installer.sha256(changed)
        (staged.data / "manifest.json").write_text(json.dumps(staged.manifest))
    before = snapshot(staged)
    with pytest.raises((RuntimeError, SyntaxError)): staged.run(apply=True)
    assert snapshot(staged) == before


@pytest.mark.parametrize("name", ["flashinfer/../escape.py", "/flashinfer/escape.py",
                                  "vllm/escape.py", "flashinfer/comm/./escape.py"])
def test_flashinfer_namespace_cannot_bypass_path_checks(staged, name):
    group = staged.manifest["namespaces"]["flashinfer"]
    group["files"][name] = next(iter(group["files"].values()))
    (staged.data / "manifest.json").write_text(json.dumps(staged.manifest))
    with pytest.raises(RuntimeError, match="Invalid runtime source path"):
        staged.run(apply=True)
    assert snapshot(staged) == staged.before


@pytest.mark.parametrize("namespace", ["vllm", "flashinfer"])
def test_lock_and_symlink_rejected_for_either_distribution(staged, monkeypatch, namespace):
    lock = staged.sites[namespace] / ".vllm-mach-mxfp8-install.lock"
    lock.write_text("another installer")
    with pytest.raises(RuntimeError, match="locked"): staged.run(apply=True)
    lock.unlink()
    assert snapshot(staged) == staged.before
    target = next(target for name, target in staged.targets.items() if name.startswith(namespace+"/"))
    backing = target.with_suffix(".original"); target.rename(backing); target.symlink_to(backing)
    with pytest.raises(RuntimeError, match="Symlinked"): staged.run(apply=True)


def test_recheck_detects_changes_between_stage_and_lock(staged, monkeypatch):
    original = installer._commit
    def mutate(*args):
        staged.targets["vllm/guard.py"].write_bytes(b"concurrent = True\n")
        return original(*args)
    monkeypatch.setattr(installer, "_commit", mutate)
    with pytest.raises(RuntimeError, match="changed during staging"): staged.run(apply=True)
    assert staged.targets["vllm/two.py"].read_bytes() == staged.before["vllm/two.py"]
    assert staged.targets["flashinfer/comm/one.py"].read_bytes() == staged.before["flashinfer/comm/one.py"]


def test_direct_install_verifies_versions_by_default(staged, monkeypatch):
    called = []
    monkeypatch.setattr(installer, "check_versions", lambda: called.append(True))
    installer.install_profile(staged.sites["vllm"], staged.sites["flashinfer"],
                              data_directory=staged.data)
    assert called == [True]


def test_bundled_profile_has_one_runtime_layer_and_exact_hashes():
    manifest = installer.load_manifest()
    assert set(p.name for p in installer.PROFILE.glob("*.patch")) == {
        "runtime.patch", "flashinfer-local-ipc.patch"}
    assert manifest["vllm"] == "0.33.0"
    assert "moe_source" not in manifest
    assert len(manifest["namespaces"]["flashinfer"]["files"]) == 2
    assert "vllm/model_executor/layers/quantization/quark/quark_moe.py" in manifest["namespaces"]["vllm"]["files"]


def test_actual_frozen_runtime_install_without_cuda_imports(tmp_path):
    # CI and offline migration audits provide pinned pristine sources explicitly;
    # no installed CUDA package or network download is required by this test.
    source = os.getenv("VLLM_MACH_TEST_UPSTREAM_SOURCE")
    fi_source = os.getenv("VLLM_MACH_TEST_FLASHINFER_SOURCE")
    if not source or not fi_source:
        pytest.skip("Set frozen vLLM/FlashInfer source roots for actual-source installation")
    manifest = installer.load_manifest()
    sites = {"vllm": tmp_path/"vllm-site", "flashinfer": tmp_path/"fi-site"}
    for namespace, root in (("vllm",Path(source)),("flashinfer",Path(fi_source))):
        for name in manifest["namespaces"][namespace]["files"]:
            target=sites[namespace]/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/name,target)
    before={str(p):p.read_bytes() for site in sites.values() for p in site.rglob("*.py")}
    result=installer.install_profile(sites["vllm"],sites["flashinfer"],verify_packages=False)
    assert result["changed_files"] and not result["packages_verified"]
    assert before=={str(p):p.read_bytes() for site in sites.values() for p in site.rglob("*.py")}
    result=installer.install_profile(sites["vllm"],sites["flashinfer"],apply=True,verify_packages=False)
    assert result["applied"]
    for group in manifest["namespaces"].values():
        for name,entry in group["files"].items():
            assert result["result_sha256"][name]==entry["installed_sha256"]
    assert not installer.install_profile(sites["vllm"],sites["flashinfer"],apply=True,
                                        verify_packages=False)["changed_files"]


def test_launcher_resets_stale_optimization_flags(monkeypatch):
    monkeypatch.setenv("VLLM_HYBRID_NVFP4_LM_HEAD", "1")
    monkeypatch.setenv("VLLM_MACH_LM_HEAD_SAMPLING", "1")
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
    assert env["VLLM_MACH_LM_HEAD_SAMPLING"] == "0"
    assert env["VLLM_QWEN3_5_FP16_SSM"] == "0"
    assert env["VLLM_PLUGINS"] == "mach"
    assert command[-2:] == ["--port", "8123"]
    assert "--mamba-ssm-cache-dtype" not in command


@pytest.mark.parametrize("enabled", [None, True, False])
def test_launcher_hybrid_switch_controls_greedy_and_sampling(enabled):
    args = argparse.Namespace(
        model="model", fp16_ssm=False, lossless_prefill=None,
        owner_prefill=None, verify_prefill=False,
    )
    if enabled is not None:
        args.nvfp4_lm_head = enabled
    _, env = build_command(args, [])
    assert env["VLLM_MACH_LM_HEAD_SAMPLING"] == ("0" if enabled is False else "1")
    assert env["VLLM_HYBRID_NVFP4_LM_HEAD"] == env["VLLM_MACH_LM_HEAD_SAMPLING"]


def test_plugin_registers_native_without_exl3():
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("Full plugin import requires the installed target vLLM stack")
    import subprocess
    import sys

    # This test exercises native registration, independently of the profile used
    # to launch the surrounding suite (which may be the MXFP8 overlay).
    env = dict(os.environ)
    for name in ("VLLM_MACH_PROFILE", "VLLM_MACH_NATIVE_MXFP8", "VLLM_MACH_LM_HEAD_SAMPLING"):
        env.pop(name, None)
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
        env=env,
        check=True,
    )


def test_dense_prefill_defaults_and_explicit_opt_out():
    args = argparse.Namespace(
        model="dense",
        fp16_ssm=False,
        lossless_prefill=None,
        owner_prefill=None,
        nvfp4_lm_head=False,
        verify_prefill=False,
    )
    _, env = build_command(args, [])
    assert env["VLLM_SM120_LOSSLESS_PREFILL"] == "1"
    assert env["VLLM_SM120_OWNER_PREFILL"] == "1"
    args.lossless_prefill = args.owner_prefill = False
    _, env = build_command(args, [])
    assert env["VLLM_SM120_LOSSLESS_PREFILL"] == "0"
    assert env["VLLM_SM120_OWNER_PREFILL"] == "0"
