"""CPU checks for the public launch and model export entry points."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location(
        "deployment_test_module", ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_is_complete():
    from vllm_mach.mxfp6.serve import build_command

    args = argparse.Namespace(
        model="/models/mxfp6",
        fp16_ssm=True,
        owner_prefill=True,
        lossless_prefill=True,
        nvfp4_lm_head=True,
        verify_prefill=False,
    )
    cmd, env = build_command(args, ["--kv-cache-memory-bytes", "8218214400"])
    for flag, value in {
        "--max-num-seqs": "32",
        "--mamba-ssm-cache-dtype": "float16",
        "--kv-cache-memory-bytes": "8218214400",
        "--generation-config": "vllm",
        "--tensor-parallel-size": "2",
    }.items():
        assert cmd[cmd.index(flag) + 1] == value
    assert (
        json.loads(cmd[cmd.index("--compilation-config") + 1])[
            "cudagraph_capture_sizes"
        ][-1]
        == 32
    )
    assert env["VLLM_HYBRID_NVFP4_LM_HEAD"] == "1"
    assert env["VLLM_SM120_OWNER_PREFILL"] == "1"


def test_checkout_launcher_uses_package_entrypoint():
    serve = load("deploy/serve.py")
    from vllm_mach.mxfp6.serve import main

    assert serve.main is main


@pytest.mark.parametrize("b12x_version", [None, "0.0.0"])
def test_nvfp4_launcher_does_not_require_standalone_b12x(
    monkeypatch, tmp_path, b12x_version
):
    from importlib import metadata
    from types import SimpleNamespace

    from vllm_mach.mxfp6 import install, serve

    queried = []

    def version(name):
        queried.append(name)
        if name == "b12x" and b12x_version is not None:
            return b12x_version
        raise metadata.PackageNotFoundError(name)

    launched = []
    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda path: tmp_path),
    )
    monkeypatch.setattr(install, "check_versions", lambda: None)
    monkeypatch.setattr(install, "install_profile", lambda *args: {"changed_files": []})
    monkeypatch.setattr(
        serve.sys,
        "argv",
        ["vllm-mach-serve", "--model", "/models/mxfp6", "--nvfp4-lm-head"],
    )
    monkeypatch.setattr(serve.os, "execvpe", lambda *args: launched.append(args))
    serve.main()
    assert "b12x" not in queried
    assert len(launched) == 1
    assert launched[0][2]["VLLM_HYBRID_NVFP4_LM_HEAD_BACKEND"] == "b12x"


def fixture_model(root, shard="model.safetensors"):
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"fixture weights")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": shard}})
    )


def test_export_and_verify(tmp_path):
    tool = load("tools/export_model_assets.py")
    src, dst = tmp_path / "src", tmp_path / "dst"
    fixture_model(src)
    manifest = tool.export(src, dst, "test/model", "abc")
    assert tool.verify(dst) == manifest
    (dst / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="mismatch"):
        tool.verify(dst)


def test_export_refuses_missing_and_escaping_shards(tmp_path):
    tool = load("tools/export_model_assets.py")
    src, dst = tmp_path / "src", tmp_path / "dst"
    fixture_model(src, "../model.safetensors")
    with pytest.raises(ValueError, match="Unsafe"):
        tool.export(src, dst, "test/model", "abc")
    assert not dst.exists()


def test_export_never_overwrites(tmp_path):
    tool = load("tools/export_model_assets.py")
    src, dst = tmp_path / "src", tmp_path / "dst"
    fixture_model(src)
    dst.mkdir()
    with pytest.raises(FileExistsError):
        tool.export(src, dst, "test/model", "abc")


def test_patch_applies_once_and_rejects_unknown_source(tmp_path):
    import subprocess

    tool = load("deploy/install.py")
    target = tmp_path / "hello.txt"
    target.write_text("original\n")
    patch = tmp_path / "change.patch"
    patch.write_text(
        "--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n-original\n+patched\n"
    )
    tool.patch_file(tmp_path, patch)
    assert target.read_text() == "patched\n"
    tool.patch_file(tmp_path, patch)
    assert target.read_text() == "patched\n"
    target.write_text("unrelated\n")
    with pytest.raises(subprocess.CalledProcessError):
        tool.patch_file(tmp_path, patch)
    assert target.read_text() == "unrelated\n"
