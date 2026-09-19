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
    assert env["VLLM_MACH_FUSED_AR_QUANT"] == "1"


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
        if name == "vllm-mach-owner-prefill":
            return "0.1.0a1"
        if name == "vllm-mach-lossless-prefill":
            return "0.1.0a4"
        if name == "vllm-mach-ar-norm":
            return "0.1.0a1"
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
    assert launched[0][2]["VLLM_MACH_FUSED_AR_QUANT"] == "1"


@pytest.mark.parametrize(
    ("fused_ar_quant", "fused_ar_norm", "expected"),
    [(True, True, "1"), (False, True, "0"), (True, False, "0")],
)
def test_dense_fused_ar_quant_switches(
    tmp_path, fused_ar_quant, fused_ar_norm, expected
):
    from vllm_mach.mxfp6.serve import profile_environment

    environment = profile_environment(
        argparse.Namespace(
            model=tmp_path / "dense",
            fp16_ssm=False,
            lossless_prefill=None,
            owner_prefill=None,
            nvfp4_lm_head=False,
            verify_prefill=False,
            fused_ar_quant=fused_ar_quant,
            fused_ar_norm=fused_ar_norm,
        )
    )
    assert environment["VLLM_MACH_FUSED_AR_QUANT"] == expected


def test_moe_disables_dense_fused_ar_quant(tmp_path):
    from vllm_mach.mxfp6.serve import profile_environment

    model = tmp_path / "moe"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3_5_moe"}')
    environment = profile_environment(
        argparse.Namespace(
            model=model,
            fp16_ssm=False,
            lossless_prefill=False,
            owner_prefill=False,
            nvfp4_lm_head=False,
            verify_prefill=False,
            fused_ar_quant=True,
            fused_ar_norm=True,
        )
    )
    assert environment["VLLM_MACH_FUSED_AR_QUANT"] == "0"


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


def test_native_installer_selects_ar_norm(monkeypatch, tmp_path):
    tool = load("deploy/install.py")
    calls = []
    monkeypatch.setattr(tool, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        tool.sys,
        "argv",
        [
            "install.py", "--native-only", "--native-part", "ar_norm",
            "--cuda-home", "/cuda", "--work-dir", str(tmp_path),
        ],
    )
    tool.main()
    wheel_calls = [args for args, _ in calls if "wheel" in args]
    assert len(wheel_calls) == 1
    assert wheel_calls[0][-3] == ROOT / "native" / "ar_norm"
    assert calls[-1][0][1:6] == (
        "-m", "pip", "install", "--no-deps", "--force-reinstall"
    )


def test_native_installer_defaults_include_ar_norm(monkeypatch, tmp_path):
    tool = load("deploy/install.py")
    calls = []
    monkeypatch.setattr(tool, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        tool.sys,
        "argv",
        [
            "install.py", "--native-only", "--cuda-home", "/cuda",
            "--work-dir", str(tmp_path),
        ],
    )
    tool.main()
    wheel_calls = [args for args, _ in calls if "wheel" in args]
    assert [args[-3] for args in wheel_calls] == [
        ROOT / "native" / "lossless_prefill",
        ROOT / "native" / "owner_prefill",
        ROOT / "native" / "ar_norm",
    ]
