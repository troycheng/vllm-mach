#!/usr/bin/env python3
"""Build Mach native dependencies and apply the vLLM 0.29 source profile."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EXL_REVISION = "0740edc2da569fb99174023c1d2988b1e98cb41e"
VERSIONS = {"vllm": "0.29.0", "torch": "2.13.0", "flashinfer-python": "0.6.18",
            "b12x": "1.3.0", "mxfp6-sm120": "0.2.1", "nvidia-cutlass-dsl": "4.6.2"}


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def check_versions():
    found = {name: metadata.version(name) for name in VERSIONS}
    for name, expected in VERSIONS.items():
        if found[name].split("+")[0] != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {found[name]}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("This binary profile requires Python 3.12")
    return found


def patch_file(target, patch):
    """Apply once, accept the exact already-applied patch, reject other edits."""
    # --force disables patch's automatic reversal heuristic. Without it a
    # reverse dry-run can succeed on a pristine file by ignoring --reverse.
    command = ["patch", "--batch", "--force", "--fuzz=0", "-p1", "-d", str(target), "-i", str(patch)]
    reverse = subprocess.run(command + ["--dry-run", "--reverse"], capture_output=True)
    if reverse.returncode == 0:
        return
    run(*command, "--dry-run", "--forward")
    run(*command, "--forward")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--lossless-cuda-home", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/mach-build"))
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--native-only", action="store_true")
    phases.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--native-part", choices=("exllamav3", "exl3_m32", "exl3_temporal_m24", "lossless_prefill", "owner_prefill"))
    args = parser.parse_args()
    versions = check_versions()
    for home, version in [(args.cuda_home, "13.2"), (args.lossless_cuda_home, "13.0")]:
        output = subprocess.check_output([str(home / "bin/nvcc"), "--version"], text=True)
        if f"release {version}," not in output:
            raise RuntimeError(f"{home}: requires CUDA {version}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.work_dir / f"exllamav3-{EXL_REVISION}"
    archive = args.work_dir / "exllamav3.tar.gz"
    if not args.runtime_only and not source.exists():
        urllib.request.urlretrieve(
            f"https://codeload.github.com/turboderp-org/exllamav3/tar.gz/{EXL_REVISION}", archive)
        with tarfile.open(archive) as stream:
            stream.extractall(args.work_dir, filter="data")
    if not args.runtime_only:
        patch_file(source, ROOT / "profiles/exllamav3-1.5.0/bf16-io.patch")
    environment = dict(os.environ, CUDA_HOME=str(args.cuda_home), TORCH_CUDA_ARCH_LIST="12.0a",
                       EXLLAMA_V3_SOURCE=str(source / "exllamav3/exllamav3_ext"))
    environment["PATH"] = str(args.cuda_home / "bin") + os.pathsep + environment["PATH"]
    wheels = args.work_dir / "wheels"
    wheels.mkdir(exist_ok=True)
    packages = [source, ROOT / "native/exl3_m32", ROOT / "native/exl3_temporal_m24",
                ROOT / "native/lossless_prefill", ROOT / "native/owner_prefill"]
    if args.runtime_only:
        packages = []
    elif args.native_part:
        packages = [source] if args.native_part == "exllamav3" else [ROOT / "native" / args.native_part]
    for package in packages:
        if not (package / "setup.py").exists() and not (package / "pyproject.toml").exists():
            raise RuntimeError(f"Incomplete source checkout: {package}")
        env = dict(environment)
        if package.name == "lossless_prefill":
            env["CUDA_HOME"] = str(args.lossless_cuda_home)
            env["PATH"] = str(args.lossless_cuda_home / "bin") + os.pathsep + environment["PATH"]
        run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
            str(package), "-w", str(wheels), env=env)
    if not args.runtime_only:
        run(sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
            *sorted(wheels.glob("*.whl")))
    if args.native_only:
        return
    run(sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(ROOT))
    site = Path(metadata.distribution("vllm").locate_file(""))
    run(sys.executable, "-m", "pip", "uninstall", "-y", "flashinfer-jit-cache")
    patches = [ROOT / f"profiles/vllm-0.29.0/{p}.patch" for p in
               ("mxfp6-graph-warmup", "runtime", "flashinfer-local-ipc")]
    patches.append(ROOT / "profiles/vllm-0.28.0/flashinfer-mxfp8-packed-layout.patch")
    for patch in patches:
        patch_file(site, patch)
    run(sys.executable, ROOT / "profiles/flashinfer-0.6.18-gdn/install.py", "--apply")
    owner_patch = ROOT / "profiles/vllm-0.29.0/owner-prefill.patch"
    if owner_patch.exists():
        patch_file(site, owner_patch)
    run(sys.executable, "-c", "import torch; import exllamav3_ext as e; "
        "assert callable(e.exl3_mgemm_bf16_io_grouped_had); "
        "from vllm_mach.exl3.fused_allreduce import verify_flashinfer_profile; "
        "verify_flashinfer_profile()")
    receipt = {"versions": versions, "exllamav3_commit": EXL_REVISION,
               "wheels": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels.glob("*.whl")},
               "patches": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in patches + [owner_patch] if p.exists()}}
    (args.work_dir / "installed.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
