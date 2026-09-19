#!/usr/bin/env python3
"""Rebuild the pinned MXFP6 revision against the image's PyTorch ABI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

from install import patch_file

REVISION = "cd4e964c391fcb8aaf1a27d28a63d778e3a38ece"  # exact TP2 producers; package version 0.2.1
CUTLASS = "e6233cbac5d7c7a865c19c91cd684ceece19513c"
WORK = Path("/opt/mach-build")


def archive(repo, revision, parent):
    parent.mkdir(parents=True, exist_ok=True)
    target = WORK / (repo.split("/")[-1] + ".tar.gz")
    urllib.request.urlretrieve(f"https://codeload.github.com/{repo}/tar.gz/{revision}", target)
    with tarfile.open(target) as stream:
        stream.extractall(parent, filter="data")
    return parent / (repo.split("/")[-1] + "-" + revision)


def main():
    source = archive("Nekofish-L/mxfp6_sm120", REVISION, WORK)
    cutlass = archive("NVIDIA/cutlass", CUTLASS, WORK / "cutlass-source")
    for name in ("0001-sm120-mxfp6-small-tile-runtime.patch", "0003-sm120-streamk-persistent-workspace.patch"):
        patch_file(cutlass, source / "patches/cutlass" / name)
    env = dict(os.environ)
    cuda = Path(env["CUDA_HOME"])
    env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
    env["CUDACXX"] = str(cuda / "bin/nvcc")
    env["CMAKE_ARGS"] = f"-DCUTLASS_DIR={cutlass} -DCMAKE_CUDA_COMPILER={cuda / 'bin/nvcc'}"
    wheels = WORK / "wheels"
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    str(source), "-w", str(wheels)], env=env, check=True)
    wheel, = wheels.glob("mxfp6_sm120-0.2.1-*.whl")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
    # This needs no GPU, but catches the dispatcher ABI mismatch of the PyPI wheel.
    subprocess.run([sys.executable, "-c", "import torch, mxfp6; mxfp6.load_library(); "
                    "assert callable(mxfp6.gemm_from_gdn); "
                    "assert hasattr(torch.ops.mxfp6, 'gemm_from_swiglu'); "
                    "assert hasattr(torch.ops.mxfp6, 'gemm_w6a8_pdl')"], check=True)
    (WORK / "mxfp6-source.json").write_text(json.dumps({"revision": REVISION, "cutlass": CUTLASS}) + "\n")


if __name__ == "__main__":
    main()
