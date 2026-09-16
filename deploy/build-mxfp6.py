#!/usr/bin/env python3
"""Rebuild the pinned MXFP6 release against the image's PyTorch ABI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

from install import patch_file

REVISION = "7c891d07b65ce2f4e5e8e10a6934c1a298755b8d"  # v0.2.1
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
    tp2_patch = Path(__file__).with_name('mxfp6-tp2.patch')
    patch_file(source, tp2_patch)
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
    subprocess.run([sys.executable, "-c", "import torch, mxfp6; mxfp6.load_library()"], check=True)
    (WORK / "mxfp6-source.json").write_text(json.dumps({"revision": REVISION, "cutlass": CUTLASS, "tp2_patch_sha256": hashlib.sha256(tp2_patch.read_bytes()).hexdigest()}) + "\n")


if __name__ == "__main__":
    main()
