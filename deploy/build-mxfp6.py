#!/usr/bin/env python3
"""Rebuild the pinned MXFP6 revision against the image's PyTorch ABI."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request

from install import patch_file

REVISION = "ce0cd05a007504853d2bf3963a183e3c76f3c75b"  # unified MXFP8 decode kernels; official PDL ABI; package version 0.2.1
CUTLASS = "e6233cbac5d7c7a865c19c91cd684ceece19513c"
PATCHES = (
    "0001-sm120-mxfp6-small-tile-runtime.patch",
    "0003-sm120-streamk-persistent-workspace.patch",
    "0004-sm120-single-stage-mainloop.patch",
    "0005-sm120-static-problem-shape.patch",
    "0006-sm120-pdl-release.patch",
)
WORK = Path("/opt/mach-build")
SMOKE = """import torch, mxfp6
from mxfp6 import mxfp8
mxfp6.load_library()
mxfp8.load_library()
assert callable(mxfp6.gemm_from_gdn)
assert hasattr(torch.ops.mxfp6, 'gemm_from_swiglu')
assert hasattr(torch.ops.mxfp6, 'gemm_w6a8_pdl')
ba = str(torch.ops.mxfp8_sm120.bf16_ba_m16.default._schema)
assert 'Tensor x' in ba and 'Tensor weight' in ba, ba
gemm = str(torch.ops.mxfp8_sm120.gemm.default._schema)
assert 'bool use_pdl=False' in gemm, gemm
"""


def discover_toolchain(cuda, site_paths=None):
    """Admit CUDA 13 NVRTC sonames even without an unversioned linker alias."""
    cuda = Path(cuda).resolve()
    nvcc = cuda / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA compiler not found: {nvcc}")
    libraries = {path.resolve() for directory in (cuda / "targets/x86_64-linux/lib", cuda / "lib64")
                 for pattern in ("libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.13.*")
                 for path in directory.glob(pattern) if path.is_file()}
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one CUDA 13 NVRTC library, found: {sorted(libraries)}")
    if site_paths is None:
        site_paths = [Path(value) for value in {sysconfig.get_path("purelib"),
                      sysconfig.get_path("platlib"), *sys.path} if value]
    includes = {path.resolve() for site in site_paths for path in Path(site).glob("nvidia/cu13/include")
                if (path / "cuda_fp16.h").is_file() and (path / "cuda_bf16.h").is_file()}
    if len(includes) != 1:
        raise RuntimeError(f"Expected one installed nvidia/cu13/include directory, found: {sorted(includes)}")
    return {"nvcc": str(nvcc), "nvrtc": str(next(iter(libraries))),
            "cu13_include": str(next(iter(includes)))}


def cmake_arguments(cutlass, toolchain):
    return [f"-DCUTLASS_DIR={cutlass}", f"-DCMAKE_CUDA_COMPILER={toolchain['nvcc']}",
            f"-DCUDA_NVRTC_LIB={toolchain['nvrtc']}",
            "-DCMAKE_CUDA_FLAGS=-isystem " + shlex.quote(toolchain["cu13_include"])]


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
    for name in PATCHES:
        patch_file(cutlass, source / "patches/cutlass" / name)
    env = dict(os.environ)
    cuda = Path(env["CUDA_HOME"])
    toolchain = discover_toolchain(cuda)
    env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
    env["CUDACXX"] = str(cuda / "bin/nvcc")
    env["CMAKE_ARGS"] = shlex.join(cmake_arguments(cutlass, toolchain))
    wheels = WORK / "wheels"
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    str(source), "-w", str(wheels)], env=env, check=True)
    wheel, = wheels.glob("mxfp6_sm120-0.2.1-*.whl")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
    # This needs no GPU, but catches the dispatcher ABI mismatch of the PyPI wheel.
    subprocess.run([sys.executable, "-c", SMOKE], check=True)
    (WORK / "mxfp6-source.json").write_text(json.dumps({"revision": REVISION, "cutlass": CUTLASS,
        "patches": list(PATCHES), "toolchain": toolchain}) + "\n")


if __name__ == "__main__":
    main()
