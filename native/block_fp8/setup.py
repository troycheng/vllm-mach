# SPDX-License-Identifier: Apache-2.0
"""Build the selected SM120 n64 kernel with the validated header/toolchain pin."""

from importlib.metadata import distribution, version
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import sysconfig

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


if version("flashinfer-python") != "0.6.18":
    raise RuntimeError("block-FP8 requires FlashInfer 0.6.18 CUTLASS headers")
if torch.__version__ != "2.13.0+cu130":
    raise RuntimeError("block-FP8 requires torch==2.13.0+cu130")
if not CUDA_HOME:
    raise RuntimeError("CUDA 13.0.88 compiler is required")
compiler = subprocess.check_output(
    [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
)
if "release 13.0, V13.0.88" not in compiler:
    raise RuntimeError("block-FP8 requires the validated CUDA 13.0.88 compiler")
headers = Path(distribution("flashinfer-python").locate_file("flashinfer/data/cutlass"))
if not (headers / "include/cutlass/cutlass.h").is_file():
    raise RuntimeError("pinned FlashInfer CUTLASS headers are absent")
site_paths = {Path(value) for value in (
    sysconfig.get_path("purelib"), sysconfig.get_path("platlib"), *sys.path
) if value}
runtime_headers = {path.resolve() for site in site_paths
                   for path in site.glob("nvidia/cu13/include")
                   if (path / "cuda_fp16.h").is_file()
                   and (path / "cuda_bf16.h").is_file()}
if len(runtime_headers) != 1:
    raise RuntimeError("expected one installed nvidia/cu13/include directory")
runtime_include, = runtime_headers
# System include search keeps nvcc's own crt ahead of the pip runtime headers.
system_include_flags = ["-isystem", str(runtime_include)]
header_digest = hashlib.sha256()
for path in sorted(headers.rglob("*")):
    if path.is_file():
        header_digest.update(path.relative_to(headers).as_posix().encode())
        header_digest.update(b"\0")
        header_digest.update(hashlib.sha256(path.read_bytes()).digest())
Path("vllm_mach_block_fp8/build_metadata.json").write_text(json.dumps({
    "torch": torch.__version__, "flashinfer-python": version("flashinfer-python"),
    "nvcc": compiler, "cutlass_tree_sha256": header_digest.hexdigest(),
    "source_sha256": hashlib.sha256(Path("block_fp8.cu").read_bytes()).hexdigest(),
}, sort_keys=True, indent=2) + "\n")

setup(
    name="vllm-mach-block-fp8",
    version="0.1.0a1",
    description="Selected SM120 block-FP8 n64 Pingpong GEMM",
    license="Apache-2.0",
    license_files=["LICENSE"],
    packages=["vllm_mach_block_fp8"],
    package_data={"vllm_mach_block_fp8": ["build_metadata.json"]},
    install_requires=["torch==2.13.0"],
    ext_modules=[CUDAExtension(
        "vllm_mach_block_fp8._C", ["block_fp8.cu"],
        include_dirs=[str(headers / "include"), str(headers / "tools/util/include")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", *system_include_flags],
            "nvcc": [
                "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                "--expt-extended-lambda", "-gencode=arch=compute_120a,code=sm_120a",
                "--threads=2", "-Xptxas=-v",
                *system_include_flags,
            ],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
