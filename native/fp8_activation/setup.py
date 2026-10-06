"""Build the optional precompiled SM120 SiLU/group128 quantization wheel."""

from pathlib import Path
import subprocess
import sys
import sysconfig

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

if torch.__version__ != "2.13.0+cu130":
    raise RuntimeError("Build FP8 activation with Torch2.13.0+cu130")
if torch.version.cuda != "13.0" or not CUDA_HOME:
    raise RuntimeError("Build FP8 activation with Torch CUDA13.0 and CUDA_HOME")
nvcc = subprocess.check_output([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"],
                               text=True)
if "release 13.0, V13.0.88" not in nvcc:
    raise RuntimeError("FP8 activation requires the CUDA13.0.88 compiler")
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
system_include_flags = ["-isystem", str(runtime_include)]

setup(
    name="vllm-mach-fp8-activation",
    version="0.1.0a1",
    license="Apache-2.0",
    license_files=["LICENSE"],
    ext_modules=[CUDAExtension(
        "mach_fp8_activation_ext", ["activation.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", *system_include_flags],
            "nvcc": ["-O3", "-std=c++20", "-lineinfo",
                     "-gencode=arch=compute_120,code=sm_120", "--threads=2",
                     *system_include_flags],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
