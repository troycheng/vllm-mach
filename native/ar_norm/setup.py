"""Build the optional SM120 TP2 AR+RMSNorm+MXFP8 primitive."""
from importlib import metadata
from pathlib import Path
import re
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

if metadata.version("flashinfer-python") != "0.6.18":
    raise RuntimeError("AR norm requires flashinfer-python==0.6.18 headers")
if not CUDA_HOME:
    raise RuntimeError("Set CUDA_HOME to a CUDA 13.0 toolkit")
nvcc = subprocess.check_output(
    [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
)
if not re.search(r"release 13\.0,", nvcc):
    raise RuntimeError("This AR norm profile requires the CUDA 13.0 compiler")
root = Path(metadata.distribution("flashinfer-python").locate_file("flashinfer/data"))
includes = [root / path for path in ("include", "spdlog/include", "cutlass/include")]
for path in includes:
    if not path.is_dir():
        raise RuntimeError(f"Missing FlashInfer build headers: {path}")

setup(
    name="vllm-mach-ar-norm",
    version="0.1.0a1",
    license="Apache-2.0",
    license_files=["LICENSE"],
    ext_modules=[
        CUDAExtension(
            "mach_ar_norm_ext",
            ["binding.cu"],
            include_dirs=[str(path) for path in includes],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17", "-arch=sm_120f", "-use_fast_math", "-DNDEBUG",
                    "-DFLASHINFER_ENABLE_F16", "-DFLASHINFER_ENABLE_BF16",
                    "-DFLASHINFER_ENABLE_FP8_E4M3", "-DFLASHINFER_ENABLE_FP8_E5M2",
                    "-DFLASHINFER_ENABLE_FP8_E8M0", "-DFLASHINFER_ENABLE_FP4_E2M1",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
