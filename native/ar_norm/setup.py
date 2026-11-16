"""Build the optional SM120 TP2 AR+RMSNorm+MXFP8 primitive."""
from importlib import metadata
from pathlib import Path
import re
import subprocess
import sysconfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vllm_mach.stack import BASE_PACKAGES, NATIVE_EXTENSIONS, TORCH_BUILD, check_versions

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

check_versions({"torch": BASE_PACKAGES["torch"], "flashinfer-python": BASE_PACKAGES["flashinfer-python"]})

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


# Keep the toolkit's crt first, while making the CUDA library headers visible.
site_paths = {Path(value) for value in (
    sysconfig.get_path("purelib"), sysconfig.get_path("platlib"), *sys.path
) if value}
runtime_headers = {path.resolve() for site in site_paths
                   for path in site.glob("nvidia/cu13/include")
                   if (path / "cuda_fp16.h").is_file()
                   and (path / "cusparse.h").is_file()}
if len(runtime_headers) != 1:
    raise RuntimeError("expected one installed nvidia/cu13/include directory")
runtime_include, = runtime_headers
system_include_flags = ["-isystem", str(runtime_include)]

setup(
    name="vllm-mach-ar-norm",
    version=NATIVE_EXTENSIONS["vllm-mach-ar-norm"],
    license="Apache-2.0",
    license_files=["LICENSE"],
    ext_modules=[
        CUDAExtension(
            "mach_ar_norm_ext",
            ["binding.cu"],
            include_dirs=[str(path) for path in includes],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", *system_include_flags],
                "nvcc": [
                    "-O3", "-std=c++20", "-arch=sm_120f", "-use_fast_math", "-DNDEBUG",
                    "-DFLASHINFER_ENABLE_F16", "-DFLASHINFER_ENABLE_BF16",
                    "-DFLASHINFER_ENABLE_FP8_E4M3", "-DFLASHINFER_ENABLE_FP8_E5M2",
                    "-DFLASHINFER_ENABLE_FP8_E8M0", "-DFLASHINFER_ENABLE_FP4_E2M1",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    *system_include_flags,
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
