"""Build the M32 module separately from the reviewed exllamav3 extension.

Usage:
    EXLLAMA_V3_SOURCE=/path/to/exllamav3/exllamav3_ext \
      python setup.py build_ext --inplace

The base source tree is used only for shared headers and device helpers.  No
base binding or DevCtx object is linked into this module.
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


root = Path(__file__).parent
base_ext = os.environ.get("EXLLAMA_V3_SOURCE")
if not base_ext:
    raise RuntimeError("EXLLAMA_V3_SOURCE must point to exllamav3_ext")
base_ext = Path(base_ext).resolve()
if not (base_ext / "quant/exl3_gemm_kernel.cuh").is_file():
    raise RuntimeError(
        "EXLLAMA_V3_SOURCE must be a complete exllamav3_ext tree "
        "with quant/exl3_gemm_kernel.cuh"
    )

setup(
    name="exllamav3_m32_ext",
    version="0.1.0a1",
    license="MIT",
    ext_modules=[
        CUDAExtension(
            name="exllamav3_m32_ext",
            sources=[
                str(root / "src/m32_bindings.cpp"),
                str(root / "src/quant/exl3_mgemm_bf16_io_m32.cu"),
            ],
            include_dirs=[str(root / "src"), str(base_ext), str(base_ext / "quant")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    "--use_fast_math",
                    "-Xcudafe",
                    "--diag_suppress=177",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
