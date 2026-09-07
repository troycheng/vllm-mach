"""Build the optional exact-M24 SM120 K6 module against ExLlamaV3 headers."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent
base = Path(os.environ["EXLLAMA_V3_SOURCE"]).resolve()
if not (base / "util.h").is_file():
    raise RuntimeError("EXLLAMA_V3_SOURCE must point to the directory containing util.h")
setup(
    name="exllamav3_temporal_m24_ext",
    version="0.1.0a5",
    license="MIT",
    ext_modules=[CUDAExtension(
        "exllamav3_temporal_m24_ext",
        [str(root / "bindings.cpp"), str(root / "kernel.cu")],
        include_dirs=[str(base), str(base / "quant")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": [
            "-O3", "-std=c++17", "-lineinfo", "--use_fast_math",
            "-Xptxas=-v", "-Xcudafe", "--diag_suppress=177",
        ]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
