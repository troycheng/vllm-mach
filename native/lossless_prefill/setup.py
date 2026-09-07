"""Build the optional SM120 TP2 prefill collective against FlashInfer 0.6.18."""
import importlib.metadata
from pathlib import Path
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

if importlib.metadata.version('flashinfer-python') != '0.6.18':
    raise RuntimeError('This source profile requires flashinfer-python==0.6.18')
if not CUDA_HOME or 'release 13.0,' not in subprocess.check_output(
    [str(Path(CUDA_HOME) / 'bin/nvcc'), '--version'], text=True
):
    raise RuntimeError('This validated source profile requires the CUDA 13.0 compiler')
root = Path(importlib.metadata.distribution('flashinfer-python').locate_file('flashinfer/data'))

setup(
    name='vllm-mach-lossless-prefill', version='0.1.0a3',
    license='Apache-2.0', license_files=['LICENSE'],
    ext_modules=[CUDAExtension(
        name, [source],
        include_dirs=[str(root / p) for p in ('include', 'spdlog/include', 'cutlass/include')],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '-std=c++17', '-arch=sm_120f', '-use_fast_math', '-DNDEBUG',
                     '-DFLASHINFER_ENABLE_F16', '-DFLASHINFER_ENABLE_BF16',
                     '-DFLASHINFER_ENABLE_FP8_E4M3', '-DFLASHINFER_ENABLE_FP8_E5M2',
                     '-DFLASHINFER_ENABLE_FP8_E8M0', '-DFLASHINFER_ENABLE_FP4_E2M1',
                     '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                     '-U__CUDA_NO_HALF2_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__'],
        }) for name, source in [
            ('mach_lossless_prefill_ext', 'ar_codec_extension.cu'),
            ('mach_lossless_prefill_sum_ext', 'sum_codec_extension.cu'),
            ('mach_lossless_prefill_direct_ext', 'direct_codec_extension.cu'),
        ]], cmdclass={'build_ext': BuildExtension},
)
