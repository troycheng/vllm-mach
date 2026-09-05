#include <torch/extension.h>

#include "quant/exl3_mgemm_bf16_io.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "grouped_had_m32",
        &exl3_mgemm_bf16_io_grouped_had_m32,
        "K5/K6 MCG grouped-Hadamard M32 BF16 I/O GEMM",
        py::arg("a"),
        py::arg("b"),
        py::arg("c_scratch"),
        py::arg("c_bf16"),
        py::arg("unique_suh"),
        py::arg("a_had"),
        py::arg("svh"),
        py::arg("had_group_ids"),
        py::arg("locks"),
        py::arg("bits"),
        py::arg("force_num_sms"),
        py::arg("output_stride"),
        py::arg("mcg"));

    m.def(
        "single_had_m32_fp16",
        &exl3_gemm_bf16_io_single_had_m32_fp16,
        "K5/K6 MCG single-matrix M32 BF16-input FP16-output GEMM",
        py::arg("a"),
        py::arg("b"),
        py::arg("c"),
        py::arg("suh"),
        py::arg("a_had"),
        py::arg("svh"),
        py::arg("locks"),
        py::arg("bits"),
        py::arg("force_num_sms"),
        py::arg("mcg"));
}
