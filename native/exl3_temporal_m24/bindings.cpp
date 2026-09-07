#include <torch/extension.h>
void temporal_m24(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const at::Tensor&, at::Tensor&,
                  at::Tensor&, at::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_grouped", &temporal_m24);
}
