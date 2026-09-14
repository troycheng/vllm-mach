// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include <string>

TORCH_LIBRARY(mach_owner_build, m) {
  m.def("abi() -> str", []() { return std::string("owner-prefill-v1"); });
}
