#pragma once

#include <pybind11/pybind11.h>

namespace holistic_motion::python {

void BindCollision(pybind11::module_& module);
void BindSamplingPlanning(pybind11::module_& module);

}  // namespace holistic_motion::python
