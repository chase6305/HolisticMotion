#pragma once

#include <string>

#include <pybind11/pybind11.h>

#include "../../src/kinematics/FrameValidation.h"

namespace holistic_motion::python {
namespace py = pybind11;

using robotics::detail::IsRigidTransform;

inline void RequireRigidTransform(const Eigen::Matrix4d &transform,
                                  const char *name) {
    if (!IsRigidTransform(transform)) {
        throw py::value_error(std::string(name) +
                              " must be a finite rigid transform");
    }
}

} // namespace holistic_motion::python
