#pragma once

#include <cmath>
#include <string>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>

namespace holistic_motion::python {
namespace py = pybind11;

inline bool IsRigidTransform(const Eigen::Matrix4d &transform) {
    const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
    return transform.allFinite() &&
           transform.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0),
                                     1e-9) &&
           (rotation.transpose() * rotation)
               .isApprox(Eigen::Matrix3d::Identity(), 1e-7) &&
           std::abs(rotation.determinant() - 1.0) <= 1e-7;
}

inline void RequireRigidTransform(const Eigen::Matrix4d &transform,
                                  const char *name) {
    if (!IsRigidTransform(transform)) {
        throw py::value_error(std::string(name) +
                              " must be a finite rigid transform");
    }
}

} // namespace holistic_motion::python
