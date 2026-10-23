#pragma once

#include <cmath>
#include <limits>

#include <Eigen/Core>

namespace holistic_motion::robotics::detail {

inline double JointAxisNorm(const Eigen::Vector3d &axis) {
    const double norm = axis.norm();
    return !std::isfinite(norm) ||
                   norm < std::sqrt(std::numeric_limits<double>::min())
               ? std::hypot(axis.x(), axis.y(), axis.z())
               : norm;
}

} // namespace holistic_motion::robotics::detail
