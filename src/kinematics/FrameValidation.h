#pragma once

#include <cmath>

#include <Eigen/LU>

#include "holistic_motion/kinematics/Types.h"

namespace holistic_motion {
namespace robotics {
namespace detail {

inline bool IsRigidTransform(const Eigen::Matrix4d &transform) {
    const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
    return transform.allFinite() &&
           transform.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9) &&
           (rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-7) &&
           std::abs(rotation.determinant() - 1.0) <= 1e-7;
}

inline bool IsValidFrame(const SE3d &pose) {
    return pose.Coeffs().allFinite() &&
           std::abs(pose.Coeffs().tail<4>().squaredNorm() - 1.0) <= 1e-9;
}

} // namespace detail
} // namespace robotics
} // namespace holistic_motion
