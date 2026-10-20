#pragma once

#include <cmath>

#include "holistic_motion/kinematics/utility.h"

namespace holistic_motion {
namespace robotics {
namespace detail {

inline bool IsValidFrame(const SE3d &pose) {
    return pose.Coeffs().allFinite() &&
           std::abs(pose.Coeffs().tail<4>().squaredNorm() - 1.0) <= 1e-9;
}

} // namespace detail
} // namespace robotics
} // namespace holistic_motion
