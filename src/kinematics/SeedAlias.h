#pragma once

#include <algorithm>

#include "holistic_motion/kinematics/utility.h"

namespace holistic_motion {
namespace robotics {
namespace detail {

inline bool SeedAliasesSolutions(const IkRtn &solutions,
                                 const Eigen::VectorXd &seed) {
    return std::any_of(
        solutions.ik_joints.begin(), solutions.ik_joints.end(),
        [&seed](const auto &solution) { return &solution == &seed; });
}

} // namespace detail
} // namespace robotics
} // namespace holistic_motion
