#pragma once

#include <algorithm>
#include <vector>

#include "holistic_motion/kinematics/utility.h"

namespace holistic_motion {
namespace robotics {
namespace detail {

inline bool SeedAliasesSolutions(const std::vector<Eigen::VectorXd> &solutions,
                                 const Eigen::VectorXd &seed) {
    return std::any_of(solutions.begin(), solutions.end(),
                       [&seed](const auto &solution) { return &solution == &seed; });
}

inline bool SeedAliasesSolutions(const IkRtn &solutions, const Eigen::VectorXd &seed) {
    return SeedAliasesSolutions(solutions.ik_joints, seed);
}

} // namespace detail
} // namespace robotics
} // namespace holistic_motion
