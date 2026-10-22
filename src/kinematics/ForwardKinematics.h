#pragma once

#include "holistic_motion/kinematics/Types.h"

namespace holistic_motion::robotics {
class KinematicsBase;
namespace detail {
bool ComputeTerminalFK(const KinematicsBase &solver, const Eigen::VectorXd &joints,
                       SE3d &pose);
bool ComputeAllFK(const KinematicsBase &solver, const Eigen::VectorXd &joints,
                  std::vector<SE3d> &poses);
} // namespace detail
} // namespace holistic_motion::robotics
