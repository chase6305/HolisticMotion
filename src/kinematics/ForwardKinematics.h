#pragma once

#include "holistic_motion/kinematics/Types.h"

namespace holistic_motion::robotics {
class KinematicsBase;
namespace detail {
bool ComputeTerminalFK(const KinematicsBase &solver, const Eigen::VectorXd &joints,
                       SE3d &pose);
bool ComputeAllFK(const KinematicsBase &solver, const Eigen::VectorXd &joints,
                  std::vector<SE3d> &poses);
// The caller owns a 6-by-DOF workspace. No intermediate pose list is allocated.
bool ComputeJacobian(const KinematicsBase &solver, const Eigen::VectorXd &joints,
                     Eigen::Ref<Eigen::MatrixXd> jacobian);
} // namespace detail
} // namespace holistic_motion::robotics
