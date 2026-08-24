#pragma once

#include <array>
#include <utility>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"

namespace holistic_motion::robotics::srs_detail {

std::pair<Eigen::Vector3d, double> FitAxisIntersection(
        const std::array<Eigen::Vector3d, 3>& points,
        const std::array<Eigen::Vector3d, 3>& axes);

bool ChainCentres(const SRSKinematics& solver,
                  const Eigen::VectorXd& joints,
                  const Eigen::Vector3d& shoulder,
                  Eigen::Vector3d& elbow,
                  Eigen::Vector3d& wrist,
                  Eigen::Vector3d& elbow_axis);

}  // namespace holistic_motion::robotics::srs_detail
