#pragma once

#include <array>
#include <utility>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"

namespace holistic_motion::robotics::srs_detail {

struct CanonicalModel {
    Eigen::Matrix4d base{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d tool{Eigen::Matrix4d::Identity()};
    Eigen::Matrix<double, 7, 1> offsets;
    std::array<int, 7> directions;
    std::array<double, 7> axis_norms;
    double upper{0.0};
    double lower{0.0};
};

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
