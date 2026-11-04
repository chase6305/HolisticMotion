#pragma once

#include <array>
#include <utility>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"

namespace holistic_motion::robotics::srs_detail {

struct CanonicalModel {
    Eigen::Matrix4d base_transform{Eigen::Matrix4d::Identity()};
    // Prepared without the caller's TCP.
    Eigen::Matrix4d tool_transform{Eigen::Matrix4d::Identity()};
    // canonical_angle = axis_direction * axis_norm * coordinate + angle_offset.
    Eigen::Matrix<double, 7, 1> angle_offsets;
    std::array<int, 7> axis_directions;
    std::array<double, 7> axis_norms;
    double upper_arm_length{0.0};
    double forearm_length{0.0};
};

std::pair<Eigen::Vector3d, double>
FitAxisIntersection(const std::array<Eigen::Vector3d, 3> &points,
                    const std::array<Eigen::Vector3d, 3> &axes);

bool ChainCentres(const SRSKinematics &solver, const Eigen::VectorXd &joints,
                  const Eigen::Vector3d &shoulder, Eigen::Vector3d &elbow,
                  Eigen::Vector3d &wrist, Eigen::Vector3d &elbow_axis);

} // namespace holistic_motion::robotics::srs_detail
