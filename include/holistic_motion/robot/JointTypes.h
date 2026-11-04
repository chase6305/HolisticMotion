#pragma once

#include <Eigen/Core>

#include "holistic_motion/manif/SE3.h"

namespace holistic_motion::robotics {

// Model joint descriptions shared by parsers, kinematics and client code.
enum class JointType {
    UNKNOWN,
    REVOLUTE,
    PRISMATIC,
    CONTINUOUS,
    FLOATING,
    PLANAR,
    FIXED
};

struct JointNode {
    SE3d origin_pose{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    Eigen::Vector3d axis{Eigen::Vector3d::Zero()};
    JointType joint_type{JointType::UNKNOWN};
    double lower_limit_set{0.0};
    double upper_limit_set{0.0};
    double lower_limit{0.0};
    double upper_limit{0.0};

    JointNode() = default;
    JointNode(const SE3d &origin, const Eigen::Vector3d &joint_axis,
              JointType type, double lower, double upper)
        : origin_pose(origin), axis(joint_axis), joint_type(type),
          lower_limit_set(lower), upper_limit_set(upper), lower_limit(lower),
          upper_limit(upper) {}
};

} // namespace holistic_motion::robotics
