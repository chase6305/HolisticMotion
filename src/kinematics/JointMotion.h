#pragma once

#include <cmath>
#include <limits>

#include "holistic_motion/kinematics/Types.h"

namespace holistic_motion::robotics::detail {

inline double JointAxisNorm(const Eigen::Vector3d &axis) {
    const double norm = axis.norm();
    return !std::isfinite(norm) ||
                   norm < std::sqrt(std::numeric_limits<double>::min())
               ? std::hypot(axis.x(), axis.y(), axis.z())
               : norm;
}

// Compose one scalar joint using the same convention in FK and IK filtering.
// Axis length scales the coordinate; URDF import normalizes direction axes.
inline bool AppendJointMotion(SE3d &pose, const JointNode &node, double position) {
    const auto type = node.joint_type;
    if (type == JointType::FIXED || type == JointType::UNKNOWN ||
        (position == 0.0 && node.axis.allFinite())) {
        pose = pose * node.origin_pose;
        return true;
    }
    const SE3d origin = pose * node.origin_pose;
    if (type == JointType::REVOLUTE || type == JointType::CONTINUOUS) {
        const double norm = JointAxisNorm(node.axis);
        const double angle = position * norm;
        if (!std::isfinite(angle))
            return false;
        const SO3d rotation =
            angle == 0.0 ? SO3d() : SO3d(Eigen::AngleAxisd(angle, node.axis / norm));
        pose = SE3d(origin.GetTranslation(), SO3d(origin.GetQuat()) * rotation);
        return true;
    }
    if (type == JointType::PRISMATIC) {
        const Eigen::Vector3d translation = node.axis * position;
        if (!translation.allFinite())
            return false;
        pose = origin * SE3d(translation, SO3d());
        return true;
    }
    return false;
}

} // namespace holistic_motion::robotics::detail
