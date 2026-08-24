#include "holistic_motion/robot/Joint.h"

#include <Eigen/Dense>
namespace holistic_motion {
namespace robotics {

JointLimit::JointLimit() {}

JointLimit::JointLimit(double lower_limit,
                       double upper_limit,
                       double max_velocity,
                       double max_acceleration,
                       double max_jerk,
                       double max_effort)
    : lower_limit(lower_limit),
      upper_limit(upper_limit),
      max_velocity(max_velocity),
      max_acceleration(max_acceleration),
      max_jerk(max_jerk),
      max_effort(max_effort) {}

Joint::Joint() {}

Joint::Joint(const std::string& name,
             const std::string& parent_link,
             const std::string& child_link,
             const SE3d& origin_pose,
             JointType joint_type,
             const Eigen::Vector3d& axis,
             double lower_limit,
             double upper_limit,
             double velocity_limit,
             double acceleration_limit,
             double jerk_limit,
             double effort_limit)
    : name(name),
      parent_link(parent_link),
      child_link(child_link),
      origin_pose(origin_pose),
      joint_type(joint_type),
      axis(axis),
      limit(lower_limit,
            upper_limit,
            velocity_limit,
            acceleration_limit,
            jerk_limit,
            effort_limit) {
    holistic_motion::utility::LogDebug("Joint origin_pose:{}",
                            fmt::join(origin_pose.Coeffs(), ","));
}

std::ostream& operator<<(std::ostream& os, Joint& joint) {
    os << fmt::format(
            "\n\tjoint name:{}\n\tparent link name:{}\n\tchild link name:{}",
            joint.name, joint.parent_link, joint.child_link);
    os << fmt::format("\n\tjoint type:{}",
                      to_underlying_type(joint.joint_type));
    os << fmt::format("\n\taxis:{}", fmt::join(joint.axis, " , "));
    os << fmt::format("\n\torigin pose:SE3d[{}]",
                      fmt::join(joint.origin_pose.Coeffs(), " , "));
    os << fmt::format(
            "\n\tlimit:\n\t\tlower:{}, upper:{}\n"
            "\t\tvel: {}, acc:{}, jerk:{}, effort:{}",
            joint.limit.lower_limit, joint.limit.upper_limit,
            joint.limit.max_velocity, joint.limit.max_acceleration,
            joint.limit.max_jerk, joint.limit.max_effort);

    return os;
}

}  // namespace robotics
}  // namespace holistic_motion
