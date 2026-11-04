#include "holistic_motion/robot/Joint.h"

#include <ostream>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "holistic_motion/utility/Types.h"

namespace holistic_motion {
namespace robotics {

JointLimit::JointLimit() {}

JointLimit::JointLimit(double lower_limit, double upper_limit,
                       double max_velocity, double max_acceleration,
                       double max_jerk, double max_effort)
    : lower_limit(lower_limit), upper_limit(upper_limit),
      max_velocity(max_velocity), max_acceleration(max_acceleration),
      max_jerk(max_jerk), max_effort(max_effort) {}

std::string JointLimit::report() const { return _report(*this); }

std::string _report(const JointLimit &params) {
    std::string res;
    res += "\t JointLimit: \n";
    res += fmt::format(
        "\t - [ lower_limit = {}, upper_limit = {},\n"
        "\t - max_velocity : {}, max_acceleration : {}, max_jerk: {}, "
        "max_effort: {}\n"
        "\t - ]",
        params.lower_limit, params.upper_limit, params.max_velocity,
        params.max_acceleration, params.max_jerk, params.max_effort);
    return res;
}

Joint::Joint() {}

Joint::Joint(const std::string &name, const std::string &parent_link,
             const std::string &child_link, const SE3d &origin_pose,
             JointType joint_type, const Eigen::Vector3d &axis,
             double lower_limit, double upper_limit, double velocity_limit,
             double acceleration_limit, double jerk_limit, double effort_limit)
    : name(name), parent_link(parent_link), child_link(child_link),
      origin_pose(origin_pose), joint_type(joint_type), axis(axis),
      limit(lower_limit, upper_limit, velocity_limit, acceleration_limit,
            jerk_limit, effort_limit) {}

std::string _report(const Joint &params) {
    return fmt::format("\t {}:\n"
                       "\t - [ parent_link = {}, child_link = {}\n"
                       "\t - origin_pose : SE3d({})\n"
                       "\t - joint_type : {}\n"
                       "\t - axis: [{}]\n"
                       "\t {}\n"
                       "\t - ]\n",
                       params.name, params.parent_link, params.child_link,
                       fmt::join(params.origin_pose.Coeffs(), " , "),
                       to_underlying_type(params.joint_type),
                       fmt::join(params.axis, " , "), params.limit.report());
}

std::ostream &operator<<(std::ostream &os, const Joint &joint) {
    os << fmt::format(
        "\n\tjoint name:{}\n\tparent link name:{}\n\tchild link name:{}",
        joint.name, joint.parent_link, joint.child_link);
    os << fmt::format("\n\tjoint type:{}",
                      to_underlying_type(joint.joint_type));
    os << fmt::format("\n\taxis:{}", fmt::join(joint.axis, " , "));
    os << fmt::format("\n\torigin pose:SE3d[{}]",
                      fmt::join(joint.origin_pose.Coeffs(), " , "));
    os << fmt::format("\n\tlimit:\n\t\tlower:{}, upper:{}\n"
                      "\t\tvel: {}, acc:{}, jerk:{}, effort:{}",
                      joint.limit.lower_limit, joint.limit.upper_limit,
                      joint.limit.max_velocity, joint.limit.max_acceleration,
                      joint.limit.max_jerk, joint.limit.max_effort);

    return os;
}

} // namespace robotics
} // namespace holistic_motion
