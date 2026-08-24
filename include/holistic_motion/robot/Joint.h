#pragma once
#include "holistic_motion/kinematics/Types.h"
#include "holistic_motion/utility/Logging.h"

namespace holistic_motion {
namespace robotics {

class JointLimit : public std::enable_shared_from_this<JointLimit> {
public:
    double lower_limit{0.0};
    double upper_limit{0.0};
    double max_velocity{0.0};
    double max_acceleration{0.0};
    double max_jerk{0.0};
    double max_effort{0.0};
    JointLimit();
    JointLimit(double lower_limit,
               double upper_limit,
               double max_velocity = 0,
               double max_acceleration = 0,
               double max_jerk = 0,
               double max_effort = 0);

    std::string report() const { return _report(*this); }

    /// \brief report self
    ///
    /// \param JointLimit
    /// \return std::string
    friend std::string _report(const JointLimit& params) {
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
    };
};

class Joint : public std::enable_shared_from_this<Joint> {
public:
    std::string name;
    std::string parent_link;
    std::string child_link;
    SE3d origin_pose;
    JointType joint_type{JointType::UNKNOWN};
    Eigen::Vector3d axis{Eigen::Vector3d::Zero()};
    JointLimit limit;
    std::string mimic_joint;
    double mimic_multiplier{1.0};
    double mimic_offset{0.0};
    Joint();
    Joint(const std::string& name,
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
          double effort_limit);
    friend std::ostream& operator<<(std::ostream& os, const Joint& joint);

    /// \brief report self
    ///
    /// \param Joint
    /// \return std::string
    friend std::string _report(const Joint& params) {
        std::string res;
        res += fmt::format(
                "\t {}:\n"
                "\t - [ parent_link = {}, child_link = {}\n"
                "\t - origin_pose : SE3d({})\n"
                "\t - joint_type : {}\n"
                "\t - axis: [{}]\n"
                "\t {}\n"
                "\t - ]\n",
                params.name, params.parent_link, params.child_link,
                fmt::join(params.origin_pose.Coeffs(), " , "),
                holistic_motion::robotics::to_underlying_type(params.joint_type),
                fmt::join(params.axis, " , "), params.limit.report());

        return res;
    };
};
}  // namespace robotics
}  // namespace holistic_motion
