#include "holistic_motion/robot/RobotModel.h"

#include <chrono>
#include <filesystem>
#include <fstream>

int main() {
    const auto path =
        std::filesystem::temp_directory_path() /
        ("holistic-motion-model-consumer-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".urdf");
    std::ofstream(path)
        << R"(<robot name="model"><link name="base"/><link name="tip"/>
      <joint name="joint" type="prismatic"><parent link="base"/><child link="tip"/>
      <axis xyz="2 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
      </joint></robot>)";
    holistic_motion::robotics::RobotModel model(path.string());
    std::filesystem::remove(path);
    const auto joint = model.GetJoint("joint");
    if (model.GetModelDoF() != 1 ||
        model.GetChainJoints("base", "tip").size() != 1 || !joint ||
        !joint->axis.isApprox(Eigen::Vector3d::UnitX()))
        return 1;

    // The public report API is usable without including formatting headers.
    const std::string limit_report =
        "\t JointLimit: \n"
        "\t - [ lower_limit = -1, upper_limit = 1,\n"
        "\t - max_velocity : 1, max_acceleration : 0, max_jerk: 0, max_effort: "
        "1\n"
        "\t - ]";
    const std::string joint_report =
        "\t joint:\n"
        "\t - [ parent_link = base, child_link = tip\n"
        "\t - origin_pose : SE3d(0 , 0 , 0 , 0 , 0 , 0 , 1)\n"
        "\t - joint_type : 2\n"
        "\t - axis: [1 , 0 , 0]\n"
        "\t " +
        limit_report + "\n\t - ]\n";
    return joint->limit.report() == limit_report &&
                   _report(joint->limit) == limit_report &&
                   _report(*joint) == joint_report
               ? 0
               : 1;
}
