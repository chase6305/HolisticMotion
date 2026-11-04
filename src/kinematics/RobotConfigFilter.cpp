#include "holistic_motion/kinematics/utility.h"

#include "FrameValidation.h"
#include "JointMotion.h"

#include <array>
#include <cmath>

namespace holistic_motion::robotics {

bool IkRtn::LimitRobotConfig(const std::vector<JointNode> &joint_nodes,
                             const RobotConfigManager &robot_config) {
    joint_states.resize(0, 0);
    if (robot_config.IsDisable())
        return true;
    const auto fail = [this]() {
        Clear();
        return false;
    };
    // Only the six-axis geometric elbow classification is implemented.
    // Unsupported requests must not silently admit every candidate.
    if (dof != 6 || joint_nodes.size() < 4 || robot_config.IsBaseFront() ||
        robot_config.IsBaseBack() || robot_config.IsWristFlip() || robot_config.IsWristNotFlip())
        return fail();
    for (int index = 0; index < 4; ++index) {
        const auto &node = joint_nodes[index];
        const bool active = node.joint_type == JointType::REVOLUTE ||
                            node.joint_type == JointType::CONTINUOUS ||
                            node.joint_type == JointType::PRISMATIC;
        const double norm = detail::JointAxisNorm(node.axis);
        if (!detail::IsValidFrame(node.origin_pose) || !node.axis.allFinite() ||
            (active && (!std::isfinite(norm) || norm <= 1e-12)) ||
            (!active && node.joint_type != JointType::FIXED &&
             node.joint_type != JointType::UNKNOWN))
            return fail();
    }
    std::size_t kept = 0;
    for (std::size_t index = 0; index < ik_joints.size(); ++index) {
        auto &joints = ik_joints[index];
        if (joints.size() != dof || !joints.allFinite())
            return fail();
        SE3d pose;
        std::array<double, 3> heights;
        for (int joint = 0; joint < 4; ++joint) {
            if (!detail::AppendJointMotion(pose, joint_nodes[joint], joints[joint]) ||
                !pose.Coeffs().allFinite())
                return fail();
            if (joint > 0)
                heights[joint - 1] = pose.GetTranslation().z();
        }
        const bool elbow_up = !(heights[1] < heights[0] && heights[2] > heights[1]);
        if (elbow_up != robot_config.IsElbowUp())
            continue;
        if (kept != index)
            ik_joints[kept] = std::move(joints);
        ++kept;
    }
    ik_joints.resize(kept);
    ik_number = static_cast<int>(kept);
    success = kept != 0;
    return success;
}

} // namespace holistic_motion::robotics
