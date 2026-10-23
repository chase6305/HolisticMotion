#include "holistic_motion/kinematics/utility.h"

#include "FrameValidation.h"
#include "JointMotion.h"

#include <array>
#include <cmath>

namespace holistic_motion::robotics {

bool IkRtn::LimitRobotConfig(const std::vector<JointNode> &joint_nodes,
                             const RobotConfigManager &robot_config) {
    if (robot_config.IsDisable()) {
        return true;
    }

    int ik_number_before = this->ik_joints.size();

    switch (this->dof) {
    case 7: {
        // TODO:
        // if (robot_config.IsElbowAnglePositive()) {
        //     this->ik_joints.erase(
        //             std::remove_if(this->ik_joints.begin(),
        //                            this->ik_joints.end(),
        //                            [](const Eigen::VectorXd& joint) {
        //                                return joint[3] < 0;
        //                            }),
        //             this->ik_joints.end());
        // } else if (robot_config.IsElbowAngleNegative()) {
        //     this->ik_joints.erase(
        //             std::remove_if(this->ik_joints.begin(),
        //                            this->ik_joints.end(),
        //                            [](const Eigen::VectorXd& joint) {
        //                                return joint[3] >= 0;
        //                            }),
        //             this->ik_joints.end());
        // }

        break;
    }
    case 6: {
        // This filter needs four prefix poses. Validate before indexing or
        // erasing candidates, including direct callers of the public IkRtn
        // utility rather than a validated kinematics model.
        if (joint_nodes.size() < 4) {
            Clear();
            return false;
        }
        std::array<double, 4> axis_norms{};
        for (int index = 0; index < 4; ++index) {
            const auto &node = joint_nodes[index];
            const bool active = node.joint_type == JointType::REVOLUTE ||
                                node.joint_type == JointType::CONTINUOUS ||
                                node.joint_type == JointType::PRISMATIC;
            const double axis_norm = detail::JointAxisNorm(node.axis);
            axis_norms[index] = axis_norm;
            if (!detail::IsValidFrame(node.origin_pose) ||
                !node.axis.allFinite() ||
                (active && (!std::isfinite(axis_norm) || axis_norm <= 1e-12)) ||
                (!active && node.joint_type != JointType::FIXED &&
                 node.joint_type != JointType::UNKNOWN)) {
                Clear();
                return false;
            }
        }
        for (const auto &candidate : ik_joints) {
            if (candidate.size() != dof || !candidate.allFinite()) {
                Clear();
                return false;
            }
        }
        std::vector<SE3d> pose_list;
        int num = 4;

        for (auto it = this->ik_joints.begin(); it != this->ik_joints.end();) {
            const Eigen::VectorXd &target_joint = *it;
            pose_list.clear();
            pose_list.reserve(num + 1);

            if (target_joint.size() != this->dof) {
                holistic_motion::utility::LogWarning(
                    "The size of target joint [{}] is not match to "
                    "dof[{}]",
                    target_joint.size(), this->dof);
                return false;
            }
            SE3d pre_pose = SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            for (int i = 0; i < num; i++) {
                Eigen::Vector3d xyz(0.0, 0.0, 0.0);
                SO3d rot = SO3d(0.0, 0.0, 0.0, 1.0);
                if (JointType::REVOLUTE == joint_nodes[i].joint_type ||
                    JointType::CONTINUOUS == joint_nodes[i].joint_type) {
                    const double norm = axis_norms[i];
                    const double angle = target_joint[i] * norm;
                    if (!std::isfinite(angle)) {
                        Clear();
                        return false;
                    }
                    rot = SO3d(
                        Eigen::AngleAxisd(angle, joint_nodes[i].axis / norm));
                }
                if (JointType::PRISMATIC == joint_nodes[i].joint_type) {
                    xyz = joint_nodes[i].axis * target_joint[i];
                }
                SE3d transform = SE3d(xyz, rot);
                pose_list.push_back(pre_pose * joint_nodes[i].origin_pose *
                                    transform);
                if (!pose_list.back().Coeffs().allFinite()) {
                    Clear();
                    return false;
                }
                pre_pose = pose_list.back();
            }
            // Eigen::Vector3d p0 = pose_list[0].GetTranslation();
            Eigen::Vector3d p1 = pose_list[1].GetTranslation();
            Eigen::Vector3d p2 = pose_list[2].GetTranslation();
            Eigen::Vector3d p3 = pose_list[3].GetTranslation();

            if (robot_config.IsElbowUp()) {
                if (p2.z() < p1.z()) {
                    if (p3.z() > p2.z()) {
                        this->ik_joints.erase(it);
                        continue;
                    }
                }
            }

            ++it;
        }
        break;
    }
    default:
        break;
    }

    ik_number = this->ik_joints.size();

    holistic_motion::utility::LogDebug("Enter {} IK and filter {} IK.",
                                       ik_number_before,
                                       ik_number_before - ik_number);
    if (ik_number == 0) {
        success = false;
        return false;
    }
    success = true;
    return true;
}

} // namespace holistic_motion::robotics
