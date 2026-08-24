
#include "holistic_motion/kinematics/utility.h"

namespace holistic_motion {
namespace robotics {

// add joints vector to ik_joints
bool IkRtn::PushBack(const Eigen::VectorXd& joints) {
    if (joints.size() == 0 || !joints.allFinite()) return false;
    if (!ik_joints.empty() && joints.size() != this->dof) {
        holistic_motion::utility::LogWarning(
                "Ignoring IK with inconsistent DOF: expected {}, got {}",
                this->dof, joints.size());
        return false;
    }
    this->dof = joints.size();
    ik_joints.push_back(joints);
    ik_number++;
    return true;
};

bool IkRtn::RemoveRepeatedIK(const double& eps) {
    if (0 == ik_number) {
        return false;
    }
    const int ik_number_before = ik_number;

    auto compare = [this, eps](const Eigen::VectorXd& q1,
                               const Eigen::VectorXd& q2) {
        for (int i = 0; i < this->dof; ++i) {
            if (std::abs(q1[i] - q2[i]) >= eps) {
                return q1[i] < q2[i];
            }
        }
        return false;
    };

    std::set<Eigen::VectorXd, decltype(compare)> unique_ik_joints(compare);

    for (const auto& ik_joint : ik_joints) {
        unique_ik_joints.insert(ik_joint);
    }

    ik_joints.assign(unique_ik_joints.begin(), unique_ik_joints.end());
    ik_number = ik_joints.size();

    holistic_motion::utility::LogDebug("Enter {} IK and remove {} IK.", ik_number_before,
                            ik_number_before - ik_number);

    if (ik_number == 0) {
        success = false;
        return false;
    }

    success = true;
    return true;
}

void _GenerateAllIk(const std::vector<std::vector<double>>& joints,
                    std::vector<double>& ik,
                    std::vector<std::vector<double>>& new_ik) {
    size_t i = ik.size();

    if (i == joints.size()) {
        new_ik.push_back(ik);
        return;
    }
    for (auto& j : joints[i]) {
        ik.push_back(j);
        _GenerateAllIk(joints, ik, new_ik);
        ik.pop_back();
    }
}

std::vector<double> _Lim(const double& upper_limit,
                         const double& lower_limit,
                         double joint) {
    std::vector<double> results;
    if (!std::isfinite(upper_limit) || !std::isfinite(lower_limit) ||
        !std::isfinite(joint) || lower_limit > upper_limit) {
        return results;
    }
    const double two_pi = 2 * M_PI;

    // Judging the joint angle out-of-bounds problem,
    // retract the out-of-bounds solution plus or minus 2 PI to the joint angle
    // range Z
    if (joint > upper_limit) {
        while (joint > upper_limit) {
            joint -= two_pi;
        }

        while (joint > lower_limit) {
            results.push_back(joint);
            joint -= two_pi;
        }
    } else if (joint < lower_limit) {
        while (joint < lower_limit) {
            joint += two_pi;
        }

        while (joint < upper_limit) {
            results.push_back(joint);
            joint += two_pi;
        }
    } else {
        results.push_back(joint);

        double joint_tmp = joint - two_pi;
        while (joint_tmp > lower_limit) {
            results.push_back(joint_tmp);
            joint_tmp -= two_pi;
        }
        joint_tmp = joint + two_pi;
        while (joint_tmp < upper_limit) {
            results.push_back(joint_tmp);
            joint_tmp += two_pi;
        }
    }
    return results;
}

bool IkRtn::GetLimitsIK(const std::vector<JointNode>& joint_nodes) {
    if (0 == ik_number || joint_nodes.size() < static_cast<size_t>(dof)) {
        success = false;
        return false;
    }
    const size_t initial_ik_count = ik_joints.size();

    std::vector<std::vector<double>> new_ik;
    std::vector<Eigen::VectorXd> new_ik_joints;

    // Pre-calculate joint limits for all DOFs
    std::vector<std::pair<double, double>> joint_limits(this->dof);
    for (int j = 0; j < this->dof; ++j) {
        const auto& node = joint_nodes[j];
        joint_limits[j] = {std::min(node.upper_limit, node.upper_limit_set),
                           std::max(node.lower_limit, node.lower_limit_set)};
    }

    new_ik.reserve(initial_ik_count * 2);

    // Process each IK solution
    for (const auto& ik : ik_joints) {  // Use range-based for loop
        if (ik.size() != this->dof || !ik.allFinite()) continue;
        std::vector<std::vector<double>> joints(this->dof);

        // Generate valid joint configurations
        for (int j = 0; j < this->dof; ++j) {
            const auto& [upper, lower] = joint_limits[j];
            joints[j] = _Lim(upper, lower, ik[j]);
        }

        // Generate valid IK combinations
        std::vector<double> ik_mid;
        _GenerateAllIk(joints, ik_mid, new_ik);
    }

    // Direct map conversion without element-wise copy
    new_ik_joints.reserve(new_ik.size());
    for (const auto& sol : new_ik) {
        if (sol.size() != static_cast<size_t>(this->dof)) continue;
        new_ik_joints.emplace_back(
                Eigen::Map<const Eigen::VectorXd>(sol.data(), this->dof));
    }

    // Update results
    ik_number = new_ik_joints.size();
    ik_joints = std::move(new_ik_joints);  // Move assignment for O(1) transfer

    holistic_motion::utility::LogDebug(
            "IK limit filtering: Input {} solutions, Output {} solutions",
            initial_ik_count, ik_number);

    if (ik_number == 0) {
        success = false;
        return false;
    }

    success = !ik_joints.empty();
    return success;
}

bool IkRtn::WrapToLimitsNear(const std::vector<JointNode>& joint_nodes,
                             const Eigen::VectorXd& reference) {
    if (ik_joints.empty() || reference.size() != dof ||
        joint_nodes.size() < static_cast<size_t>(dof) ||
        !reference.allFinite()) {
        success = false;
        return false;
    }
    constexpr double kTwoPi = 2.0 * M_PI;
    std::vector<Eigen::VectorXd> wrapped;
    wrapped.reserve(ik_joints.size());
    for (auto solution : ik_joints) {
        if (solution.size() != dof || !solution.allFinite()) continue;
        bool valid = true;
        for (int joint = 0; joint < dof; ++joint) {
            const auto& node = joint_nodes[joint];
            const double lower = std::max(node.lower_limit,
                                          node.lower_limit_set);
            const double upper = std::min(node.upper_limit,
                                          node.upper_limit_set);
            if (!std::isfinite(lower) || !std::isfinite(upper) ||
                lower > upper) {
                valid = false;
                break;
            }
            const double minimum_turn = std::ceil(
                    (lower - solution[joint]) / kTwoPi);
            const double maximum_turn = std::floor(
                    (upper - solution[joint]) / kTwoPi);
            if (minimum_turn > maximum_turn) {
                valid = false;
                break;
            }
            const double nearest_turn = std::round(
                    (reference[joint] - solution[joint]) / kTwoPi);
            const double turn = std::clamp(nearest_turn, minimum_turn,
                                           maximum_turn);
            solution[joint] += turn * kTwoPi;
        }
        if (valid) wrapped.push_back(std::move(solution));
    }
    ik_joints = std::move(wrapped);
    ik_number = static_cast<int>(ik_joints.size());
    success = ik_number > 0;
    if (success) RemoveRepeatedIK();
    return success;
}

bool IkRtn::SortMinMovement(const Eigen::VectorXd& target_joints,
                            std::vector<double>& dist) {
    (void)target_joints;
    if (0 == ik_number || dist.size() != static_cast<size_t>(ik_number)) {
        success = false;
        return false;
    }

    std::vector<size_t> indices(ik_number);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&dist](size_t i1, size_t i2) { return dist[i1] < dist[i2]; });

    std::vector<Eigen::VectorXd> sorted_ik_joints(ik_number);
    std::vector<double> sorted_dist(ik_number);

    for (size_t i = 0; i < static_cast<size_t>(ik_number); ++i) {
        sorted_ik_joints[i] = ik_joints[indices[i]];
        sorted_dist[i] = dist[indices[i]];
    }

    ik_joints = std::move(sorted_ik_joints);
    dist = std::move(sorted_dist);

    success = true;
    return true;
}

double IkRtn::_CalculateAngleBetweenVectors(const Eigen::Vector3d& v1,
                                            const Eigen::Vector3d& v2) {
    double dot_product = v1.dot(v2);
    double v1_magnitude = v1.norm();
    double v2_magnitude = v2.norm();
    double cos_theta = dot_product / (v1_magnitude * v2_magnitude);
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);
    double angle_radians = std::acos(cos_theta);

    Eigen::Vector3d cross_product = v1.cross(v2);
    if (cross_product.z() < 0) {
        angle_radians = -angle_radians;
    }

    return angle_radians;
}

bool IkRtn::LimitRobotConfig(const std::vector<JointNode>& joint_nodes,
                             const RobotConfigManager& robot_config) {
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
            std::vector<SE3d> pose_list;
            int num = 4;

            for (auto it = this->ik_joints.begin();
                 it != this->ik_joints.end();) {
                Eigen::VectorXd target_joint = *it;
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
                    if (JointType::REVOLUTE == joint_nodes[i].joint_type) {
                        Eigen::Vector3d rota_vec =
                                target_joint[i] * joint_nodes[i].axis;
                        rot = SO3d(rota_vec);
                    }
                    if (JointType::PRISMATIC == joint_nodes[i].joint_type) {
                        xyz = joint_nodes[i].axis * target_joint[i];
                    }
                    SE3d transform = SE3d(xyz, rot);
                    pose_list.push_back(pre_pose * joint_nodes[i].origin_pose *
                                        transform);
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

    holistic_motion::utility::LogDebug("Enter {} IK and filter {} IK.", ik_number_before,
                            ik_number_before - ik_number);
    if (ik_number == 0) {
        success = false;
        return false;
    }
    success = true;
    return true;
}

bool IkRtn::FilterJointSolutions(const JointFilterManager& filter) {
    if (ik_joints.empty()) {
        return false;
    }

    const int dof_size = this->dof;
    if (filter.Size() != static_cast<size_t>(dof_size)) {
        holistic_motion::utility::LogWarning(
                "Filter size ({}) does not match robot DOF ({})", filter.Size(),
                dof_size);
        return false;
    }

    const int total = static_cast<int>(ik_joints.size());
    std::vector<bool> is_valid(total, true);

    for (int i = 0; i < total; ++i) {
        const auto& solution = ik_joints[i];
        for (int j = 0; j < dof_size; ++j) {
            if (!filter.IsValidJointAngle(static_cast<size_t>(j),
                                          solution[j])) {
                is_valid[i] = false;
                break;
            }
        }
    }

    std::vector<Eigen::VectorXd> filtered_solutions;
    filtered_solutions.reserve(static_cast<size_t>(total));

    for (int i = 0; i < total; ++i) {
        if (is_valid[i]) {
            filtered_solutions.push_back(std::move(ik_joints[i]));
        }
    }

    this->ik_joints = std::move(filtered_solutions);
    this->ik_number = this->ik_joints.size();
    this->success = (this->ik_number > 0);

    holistic_motion::utility::LogDebug(
            "Joint filtering: Input {} solutions, Output {} solutions", total,
            this->ik_number);

    return this->success;
}

}  // namespace robotics
}  // namespace holistic_motion
