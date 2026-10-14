
#include "holistic_motion/kinematics/utility.h"

#include <algorithm>
#include <limits>

namespace holistic_motion {
namespace robotics {

// add joints vector to ik_joints
bool IkRtn::PushBack(const Eigen::VectorXd& joints) {
    if (!joints.allFinite()) return false;
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

namespace {
constexpr double kTwoPi = 2.0 * M_PI;
constexpr std::size_t kDefaultSolutionLimit = 65536;

struct CoordinateLimits {
    double lower;
    double upper;
    double period;
};

bool SameRotation(double first, double second) {
    // A large turn offset can lose its phase when rounded back to double.
    return std::abs(std::sin(first) - std::sin(second)) <= 1e-10 &&
           std::abs(std::cos(first) - std::cos(second)) <= 1e-10;
}

double CoordinatePeriod(const JointNode &node) {
    const double squared = node.axis.squaredNorm();
    if (squared == 1.0)
        return kTwoPi;
    const double norm = std::sqrt(squared);
    return norm > 0.0 && std::isfinite(norm) ? kTwoPi / norm : 0.0;
}

double CoordinateAngle(double joint, const JointNode &node) {
    // Retain the ordinary cardinal-axis phase exactly, including huge inputs.
    const auto &axis = node.axis;
    if ((axis.x() == 0.0 && axis.y() == 0.0 && std::abs(axis.z()) == 1.0) ||
        (axis.x() == 0.0 && axis.z() == 0.0 && std::abs(axis.y()) == 1.0) ||
        (axis.y() == 0.0 && axis.z() == 0.0 && std::abs(axis.x()) == 1.0))
        return joint;
    const Eigen::Vector3d rotation = joint * axis;
    double angle = rotation.norm();
    if (angle < std::sqrt(std::numeric_limits<double>::min()))
        angle = std::hypot(rotation.x(), rotation.y(), rotation.z());
    return std::copysign(angle, joint);
}

bool SameCoordinateRotation(double first, double second,
                            const JointNode &node) {
    const double first_angle = CoordinateAngle(first, node);
    const double second_angle = CoordinateAngle(second, node);
    return std::isfinite(first_angle) && std::isfinite(second_angle) &&
           SameRotation(first_angle, second_angle);
}

double PeriodicAnchor(double joint, const JointNode &node, double period) {
    double anchor = std::remainder(joint, period);
    if (!SameCoordinateRotation(anchor, joint, node)) {
        const double angle = CoordinateAngle(joint, node);
        if (!std::isfinite(angle))
            return std::numeric_limits<double>::quiet_NaN();
        anchor =
            std::atan2(std::sin(angle), std::cos(angle)) * (period / kTwoPi);
    }
    return anchor;
}

bool NearestPeriodicValue(double lower, double upper, double joint,
                          double reference, double &value,
                          const JointNode &node, double period) {
    if (!std::isfinite(lower) || !std::isfinite(upper))
        return false;
    const bool inside = joint >= lower && joint <= upper;
    if (inside && std::abs(reference - joint) < 0.5 * period) {
        value = joint;
        return true;
    }
    double anchor = joint;
    if (!inside) {
        anchor = PeriodicAnchor(joint, node, period);
    }
    const long double minimum =
        std::ceil((static_cast<long double>(lower) - anchor) / period);
    const long double maximum =
        std::floor((static_cast<long double>(upper) - anchor) / period);
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum)
        return false;
    const long double nearest =
        std::round((static_cast<long double>(reference) - anchor) / period);
    const long double turn = std::clamp(nearest, minimum, maximum);
    constexpr long double kMaxExactTurn = 4503599627370495.0L;
    if (!std::isfinite(turn) || std::abs(turn) > kMaxExactTurn)
        return false;
    value = turn == 0.0L ? anchor
                         : std::fma(static_cast<double>(turn), period, anchor);
    return value >= lower && value <= upper &&
           (value == joint || SameCoordinateRotation(value, joint, node));
}

// Returns false if enumeration cannot fit the budget or turn indices cannot
// be represented exactly. An empty result is an ordinary infeasible interval.
bool PeriodicValues(double lower, double upper, double joint,
                    std::size_t budget, std::vector<double> &values,
                    const JointNode &node, double period) {
    values.clear();
    if (!std::isfinite(lower) || !std::isfinite(upper) ||
        !std::isfinite(joint) || lower > upper)
        return true;
    const bool inside = joint >= lower && joint <= upper;
    if (inside && joint - lower < period && upper - joint < period) {
        values.push_back(joint);
        return true;
    }
    double anchor = joint;
    if (!inside) {
        anchor = PeriodicAnchor(joint, node, period);
    }
    const long double first =
        std::ceil((static_cast<long double>(lower) - anchor) / period);
    const long double last =
        std::floor((static_cast<long double>(upper) - anchor) / period);
    if (first > last)
        return true;
    constexpr long double kMaxExactTurn = 4503599627370495.0L;
    const long double count = last - first + 1.0L;
    if (!std::isfinite(first) || !std::isfinite(last) ||
        first < -kMaxExactTurn || last > kMaxExactTurn ||
        count > static_cast<long double>(budget))
        return false;

    values.reserve(static_cast<std::size_t>(count));
    const auto append = [&](long long turn) {
        const double candidate =
            turn == 0 ? anchor
                      : std::fma(static_cast<double>(turn), period, anchor);
        if (candidate >= lower && candidate <= upper &&
            (candidate == joint ||
             SameCoordinateRotation(candidate, joint, node)) &&
            (values.empty() || candidate != values.back()))
            values.push_back(candidate);
    };
    const auto minimum = static_cast<long long>(first);
    const auto maximum = static_cast<long long>(last);
    if (inside) {
        append(0);
        for (long long turn = -1; turn >= minimum; --turn)
            append(turn);
        for (long long turn = 1; turn <= maximum; ++turn)
            append(turn);
    } else if (joint > upper) {
        for (long long turn = maximum; turn >= minimum; --turn)
            append(turn);
    } else {
        for (long long turn = minimum; turn <= maximum; ++turn)
            append(turn);
    }
    return true;
}
} // namespace

bool IkRtn::GetLimitsIK(const std::vector<JointNode> &joint_nodes) {
    return GetLimitsIK(joint_nodes, kDefaultSolutionLimit);
}

bool IkRtn::GetLimitsIK(const std::vector<JointNode>& joint_nodes,
                        std::size_t max_solutions) {
    if (joint_states.rows() != 0 || joint_states.cols() != 0)
        joint_states.resize(0, 0);
    const auto fail = [this]() {
        ik_joints.clear();
        ik_number = 0;
        success = false;
        return false;
    };
    if (ik_joints.empty() || dof < 0 ||
        joint_nodes.size() < static_cast<std::size_t>(dof) ||
        max_solutions == 0 ||
        max_solutions >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return fail();

    std::vector<CoordinateLimits> limits(dof);
    for (int joint = 0; joint < dof; ++joint) {
        const auto& node = joint_nodes[joint];
        if (std::isnan(node.lower_limit) || std::isnan(node.upper_limit) ||
            std::isnan(node.lower_limit_set) ||
            std::isnan(node.upper_limit_set))
            return fail();
        limits[joint] = {std::max(node.lower_limit, node.lower_limit_set),
                         std::min(node.upper_limit, node.upper_limit_set), 0.0};
        if (limits[joint].lower > limits[joint].upper)
            return fail();
        if (node.joint_type == JointType::REVOLUTE ||
            node.joint_type == JointType::CONTINUOUS) {
            limits[joint].period = CoordinatePeriod(node);
            if (!(limits[joint].period > 0.0) ||
                !std::isfinite(limits[joint].period))
                return fail();
        }
    }

    std::vector<Eigen::VectorXd> filtered;
    std::vector<std::vector<double>> choices;
    std::vector<std::size_t> indices;
    Eigen::VectorXd candidate;
    for (const auto &solution : ik_joints) {
        if (solution.size() != dof || !solution.allFinite())
            continue;
        bool single_representation = true;
        for (int joint = 0; joint < dof; ++joint) {
            const auto [lower, upper, period] = limits[joint];
            const auto type = joint_nodes[joint].joint_type;
            const double value = solution[joint];
            if (value < lower || value > upper ||
                ((type == JointType::REVOLUTE || type == JointType::CONTINUOUS)
                     ? (!std::isfinite(lower) || !std::isfinite(upper) ||
                        value - lower >= period || upper - value >= period)
                     : (type != JointType::PRISMATIC &&
                        type != JointType::FIXED &&
                        type != JointType::UNKNOWN))) {
                single_representation = false;
                break;
            }
        }
        if (single_representation) {
            if (filtered.size() == max_solutions)
                return fail();
            filtered.push_back(solution);
            continue;
        }
        if (choices.empty()) {
            choices.resize(dof);
            indices.resize(dof);
            candidate.resize(dof);
        }
        bool feasible = true;
        for (int joint = 0; joint < dof; ++joint) {
            const auto [lower, upper, period] = limits[joint];
            auto &values = choices[joint];
            values.clear();
            if (joint_nodes[joint].joint_type == JointType::PRISMATIC ||
                joint_nodes[joint].joint_type == JointType::FIXED ||
                joint_nodes[joint].joint_type == JointType::UNKNOWN) {
                if (solution[joint] >= lower && solution[joint] <= upper)
                    values.push_back(solution[joint]);
            } else if (joint_nodes[joint].joint_type == JointType::REVOLUTE ||
                       joint_nodes[joint].joint_type == JointType::CONTINUOUS) {
                if (!PeriodicValues(lower, upper, solution[joint],
                                    max_solutions, values, joint_nodes[joint],
                                    period)) {
                    holistic_motion::utility::LogWarning(
                        "IK limit expansion exceeds its budget or turn "
                        "precision");
                    return fail();
                }
            }
            if (values.empty()) {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;
        std::size_t combinations = 1;
        const std::size_t remaining = max_solutions - filtered.size();
        for (const auto& values : choices) {
            if (combinations > remaining / values.size()) {
                holistic_motion::utility::LogWarning(
                    "IK limit expansion exceeds {} solutions", max_solutions);
                return fail();
            }
            combinations *= values.size();
        }
        if (combinations > remaining) return fail();  // Also covers zero DOF.
        filtered.reserve(filtered.size() + combinations);
        std::fill(indices.begin(), indices.end(), 0);
        for (std::size_t row = 0; row < combinations; ++row) {
            for (int joint = 0; joint < dof; ++joint)
                candidate[joint] = choices[joint][indices[joint]];
            filtered.push_back(candidate);
            // Mixed-radix counting avoids recursion and intermediate products.
            for (int joint = dof; joint-- > 0;) {
                if (++indices[joint] < choices[joint].size()) break;
                indices[joint] = 0;
            }
        }
    }
    ik_joints = std::move(filtered);
    ik_number = static_cast<int>(ik_joints.size());
    success = ik_number > 0;
    return success;
}

bool IkRtn::WrapToLimitsNear(const std::vector<JointNode> &joint_nodes,
                             const Eigen::VectorXd &reference) {
    if (joint_states.rows() != 0 || joint_states.cols() != 0)
        joint_states.resize(0, 0);
    const auto fail = [this]() {
        ik_joints.clear();
        ik_number = 0;
        success = false;
        return false;
    };
    if (ik_joints.empty() || dof < 0 || reference.size() != dof ||
        joint_nodes.size() < static_cast<size_t>(dof) || !reference.allFinite())
        return fail();
    // The caller may use one of these solutions as its reference. Keep that
    // seed stable while updating and compacting the existing result storage.
    Eigen::VectorXd reference_copy;
    const Eigen::VectorXd *stable_reference = &reference;
    if (std::any_of(ik_joints.begin(), ik_joints.end(),
                    [&](const auto &solution) {
                        return solution.data() == reference.data();
                    })) {
        reference_copy = reference;
        stable_reference = &reference_copy;
    }
    std::size_t kept = 0;
    for (std::size_t index = 0; index < ik_joints.size(); ++index) {
        auto &solution = ik_joints[index];
        if (solution.size() != dof || !solution.allFinite())
            continue;
        bool valid = true;
        for (int joint = 0; joint < dof; ++joint) {
            const auto &node = joint_nodes[joint];
            if (std::isnan(node.lower_limit) || std::isnan(node.upper_limit) ||
                std::isnan(node.lower_limit_set) ||
                std::isnan(node.upper_limit_set))
                return fail();
            const double lower =
                std::max(node.lower_limit, node.lower_limit_set);
            const double upper =
                std::min(node.upper_limit, node.upper_limit_set);
            if (lower > upper)
                return fail();
            if (node.joint_type == JointType::PRISMATIC ||
                node.joint_type == JointType::FIXED ||
                node.joint_type == JointType::UNKNOWN) {
                if (solution[joint] < lower || solution[joint] > upper) {
                    valid = false;
                    break;
                }
                continue;
            }
            if (node.joint_type != JointType::REVOLUTE &&
                node.joint_type != JointType::CONTINUOUS)
                return fail();
            double value;
            const double period = CoordinatePeriod(node);
            if (!(period > 0.0) || !std::isfinite(period))
                return fail();
            if (!NearestPeriodicValue(lower, upper, solution[joint],
                                      (*stable_reference)[joint], value, node,
                                      period)) {
                valid = false;
                break;
            }
            solution[joint] = value;
        }
        if (valid) {
            if (kept != index)
                ik_joints[kept] = std::move(solution);
            ++kept;
        }
    }
    ik_joints.resize(kept);
    ik_number = static_cast<int>(ik_joints.size());
    success = ik_number > 0;
    if (success && ik_number > 1)
        RemoveRepeatedIK();
    return success;
}

bool IkRtn::SortMinMovement(const Eigen::VectorXd &target_joints,
                            std::vector<double> &dist) {
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
