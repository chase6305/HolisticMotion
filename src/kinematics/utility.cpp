
#include "holistic_motion/kinematics/utility.h"
#include "JointPeriod.h"

#include <algorithm>
#include <limits>

namespace holistic_motion {
namespace robotics {


namespace {
constexpr double kTwoPi = 2.0 * M_PI;
constexpr std::size_t kDefaultSolutionLimit = 65536;

struct CoordinateLimits {
    double lower;
    double upper;
    double period;
};

using detail::CoordinatePeriod;
using detail::NearestPeriodicValue;
using detail::PeriodicAnchor;
using detail::SameCoordinateRotation;

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



}  // namespace robotics
}  // namespace holistic_motion
