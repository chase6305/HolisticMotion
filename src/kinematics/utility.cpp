
#include "holistic_motion/kinematics/utility.h"
#include "JointPeriod.h"
#include "holistic_motion/utility/Logging.h"
#include <fmt/ranges.h>

#include <algorithm>
#include <limits>

namespace holistic_motion {
namespace robotics {

bool JointFilterManager::SetJointAngleThresholds(
    const std::vector<double> &thresholds) {
    if (thresholds.size() != angle_thresholds_.size()) {
        holistic_motion::utility::LogWarning(
            "Thresholds size ({}) does not match robot DOF ({})",
            thresholds.size(), angle_thresholds_.size());
        return false;
    }
    if (!std::all_of(thresholds.begin(), thresholds.end(),
                     [](double value) { return std::isfinite(value); })) {
        return false;
    }
    std::copy(thresholds.begin(), thresholds.end(), angle_thresholds_.begin());
    return true;
}

bool JointFilterManager::SetJointAngleThreshold(size_t joint_index,
                                                double threshold) {
    if (!std::isfinite(threshold))
        return false;
    if (joint_index >= angle_thresholds_.size()) {
        holistic_motion::utility::LogWarning(
            "Joint index {} out of range (max {})", joint_index,
            angle_thresholds_.size() - 1);
        return false;
    }
    angle_thresholds_[joint_index] = threshold;
    return true;
}

double JointFilterManager::GetJointAngleThreshold(size_t joint_index) const {
    if (joint_index >= angle_thresholds_.size()) {
        holistic_motion::utility::LogWarning(
            "Joint index {} out of range (max {})", joint_index,
            angle_thresholds_.size() - 1);
        return 0.0;
    }
    return angle_thresholds_[joint_index];
}

std::string _report(const OPWParameters &params) {
    std::string res;
    res += "\t OPWParameters: \n";
    res += fmt::format("\t - [ a1 = {}, a2 = {}, b = {}, c1 = {}, c2 = "
                       "{}, c3 = {}, c4 = {} ]\n"
                       "\t - T_b_ob : SE3d({})\n"
                       "\t - T_oe_e : SE3d({})\n"
                       "\t - offsets: ({})\n"
                       "\t - rotation_directions: ({})\n",
                       params.a1, params.a2, params.b, params.c1, params.c2,
                       params.c3, params.c4,
                       fmt::join(params.T_b_ob.Coeffs(), " , "),
                       fmt::join(params.T_oe_e.Coeffs(), " , "),
                       fmt::join(params.offsets, " , "),
                       fmt::join(params.rotation_directions, " , "));
    return res;
}

std::string _report(const URParameters &params) {
    std::string res;
    res += "\t URParameters: \n";
    res += fmt::format("\t - [ d1 = {}, a2 = {}, a3 = {}, d4 = {}, a5 = "
                       "{}, d6 = {} ]\n"
                       "\t - T_b_ob : SE3d({})\n"
                       "\t - T_oe_e : SE3d({})\n"
                       "\t - offsets: ({})\n"
                       "\t - rotation_directions: ({})\n",
                       params.d1, params.a2, params.a3, params.d4, params.d5,
                       params.d6, fmt::join(params.T_b_ob.Coeffs(), " , "),
                       fmt::join(params.T_oe_e.Coeffs(), " , "),
                       fmt::join(params.offsets, " , "),
                       fmt::join(params.rotation_directions, " , "));
    return res;
}

std::string _report(const FrankaParameters &params) {
    std::string res;
    res += "\t FrankaParameters: \n";
    res +=
        fmt::format("\t - [ d1 = {}, d3 = {}, d5 = {}, d7e = {}, a4 = {}, a7 = "
                    "{}]\n"
                    "\t - [ LL24 = {}, LL46 = {}, L24 = {}, L46 = {}]\n"
                    "\t - [ thetaH46 = {}, theta342 = {}, theta46H = {}]\n",
                    params.d1, params.d3, params.d5, params.d7e, params.a4,
                    params.a7, params.LL24, params.LL46, params.L24, params.L46,
                    params.thetaH46, params.theta342, params.theta46H);
    return res;
}

std::string _report(const AgileDiana7Parameters &params) {
    std::string res;
    res += "\t AgileDiana7Parameters: \n";
    res +=
        fmt::format("\t - [ d1 = {}, d3 = {}, d5 = {}, d7e = {}, a4 = {}, a7 = "
                    "{}]\n"
                    "\t - [ LL24 = {}, LL46 = {}, L24 = {}, L46 = {}]\n"
                    "\t - [ thetaH46 = {}, theta342 = {}, theta46H = {}]\n",
                    params.d1, params.d3, params.d5, params.d7e, params.a4,
                    params.a7, params.LL24, params.LL46, params.L24, params.L46,
                    params.thetaH46, params.theta342, params.theta46H);
    return res;
}

std::string _report(const IkRtn &report) {
    std::string res;
    res += "\t IkRtn: \n";
    res += fmt::format("\t - success: {0} \n"
                       "\t - ik number: {1}\n"
                       "\t - ik joints:\n"
                       "\t   [\n",
                       report.success, report.ik_number);
    for (int i = 0; i < report.ik_number; ++i) {
        if (i == (report.ik_number - 1))
            res +=
                fmt::format("\t\t[{0}]\n", fmt::join(report.ik_joints[i], ","));
        else {
            res += fmt::format("\t\t[{0}],\n",
                               fmt::join(report.ik_joints[i], ","));
        }
    }
    res += fmt::format("\t   ]");
    return res;
}

FrankaParameters::FrankaParameters() {
    d1 = 0.3330;
    d3 = 0.3160;
    d5 = 0.3840;
    // const double d7e = 0.2104;  // use Hand Data (0.107 + 0.1034)
    d7e = 0.107;
    a4 = 0.0825; // a4 = abs(-a5)
    a5 = 0.0825;
    a7 = 0.0880;

    LL24 = 0.10666225;    // a4^2 + d3^2
    LL46 = 0.15426225;    // a4^2 + d5^2
    L24 = 0.326591870689; // sqrt(LL24)
    L46 = 0.392762332715; // sqrt(LL46)

    thetaH46 = 1.35916951803;  // atan(d5/a4);
    theta342 = 1.31542071191;  // atan(d3/a4);
    theta46H = 0.211626808766; // acot(d5/a4); np.pi/2 - atan(d5/a4)

    q_min = {{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
    q_max = {{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};

    q7_start = q_min[6];
    q7_end = q_max[6];

    holistic_motion::utility::LogDebug(
        "\nLoad Franka params:\n\t- [ d1 = {}, d3 = {}, d5 = {}, d7e = "
        "{}, a4 = {}, a7 = {}]"
        "\t - [ LL24 = {}, LL46 = {}, L24 = {}, L46 = {}]\n"
        "\t - [ thetaH46 = {}, theta342 = {}, theta46H = {}]\n",
        d1, d3, d5, d7e, a4, a7, LL24, LL46, L24, L46, thetaH46, theta342,
        theta46H);
}

AgileDiana7Parameters::AgileDiana7Parameters() {
    d1 = 0.2856;
    d3 = 0.4586;
    d5 = 0.4554;
    d7e = 0.1169;
    a4 = 0.0650; // a4 = abs(-a5 - a6)
    a5 = 0.0528;
    a6 = 0.0122;
    a7 = 0.0870;

    LL24 = 0.21453896;        // a4^2 + d3^2
    LL46 = 0.21161416;        // a4^2 + d5^2
    L24 = 0.4631835057512303; // sqrt(LL24)
    L46 = 0.4600153910468649; // sqrt(LL46)

    thetaH46 = 1.4290222431689619; // atan(d5/a4);
    theta342 = 1.429998441264436;  // atan(d3/a4);
    theta46H = 0.1417740836259348; // atan(a4/d5)

    q_min = {{-3.12413, -1.57079, -3.12413, 0, -3.12413, -3.12413, -3.12413}};
    q_max = {{3.12413, 1.57079, 3.12413, 3.05432, 3.12413, 3.12413, 3.12413}};

    q7_start = q_min[6];
    q7_end = q_max[6];

    holistic_motion::utility::LogDebug(
        "\nLoad AgileDiana7 params:\n\t- [ d1 = {}, d3 = {}, d5 = {}, "
        "d7e = "
        "{}, a4 = {}, a7 = {}]\n"
        "\t - [ LL24 = {}, LL46 = {}, L24 = {}, L46 = {}]\n"
        "\t - [ thetaH46 = {}, theta342 = {}, theta46H = {}]\n",
        d1, d3, d5, d7e, a4, a7, LL24, LL46, L24, L46, thetaH46, theta342,
        theta46H);
}

namespace {
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
    for (auto &solution : ik_joints) {
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
            filtered.push_back(std::move(solution));
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
