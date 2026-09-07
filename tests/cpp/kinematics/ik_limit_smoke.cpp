#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "holistic_motion/kinematics/utility.h"

using namespace holistic_motion::robotics;

namespace {
int Fail(int line) {
    std::cerr << "IK limit regression failed at line " << line << '\n';
    return 1;
}
JointNode Revolute(double lower, double upper) {
    return JointNode(SE3d(), Eigen::Vector3d::UnitZ(), JointType::REVOLUTE,
                     lower, upper);
}
bool EmptyFailure(const IkRtn& result) {
    return !result.success && result.ik_number == 0 && result.ik_joints.empty();
}
std::vector<double> Values(const IkRtn& result) {
    std::vector<double> values;
    for (const auto& q : result.ik_joints) values.push_back(q[0]);
    std::sort(values.begin(), values.end());
    return values;
}
}  // namespace

int main() {
    constexpr double tau = 2.0 * M_PI;
    const double infinity = std::numeric_limits<double>::infinity();
    // Inclusive endpoints must not depend on which equivalent seed was used.
    for (double seed : {-tau, 0.0, tau, 2.0 * tau}) {
        IkRtn result;
        result.PushBack(Eigen::VectorXd::Constant(1, seed));
        if (!result.GetLimitsIK({Revolute(0.0, tau)}) ||
            Values(result) != std::vector<double>({0.0, tau}))
            return Fail(__LINE__);
    }
    for (double seed : {-1e100, -1e20, 1e20, 1e100}) {
        IkRtn result;
        result.PushBack(Eigen::VectorXd::Constant(1, seed));
        if (!result.GetLimitsIK({Revolute(-M_PI, M_PI)}) ||
            result.ik_number != 1)
            return Fail(__LINE__);
        const double value = result.ik_joints.front()[0];
        if (value < -M_PI || value > M_PI ||
            std::abs(std::sin(value) - std::sin(seed)) > 1e-10 ||
            std::abs(std::cos(value) - std::cos(seed)) > 1e-10)
            return Fail(__LINE__);
    }

    // Independent bounded turn enumeration for ordinary inputs. Compare sets,
    // including cases with only a single admissible representation.
    for (double seed : {-0.4, 0.0, 0.35}) {
        for (int lower_turn = -3; lower_turn <= 2; ++lower_turn) {
            for (int upper_turn = lower_turn; upper_turn <= 3; ++upper_turn) {
                const double lower = lower_turn * tau - 0.5;
                const double upper = upper_turn * tau + 0.5;
                std::vector<double> expected;
                for (int turn = -4; turn <= 4; ++turn) {
                    const double value = seed + turn * tau;
                    if (value >= lower && value <= upper)
                        expected.push_back(value);
                }
                IkRtn result;
                result.PushBack(Eigen::VectorXd::Constant(1, seed));
                if (!result.GetLimitsIK({Revolute(lower, upper)}))
                    return Fail(__LINE__);
                const auto actual = Values(result);
                if (actual.size() != expected.size()) return Fail(__LINE__);
                for (std::size_t i = 0; i < actual.size(); ++i)
                    if (std::abs(actual[i] - expected[i]) > 1e-12)
                        return Fail(__LINE__);
            }
        }
    }

    // Products are expanded iteratively, with the last coordinate varying
    // fastest. The original in-limit state remains the first representation.
    std::vector<JointNode> joints(3, Revolute(0.0, tau));
    IkRtn result;
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(3);
    result.PushBack(zeros);
    if (!result.GetLimitsIK(joints, 8) || result.ik_number != 8)
        return Fail(__LINE__);
    for (int row = 0; row < 8; ++row)
        for (int joint = 0; joint < 3; ++joint)
            if (result.ik_joints[row][joint] !=
                (((row >> (2 - joint)) & 1) ? tau : 0.0))
                return Fail(__LINE__);
    result.Clear();
    result.PushBack(zeros);
    if (result.GetLimitsIK(joints, 7) || !EmptyFailure(result))
        return Fail(__LINE__);
    result.PushBack(zeros);
    result.PushBack(zeros);
    if (result.GetLimitsIK(joints, 15) || !EmptyFailure(result))
        return Fail(__LINE__);
    // Alternate the single-result path with expansion in one call, then check
    // that the budget is shared by both paths.
    for (std::size_t budget : {9, 10}) {
        result.PushBack(zeros);
        result.PushBack(Eigen::VectorXd::Constant(3, 3.0));
        result.PushBack(zeros);
        const bool success = result.GetLimitsIK(
            std::vector<JointNode>(3, Revolute(-4.0, 4.0)), budget);
        if (budget == 9) {
            if (success || !EmptyFailure(result)) return Fail(__LINE__);
        } else if (!success || result.ik_number != 10 ||
                   !result.ik_joints.front().isZero() ||
                   !result.ik_joints.back().isZero()) {
            return Fail(__LINE__);
        }
        result.Clear();
    }
    result.PushBack(Eigen::VectorXd::Zero(17));
    if (result.GetLimitsIK(std::vector<JointNode>(17, Revolute(0.0, tau))) ||
        !EmptyFailure(result))
        return Fail(__LINE__);

    // Huge finite ranges and zero budgets must fail without a large allocation
    // or a partially populated result. Original fixed/unknown slots do not
    // wrap.
    for (const auto& node : {Revolute(-1e300, 1e300), Revolute(1e100, 1e100)}) {
        result.PushBack(Eigen::VectorXd::Zero(1));
        if (result.GetLimitsIK({node}) || !EmptyFailure(result))
            return Fail(__LINE__);
    }
    for (JointType type :
         {JointType::PRISMATIC, JointType::FIXED, JointType::UNKNOWN}) {
        auto node = Revolute(-10.0, 10.0);
        node.joint_type = type;
        result.PushBack(Eigen::VectorXd::Constant(1, 0.2));
        if (!result.GetLimitsIK({node}) || result.ik_number != 1 ||
            result.ik_joints.front()[0] != 0.2)
            return Fail(__LINE__);
        result.Clear();
    }
    result.PushBack(Eigen::VectorXd::Zero(1));
    if (result.GetLimitsIK({Revolute(-1.0, 1.0)}, 0) || !EmptyFailure(result))
        return Fail(__LINE__);
    result.PushBack(Eigen::VectorXd::Zero(1));
    if (result.GetLimitsIK({}) || !EmptyFailure(result)) return Fail(__LINE__);
    auto nan_node = Revolute(-1.0, 1.0);
    nan_node.lower_limit_set = std::numeric_limits<double>::quiet_NaN();
    result.PushBack(Eigen::VectorXd::Zero(1));
    if (result.GetLimitsIK({nan_node}) || !EmptyFailure(result))
        return Fail(__LINE__);
    result.PushBack(Eigen::VectorXd::Zero(1));
    if (result.GetLimitsIK({Revolute(-infinity, infinity)}) ||
        !EmptyFailure(result))
        return Fail(__LINE__);

    result.PushBack(Eigen::VectorXd::Zero(0));
    if (!result.GetLimitsIK({}, 1) || result.ik_number != 1 ||
        result.ik_joints.front().size() != 0)
        return Fail(__LINE__);
    return 0;
}
