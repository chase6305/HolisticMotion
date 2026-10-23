#include "holistic_motion/kinematics/utility.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using holistic_motion::robotics::IkRtn;

namespace {
int Fail(int line) {
    std::cerr << "IK result regression failed at line " << line << '\n';
    return 1;
}
using Samples = std::vector<std::vector<double>>;
Samples Reference(Samples input, double tolerance) {
    std::sort(input.begin(), input.end());
    Samples output;
    for (const auto &q : input) {
        const bool duplicate = std::any_of(
            output.begin(), output.end(), [&](const auto &existing) {
                for (std::size_t j = 0; j < q.size(); ++j) {
                    if (tolerance == 0.0
                            ? q[j] != existing[j]
                            : std::abs(q[j] - existing[j]) >= tolerance)
                        return false;
                }
                return true;
            });
        if (!duplicate)
            output.push_back(q);
    }
    return output;
}
bool Check(const Samples &input, double tolerance) {
    IkRtn result;
    for (const auto &q : input) {
        const Eigen::Map<const Eigen::VectorXd> vector(q.data(), q.size());
        if (!result.PushBack(vector))
            return false;
    }
    // Reconcile stale metadata and invalidate a previous matrix view.
    result.ik_number = 0;
    result.joint_states = Eigen::MatrixXd::Ones(2, 3);
    if (!result.RemoveRepeatedIK(tolerance) || !result.success ||
        result.joint_states.size() != 0)
        return false;
    const auto expected = Reference(input, tolerance);
    if (result.ik_number != static_cast<int>(expected.size()) ||
        result.ik_joints.size() != expected.size())
        return false;
    for (std::size_t row = 0; row < expected.size(); ++row) {
        for (std::size_t joint = 0; joint < expected[row].size(); ++joint) {
            if (result.ik_joints[row][joint] != expected[row][joint])
                return false;
        }
    }
    return true;
}
} // namespace

int main() {
    std::array<int, 3> order{0, 1, 2};
    do {
        Samples input;
        for (int index : order)
            input.push_back({0.75 * index});
        if (!Check(input, 1.0))
            return Fail(__LINE__);
    } while (std::next_permutation(order.begin(), order.end()));

    std::mt19937 random(3701);
    for (int dimensions : {0, 1, 2, 7, 16}) {
        for (int trial = 0; trial < 100; ++trial) {
            Samples input(40, std::vector<double>(dimensions));
            for (auto &q : input)
                for (double &value : q)
                    value = (static_cast<int>(random() % 13) - 6) * 0.125;
            for (double tolerance : {0.0, 0.125, 0.1875, 1.0}) {
                if (!Check(input, tolerance))
                    return Fail(__LINE__);
                std::shuffle(input.begin(), input.end(), random);
                if (!Check(input, tolerance))
                    return Fail(__LINE__);
            }
        }
    }
    const double max = std::numeric_limits<double>::max();
    const double tiny = std::numeric_limits<double>::denorm_min();
    if (!Check({{-max}, {max}, {-max}, {0.0}}, max) ||
        !Check({{0.0}, {tiny}, {2 * tiny}, {tiny}}, tiny) ||
        !Check({{0.0}, {1.0}, {std::nextafter(1.0, 0.0)}}, 1.0))
        return Fail(__LINE__);

    for (int invalid = 0; invalid < 7; ++invalid) {
        IkRtn result;
        result.PushBack(Eigen::VectorXd::Zero(2));
        result.success = true;
        result.joint_states = Eigen::MatrixXd::Ones(2, 2);
        double tolerance = 1e-4;
        if (invalid == 0)
            tolerance = -1.0;
        if (invalid == 1)
            tolerance = std::numeric_limits<double>::infinity();
        if (invalid == 2)
            tolerance = std::numeric_limits<double>::quiet_NaN();
        if (invalid == 3)
            result.dof = -1;
        if (invalid == 4)
            result.ik_joints.front().resize(1);
        if (invalid == 5)
            result.ik_joints.front()[0] = tolerance / 0.0;
        if (invalid == 6)
            result.ik_joints.clear();
        if (result.RemoveRepeatedIK(tolerance) || result.success ||
            result.ik_number != 0 || !result.ik_joints.empty() ||
            result.joint_states.size() != 0)
            return Fail(__LINE__);
    }
    // The tolerance itself may alias an existing result coordinate.
    IkRtn aliased;
    for (double value : {1.0, 0.0, 1.5})
        aliased.PushBack(Eigen::VectorXd::Constant(1, value));
    if (!aliased.RemoveRepeatedIK(aliased.ik_joints.front()[0]) ||
        aliased.ik_number != 2 || aliased.ik_joints[0][0] != 0.0 ||
        aliased.ik_joints[1][0] != 1.0)
        return Fail(__LINE__);
    for (bool filter : {false, true}) {
        for (int invalid = 0; invalid < 7; ++invalid) {
            IkRtn result;
            result.PushBack(Eigen::VectorXd::Zero(2));
            result.success = true;
            result.joint_states = Eigen::MatrixXd::Ones(2, 2);
            Eigen::VectorXd reference = Eigen::VectorXd::Zero(2);
            std::vector<double> distances{0.0};
            holistic_motion::robotics::JointFilterManager manager(2);
            if (invalid == 0)
                result.dof = -1;
            if (invalid == 1)
                result.ik_joints.front().resize(1);
            if (invalid == 2)
                result.ik_joints.front()[0] =
                    std::numeric_limits<double>::quiet_NaN();
            if (invalid == 3)
                result.ik_joints.clear();
            if (invalid == 4) {
                reference.resize(1);
                manager.Resize(1);
            }
            if (invalid == 5) {
                distances[0] = std::numeric_limits<double>::infinity();
                manager.Resize(0);
            }
            if (invalid == 6) {
                reference[0] = std::numeric_limits<double>::quiet_NaN();
                manager.Resize(3);
            }
            const bool ok = filter
                                ? result.FilterJointSolutions(manager)
                                : result.SortMinMovement(reference, distances);
            if (ok || result.success || result.ik_number != 0 ||
                !result.ik_joints.empty() || result.joint_states.size() != 0 ||
                (!filter && !distances.empty()))
                return Fail(__LINE__);
        }
    }
    IkRtn sorted;
    for (double value : {3.0, 1.0, 2.0})
        sorted.PushBack(Eigen::VectorXd::Constant(1, value));
    sorted.ik_number = 100; // Public cached count must not cause an overread.
    std::vector<double> distances{1.0, 0.0, 1.0};
    if (!sorted.SortMinMovement(sorted.ik_joints.back(), distances) ||
        sorted.ik_number != 3 || sorted.ik_joints[0][0] != 1.0 ||
        sorted.ik_joints[1][0] != 3.0 || sorted.ik_joints[2][0] != 2.0 ||
        distances != std::vector<double>({0.0, 1.0, 1.0}))
        return Fail(__LINE__);
    using Filter = holistic_motion::robotics::JointFilterManager;
    Filter manager(1);
    manager.SetAllJointFilters(Filter::FilterMode::POSITIVE);
    sorted.PushBack(Eigen::VectorXd::Constant(1, -1.0));
    sorted.ik_number = 0;
    if (!sorted.FilterJointSolutions(manager) || sorted.ik_number != 3 ||
        sorted.ik_joints[0][0] != 1.0 || sorted.ik_joints[1][0] != 3.0 ||
        sorted.ik_joints[2][0] != 2.0)
        return Fail(__LINE__);
    manager.SetAllJointFilters(Filter::FilterMode::NEGATIVE);
    if (sorted.FilterJointSolutions(manager) || sorted.success ||
        sorted.ik_number != 0 || !sorted.ik_joints.empty())
        return Fail(__LINE__);
    IkRtn zero;
    zero.PushBack(Eigen::VectorXd(0));
    zero.PushBack(Eigen::VectorXd(0));
    distances = {1.0, 0.0};
    if (!zero.SortMinMovement(Eigen::VectorXd(0), distances) ||
        !zero.FilterJointSolutions(Filter(0)) || zero.ik_number != 2)
        return Fail(__LINE__);
    return 0;
}
