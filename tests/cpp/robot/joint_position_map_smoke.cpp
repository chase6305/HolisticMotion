#include <climits>
#include <iostream>

#include "holistic_motion/robot/JointPositionMap.h"

using holistic_motion::robotics::JointPositionMap;

namespace {
bool Rejects(int nq, int nv,
             std::vector<holistic_motion::robotics::JointCoordinate> joints,
             const std::string &expected) {
    try {
        JointPositionMap mapping(nq, nv, std::move(joints));
    } catch (const std::invalid_argument &error) {
        if (error.what() == expected)
            return true;
        std::cerr << "unexpected diagnostic: " << error.what() << '\n';
    }
    return false;
}
} // namespace

int main() {
    // Preserve a free-flyer block, and select slider before continuous angle.
    const JointPositionMap mapping(
        10, 8, {{"slide", 9, 1, 7, 1}, {"turn", 7, 2, 6, 1}});
    Eigen::VectorXd reference(10);
    reference << 1, 2, 3, 0, 0, 0, 1, 1, 0, 0;
    Eigen::VectorXd positions(2);
    positions << 0.3, 0.7;
    const auto full =
        mapping.ConfigurationWithJointPositions(reference, positions);
    if (!full.head(7).isApprox(reference.head(7), 0.0) ||
        !mapping.JointPositions(full).isApprox(positions, 1e-14)) {
        std::cerr
            << "configuration mapping changed context or scalar positions\n";
        return 1;
    }
    const auto selected =
        mapping.JointVelocities(Eigen::VectorXd::LinSpaced(8, 0, 7));
    if (selected[0] != 7 || selected[1] != 6)
        return 1;
    try {
        mapping.JointVelocities(full);
        return 1;
    } catch (const std::invalid_argument &) {
    }
    try {
        JointPositionMap invalid(3, 2,
                                 {{"turn", 0, 2, 0, 1}, {"slide", 1, 1, 1, 1}});
        return 1;
    } catch (const std::invalid_argument &) {
    }
    try {
        JointPositionMap invalid(3, 2,
                                 {{"turn", 0, 2, 0, 1}, {"turn", 0, 2, 0, 1}});
        return 1;
    } catch (const std::invalid_argument &) {
    }
    const JointPositionMap empty(0, 0, {});
    if (empty.JointVelocities(Eigen::VectorXd()).size() != 0)
        return 1;

    // Mixed scalar/continuous blocks can be sparse and retain caller order.
    const JointPositionMap ordered(16, 12,
                                   {{"first", 7, 2, 6, 1},
                                    {"second", 11, 1, 9, 1},
                                    {"third", 14, 2, 11, 1}});
    Eigen::VectorXd ordered_positions(3);
    ordered_positions << -0.4, 0.2, 0.7;
    const auto ordered_full = ordered.ConfigurationWithJointPositions(
        Eigen::VectorXd::Zero(16), ordered_positions);
    if (!ordered.JointPositions(ordered_full)
             .isApprox(ordered_positions, 1e-14))
        return 1;
    // Model sizes do not imply model-sized allocations for a small selection.
    const JointPositionMap large(INT_MAX, INT_MAX,
                                 {{"first", INT_MAX - 4, 2, INT_MAX - 3, 1},
                                  {"second", INT_MAX - 2, 2, INT_MAX - 1, 1}});
    if (large.GetJoints().size() != 2)
        return 1;

    const std::string q_overlap = "joint configuration blocks must not overlap";
    const std::string v_overlap = "joint velocity blocks must not overlap";
    const std::string names = "joint names must be nonempty and unique";
    if (!Rejects(8, 8, {{"a", 0, 2, 0, 1}, {"b", 1, 1, 1, 1}}, q_overlap) ||
        !Rejects(8, 8, {{"a", 0, 1, 1, 1}, {"b", 2, 1, 1, 1}}, v_overlap) ||
        !Rejects(8, 8, {{"a", 0, 2, 0, 1}, {"b", 1, 1, 0, 1}}, q_overlap) ||
        !Rejects(8, 8, {{"a", 0, 2, 0, 1}, {"b", 1, 1, 1, 1}, {"", 7, 1, 7, 1}},
                 q_overlap) ||
        !Rejects(8, 8, {{"a", 0, 2, 0, 1}, {"a", 1, 1, 1, 1}}, names) ||
        !Rejects(8, 8, {{"a", 0, 1, 0, 1}, {"b", 2, 3, 1, 1}},
                 "joint position mapping requires a scalar joint: b") ||
        !Rejects(-1, 8, {{"", 0, 1, 0, 1}},
                 "model nq and nv must be non-negative") ||
        !Rejects(INT_MAX, INT_MAX,
                 {{"a", INT_MAX, 2, 0, 1}, {"b", INT_MIN, 1, 1, 1}},
                 "joint coordinate block is outside model nq/nv: a") ||
        !Rejects(INT_MAX, INT_MAX,
                 {{"a", INT_MAX - 2, 2, 0, 1}, {"b", INT_MAX - 1, 1, 1, 1}},
                 q_overlap))
        return 1;
    return 0;
}
