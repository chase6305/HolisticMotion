#include <iostream>

#include "holistic_motion/robot/JointPositionMap.h"

using holistic_motion::robotics::JointPositionMap;

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
    return 0;
}
