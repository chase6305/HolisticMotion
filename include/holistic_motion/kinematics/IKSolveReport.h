#pragma once

#include <vector>

#include <Eigen/Core>

#include "holistic_motion/kinematics/Capabilities.h"

namespace holistic_motion::robotics {

enum class IKSelection { NEAREST, ALL };
enum class IKSolveStatus { SUCCESS, INVALID_INPUT, NO_SOLUTION };

/// Common IK outcome without mutable seed/output arguments. ALL returns the
/// candidates available to the solver from this seed, not a completeness
/// guarantee. Distances follow the solver's configured joint-space metric.
struct IKSolveReport {
    IKSolveStatus status{IKSolveStatus::NO_SOLUTION};
    IKAlgorithm algorithm{IKAlgorithm::CUSTOM};
    IKSelection selection{IKSelection::NEAREST};
    std::vector<Eigen::VectorXd> solutions;
    std::vector<double> distances;

    bool Success() const { return status == IKSolveStatus::SUCCESS; }
};

} // namespace holistic_motion::robotics
