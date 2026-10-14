#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"

namespace holistic_motion::robotics::planning {

// Builds a discrete redundant-motion path while keeping the initial TCP pose.
// Collision checking and time parameterization are deliberately separate.
// The start must satisfy both hardware and user joint limits. Failure clears
// the output path; the magnitude of a nonzero preferred direction does not
// affect the planned step, including very small or large finite preferences.
// Each accepted step changes joint coordinates. An unrepresentable seed step
// or an IK correction back to the current configuration fails and clears path.
class NullSpacePlanner {
public:
    explicit NullSpacePlanner(std::shared_ptr<SRSKinematics> kinematics);

    bool Plan(const Eigen::VectorXd& start,
              const Eigen::VectorXd& preferred_direction,
              int steps,
              double step_size,
              std::vector<Eigen::VectorXd>& path) const;

private:
    std::shared_ptr<SRSKinematics> kinematics_;
};

}  // namespace holistic_motion::robotics::planning
