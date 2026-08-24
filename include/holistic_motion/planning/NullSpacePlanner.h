#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"

namespace holistic_motion::robotics::planning {

// Builds a discrete redundant-motion path while keeping the initial TCP pose.
// Collision checking and time parameterization are deliberately separate.
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
