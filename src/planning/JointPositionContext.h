#pragma once

#include <stdexcept>
#include <utility>

#include "holistic_motion/robot/JointPositionMap.h"

namespace holistic_motion::robotics::planning::detail {

// Prepared scalar selection and preserved model configuration. Publish through
// shared_ptr<const JointPositionContext>; request callbacks only copy a handle.
// Mutable collision/FK workspaces remain separate and owned by each worker.
class JointPositionContext {
  public:
    JointPositionContext(JointPositionMap mapping, Eigen::VectorXd reference)
        : mapping_(std::move(mapping)), reference_(std::move(reference)) {
        if (reference_.size() != mapping_.GetConfigurationSize() ||
            !reference_.allFinite())
            throw std::invalid_argument(
                "reference configuration must be finite and match model nq");
    }
    JointPositionContext(const JointPositionContext &) = delete;
    JointPositionContext &operator=(const JointPositionContext &) = delete;

    const JointPositionMap &GetMapping() const noexcept { return mapping_; }
    Eigen::VectorXd Configuration(const Eigen::VectorXd &positions) const {
        return mapping_.ConfigurationWithJointPositions(reference_, positions);
    }
    Eigen::VectorXd JointVelocities(const Eigen::VectorXd &velocities) const {
        return mapping_.JointVelocities(velocities);
    }

  private:
    const JointPositionMap mapping_;
    const Eigen::VectorXd reference_;
};

} // namespace holistic_motion::robotics::planning::detail
