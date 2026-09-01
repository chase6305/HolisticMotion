#include "holistic_motion/planning/NullSpacePlanner.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace holistic_motion::robotics::planning {

NullSpacePlanner::NullSpacePlanner(std::shared_ptr<SRSKinematics> kinematics)
    : kinematics_(std::move(kinematics)) {
  if (!kinematics_) {
    throw std::invalid_argument("kinematics must not be null");
  }
}

bool NullSpacePlanner::Plan(const Eigen::VectorXd &start,
                            const Eigen::VectorXd &preferred_direction,
                            int steps, double step_size,
                            std::vector<Eigen::VectorXd> &path) const {
  path.clear();
  if (!kinematics_->IsCompatible() || start.size() != 7 ||
      preferred_direction.size() != 7 || !start.allFinite() ||
      !preferred_direction.allFinite() || steps < 1 ||
      !std::isfinite(step_size) || step_size <= 0.0) {
    return false;
  }

  SE3d target;
  if (!kinematics_->GetFK(start, target))
    return false;
  std::vector<Eigen::VectorXd> planned_path;
  planned_path.reserve(static_cast<std::size_t>(steps) + 1);
  planned_path.push_back(start);

  Eigen::VectorXd current = start;
  for (int i = 0; i < steps; ++i) {
    Eigen::VectorXd velocity;
    if (!kinematics_->GetNullSpaceVelocity(current, preferred_direction,
                                           velocity) ||
        velocity.norm() < 1e-10) {
      return false;
    }
    Eigen::VectorXd seed = current + step_size * velocity.normalized();
    IkRtn solutions;
    double distance = 0.0;
    if (!kinematics_->GetNearestIK(target, solutions, seed, distance) ||
        solutions.ik_joints.empty()) {
      return false;
    }
    current = solutions.ik_joints.front();
    planned_path.push_back(current);
  }
  path = std::move(planned_path);
  return true;
}

} // namespace holistic_motion::robotics::planning
