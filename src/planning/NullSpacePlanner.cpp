#include "holistic_motion/planning/NullSpacePlanner.h"

#include <stdexcept>

namespace holistic_motion::robotics::planning {

NullSpacePlanner::NullSpacePlanner(std::shared_ptr<SRSKinematics> kinematics)
    : kinematics_(std::move(kinematics)) {
    if (!kinematics_) {
        throw std::invalid_argument("kinematics must not be null");
    }
}

bool NullSpacePlanner::Plan(
        const Eigen::VectorXd& start,
        const Eigen::VectorXd& preferred_direction,
        int steps,
        double step_size,
        std::vector<Eigen::VectorXd>& path) const {
    if (!kinematics_->IsCompatible() || start.size() != 7 ||
        preferred_direction.size() != 7 || steps < 1 || step_size <= 0.0) {
        return false;
    }

    SE3d target;
    if (!kinematics_->GetFK(start, target)) return false;
    path.clear();
    path.reserve(static_cast<std::size_t>(steps) + 1);
    path.push_back(start);

    Eigen::VectorXd current = start;
    for (int i = 0; i < steps; ++i) {
        Eigen::VectorXd velocity;
        if (!kinematics_->GetNullSpaceVelocity(
                    current, preferred_direction, velocity) ||
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
        path.push_back(current);
    }
    return true;
}

}  // namespace holistic_motion::robotics::planning
