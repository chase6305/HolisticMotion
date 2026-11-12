#include "holistic_motion/planning/NullSpacePlanner.h"

#include <algorithm>
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
                            const Eigen::VectorXd &preferred_direction, int steps,
                            double step_size,
                            std::vector<Eigen::VectorXd> &path) const {
    if (!kinematics_->IsCompatible() || start.size() != 7 ||
        preferred_direction.size() != 7 || !start.allFinite() ||
        !preferred_direction.allFinite() || steps < 1 || !std::isfinite(step_size) ||
        step_size <= 0.0) {
        path.clear();
        return false;
    }

    const double direction_scale = preferred_direction.cwiseAbs().maxCoeff();
    if (direction_scale == 0.0) {
        path.clear();
        return false;
    }
    // Only the direction matters: normalize once before projection so very
    // small preferences do not disappear at the null-space rank threshold.
    const Eigen::VectorXd normalized_direction = preferred_direction / direction_scale;
    Eigen::VectorXd initial = start;
    // Both inputs may refer to rows in the previous output path.
    path.clear();

    const auto &nodes = kinematics_->GetJointNodesView();
    for (Eigen::Index i = 0; i < initial.size(); ++i) {
        const auto &node = nodes[static_cast<std::size_t>(i)];
        if (initial[i] < std::max(node.lower_limit, node.lower_limit_set) ||
            initial[i] > std::min(node.upper_limit, node.upper_limit_set))
            return false;
    }

    SE3d target;
    if (!kinematics_->GetFK(initial, target))
        return false;
    std::vector<Eigen::VectorXd> planned_path;
    planned_path.reserve(static_cast<std::size_t>(steps) + 1);
    planned_path.push_back(std::move(initial));

    Eigen::VectorXd velocity, seed;
    IkRtn solutions;
    for (int i = 0; i < steps; ++i) {
        const auto &current = planned_path.back();
        if (!kinematics_->GetNullSpaceVelocity(current, normalized_direction,
                                               velocity) ||
            velocity.stableNorm() < 1e-10) {
            return false;
        }
        // Normalize after scaling: the ordinary squared norm can overflow for
        // finite directions and silently turn the planned step into zero.
        velocity /= velocity.cwiseAbs().maxCoeff();
        velocity.normalize();
        seed = current + step_size * velocity;
        if ((seed.array() == current.array()).all())
            return false;
        double distance = 0.0;
        if (!kinematics_->GetNearestIK(target, solutions, seed, distance) ||
            solutions.ik_joints.empty()) {
            return false;
        }
        auto &next = solutions.ik_joints.front();
        if ((next.array() == current.array()).all())
            return false;
        planned_path.push_back(std::move(next));
    }
    path = std::move(planned_path);
    return true;
}

} // namespace holistic_motion::robotics::planning
