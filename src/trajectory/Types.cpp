#include "holistic_motion/trajectory/Types.h"

#include <limits>

#include "holistic_motion/robot/Joint.h"

namespace holistic_motion::robotics {

TrajectoryConstraints::TrajectoryConstraints(
        const std::vector<std::shared_ptr<Joint>>& joints) {
    max_velocity_.resize(joints.size());
    max_acceleration_.resize(joints.size());
    max_jerk_.resize(joints.size());
    for (Eigen::Index index = 0;
         index < static_cast<Eigen::Index>(joints.size()); ++index) {
        const auto& joint = joints[static_cast<std::size_t>(index)];
        if (!joint) {
            const double invalid = std::numeric_limits<double>::quiet_NaN();
            max_velocity_[index] = invalid;
            max_acceleration_[index] = invalid;
            max_jerk_[index] = invalid;
            continue;
        }
        max_velocity_[index] = joint->limit.max_velocity;
        max_acceleration_[index] = joint->limit.max_acceleration;
        max_jerk_[index] = joint->limit.max_jerk;
    }
}

}  // namespace holistic_motion::robotics
