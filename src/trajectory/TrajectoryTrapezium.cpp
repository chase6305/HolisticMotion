#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

#include "TrajectoryBuilder.h"
#include "TrajectoryProfiles.h"

namespace holistic_motion {
namespace robotics {

template <typename LieGroup>
TrajectoryTrapezium<LieGroup>::TrajectoryTrapezium(
    const std::shared_ptr<PathBase<LieGroup>> &path,
    const std::shared_ptr<TrajectoryConstraints> &constraints,
    const double &vel_init, const double &vel_end) {
    detail::TrajectoryBuilder<LieGroup>::Build(*this, path, constraints,
                                               vel_init, vel_end);
}

template <typename LieGroup>
double TrajectoryTrapezium<LieGroup>::_ComputeSegmentMaxSVel(
    const std::shared_ptr<PathSegmentBase<LieGroup>> &segment,
    const Eigen::VectorXd &velocity_limits,
    const Eigen::VectorXd &acceleration_limits,
    const Eigen::VectorXd &jerk_limits) {
    return detail::TrajectoryBuilder<LieGroup>::ComputeSegmentMaxSVel(
        segment, velocity_limits, acceleration_limits, jerk_limits, this->dof_);
}

template <typename LieGroup>
bool TrajectoryTrapezium<LieGroup>::_ComputeTrapeziumProfile(
    const double &q0, const double &q1, double v0, double &v1,
    const double &requested_max_velocity, double max_acceleration,
    const double &t0, std::list<TrajectorySeg> &traj_segs, const int &seg_no) {
    return detail::ComputeTrapeziumProfile(
        q0, q1, v0, v1, requested_max_velocity, max_acceleration, t0, traj_segs,
        seg_no);
}

template <typename LieGroup>
TrajectorySeg TrajectoryTrapezium<LieGroup>::_ComputeNextTrajStep(
    const TrajectorySeg &traj_seg, const double &t, const double &next_jerk,
    const int &next_seg_no) {
    return detail::AdvanceTrajectoryStep(traj_seg, t, next_jerk, next_seg_no);
}

// Members must be defined before explicit class instantiation (Clang).
HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(TrajectoryTrapezium)

} // namespace robotics
} // namespace holistic_motion
