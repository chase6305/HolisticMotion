#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

#include "TrajectoryBuilder.h"
#include "TrajectoryProfiles.h"

namespace holistic_motion {
namespace robotics {

template <typename LieGroup>
TrajectoryDoubleS<LieGroup>::TrajectoryDoubleS(
    const std::shared_ptr<PathBase<LieGroup>> &path,
    const std::shared_ptr<TrajectoryConstraints> &constraints,
    const double &vel_init, const double &vel_end) {
    detail::TrajectoryBuilder<LieGroup>::Build(*this, path, constraints,
                                               vel_init, vel_end);
}

template <typename LieGroup>
double TrajectoryDoubleS<LieGroup>::_ComputeSegmentMaxSVel(
    const std::shared_ptr<PathSegmentBase<LieGroup>> &segment,
    const Eigen::VectorXd &velocity_limits,
    const Eigen::VectorXd &acceleration_limits,
    const Eigen::VectorXd &jerk_limits) {
    return detail::TrajectoryBuilder<LieGroup>::ComputeSegmentMaxSVel(
        segment, velocity_limits, acceleration_limits, jerk_limits, this->dof_);
}

template <typename LieGroup>
bool TrajectoryDoubleS<LieGroup>::_ComputeDoubleSProfile(
    const double &q0, const double &q1, double &start_velocity,
    double &end_velocity, const double &requested_max_velocity,
    double max_acceleration, const double &max_jerk, const double &t0,
    std::list<TrajectorySeg> &traj_seg, const int &seg_no,
    const bool &allow_concave) {
    return detail::ComputeDoubleSProfile(
        q0, q1, start_velocity, end_velocity, requested_max_velocity,
        max_acceleration, max_jerk, t0, traj_seg, seg_no, allow_concave);
}

template <typename LieGroup>
TrajectorySeg TrajectoryDoubleS<LieGroup>::_ComputeNextTrajStep(
    const TrajectorySeg &segment, const double &t, const double &next_jerk,
    const int &next_seg_no) {
    return detail::AdvanceTrajectoryStep(segment, t, next_jerk, next_seg_no);
}

template <typename LieGroup>
bool TrajectoryDoubleS<LieGroup>::_AlignProfileAfter(
    const TrajectorySeg &previous, std::list<TrajectorySeg> &profile) {
    return detail::AlignDoubleSProfileAfter(previous, profile);
}

template <typename LieGroup>
bool TrajectoryDoubleS<LieGroup>::_ReverseWithMaxJerk(
    std::list<TrajectorySeg> &traj_seg) {
    holistic_motion::utility::LogDebug("_AdjustReverseWithMaxJerk");
    const int linear_jerk_step_count =
        8;  // one linear segment have 8 trajectory steps typically
    if (traj_seg.size() < linear_jerk_step_count) {
        return false;
    }
    // pop up the linear segment traj steps and re-compute
    int seg_no = traj_seg.back().seg_no;
    double q0{0.0}, q1{0.0}, v0{0.0}, v1{0.0}, vmax{0.0}, amax{0.0}, jmax{0.0},
        t0{0.0};
    for (int i = 0; i < linear_jerk_step_count; i++) {
        auto step = traj_seg.end();
        step--;
        if (step->seg_no != seg_no) {
            break;
        }
        if (i == 0) {
            q1 = step->pos;
            v1 = step->vel;
        } else if (i == linear_jerk_step_count - 1) {
            q0 = step->pos;
            v0 = step->vel;
            t0 = step->timestamp;
        }
        vmax = std::max(std::abs(step->vel), vmax);
        amax = std::max(std::abs(step->acc), amax);
        jmax = std::max(std::abs(step->jerk), jmax);
        traj_seg.pop_back();
    }
    // An earlier profile may never have reached its acceleration limit;
    // a pure cruise records only zero acceleration. Recover the configured
    // capacity from its linear geometry instead of treating that observed
    // peak as a new constraint during backtracking.
    if (this->path_ && seg_no >= 0 && seg_no < this->path_->GetNumOfPathSegments() &&
        this->max_acceleration_.size() == static_cast<Eigen::Index>(this->dof_)) {
        const auto segment = this->path_->GetPathSegmentByIndex(seg_no);
        if (segment && segment->GetPathSegType() == PathSegType::LinearSeg) {
            const auto tangent = segment->GetTangent(segment->GetStartParameter());
            double allowed = std::numeric_limits<double>::max();
            for (std::size_t joint = 0; joint < this->dof_; ++joint)
                if (tangent[joint] != 0.0)
                    allowed = std::min(allowed, this->max_acceleration_[joint] /
                                                    std::abs(tangent[joint]));
            if (std::isfinite(allowed) && allowed > 0.0)
                amax = allowed;
        }
    }
    std::list<TrajectorySeg> seg_traj_seg;
    if (_ComputeDoubleSProfile(q0, q1, v0, v1, vmax, amax, jmax, t0,
                               seg_traj_seg, seg_no)) {
        traj_seg.splice(traj_seg.end(), seg_traj_seg);
    } else {
        if (seg_traj_seg.empty() || traj_seg.size() < linear_jerk_step_count) {
            return false;
        }
        traj_seg.back().vel = v0;
        if (_ReverseWithMaxJerk(traj_seg)) {
            if (!_AlignProfileAfter(traj_seg.back(), seg_traj_seg)) return false;
            traj_seg.splice(traj_seg.end(), seg_traj_seg);
        } else {
            return false;
        }
    }
    return true;
}

// Members must be defined before explicit class instantiation (Clang).
HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(TrajectoryDoubleS)

} // namespace robotics
} // namespace holistic_motion
