#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

#include <algorithm>
#include <array>

#include "PathSegmentEvaluation.h"
#include "TrajectoryProfiles.h"
#include "TrajectorySampling.h"

namespace holistic_motion {
namespace robotics {

template <typename LieGroup>
TrajectoryTrapezium<LieGroup>::TrajectoryTrapezium(
    const std::shared_ptr<PathBase<LieGroup>> &path,
    const std::shared_ptr<TrajectoryConstraints> &constraints, const double &vel_init,
    const double &vel_end) {
    holistic_motion::utility::LogDebug("Constructing...");
    if (!path) {
        holistic_motion::utility::LogWarning("The input path is null!");
        return;
    }
    if (!std::isfinite(vel_init) || !std::isfinite(vel_end) || vel_init < 0.0 ||
        vel_end < 0.0) {
        holistic_motion::utility::LogWarning(
            "Initial and final path velocities must be finite and non-negative");
        return;
    }
    this->path_ = path;
    auto path_type = this->path_->GetType();
    this->path_type_ = path_type;

    if (!path->IsValid()) {
        holistic_motion::utility::LogWarning("The input path is not valid!");
        return;
    }
    auto path_length = path->GetLength();
    if (0 == path_length) {
        holistic_motion::utility::LogDebug(
            "The input path length: {}, skip trajectory build!", path_length);
        return;
    } else if (path_length < 0) {
        holistic_motion::utility::LogWarning(
            "The input path is not valid, path length: {}!", path_length);
        return;
    }

    TrajectorySeg pre_seg;
    double bezier_velocity{0.0}; ///< the velocity of bezier segment
    double bezier_length{0.0};   ///< the length of bezier segment

    double &pre_time = pre_seg.timestamp; ///< start time
    double &pre_pos = pre_seg.pos;        ///< start position
    double &pre_vel = pre_seg.vel;        ///< start velocity

    // double end_time{0.0}; ///< end time
    double end_pos{0.0}; ///< end position
    double end_vel{0.0}; ///< end velocity

    double max_vel{0.0}; ///< max velocity
    double max_acc{0.0}; ///< max acceleration
    // double max_jerk{0.0}; ///< max jerk (unuse)

    Eigen::VectorXd velocity_limits, acceleration_limits, jerk_limits;
    if (!this->GetLimitFromConstraintProfile(constraints, velocity_limits,
                                             acceleration_limits, jerk_limits)) {
        return;
    }

    // update the limits
    auto max_vel_list = velocity_limits;
    auto max_acc_list = acceleration_limits;
    auto max_jerk_list = jerk_limits;

    std::shared_ptr<PathSegmentBase<LieGroup>> linear_seg;
    std::shared_ptr<PathSegmentBase<LieGroup>> blend_seg;
    std::list<TrajectorySeg> traj_segs, linear_segs;

    // add begin point with init velocity...
    traj_segs.push_back(TrajectorySeg(0, .0, .0, vel_init, .0, .0));

    int i = 0;
    int num_of_segments = this->path_->GetNumOfPathSegments();
    while (i < num_of_segments) // compute each path segment
    {
        holistic_motion::utility::LogDebug("[{:4.2f}%] computing.....",
                                           (double)i / num_of_segments * 100);
        bool is_next_bezier_segment = false;
        bool is_cur_linear_segment = false;

        // get the last segments...
        pre_seg = traj_segs.back();
        linear_seg = this->path_->GetPathSegmentByIndex(i);

        int number_of_bezier_segs = 0;

        if ((is_cur_linear_segment =
                 (linear_seg->GetPathSegType() == PathSegType::LinearSeg))) {
            auto sp = linear_seg->GetStartParameter();
            auto tangent = linear_seg->GetTangent(sp);
            max_vel = (std::numeric_limits<double>::max)();
            max_acc = max_vel;
            // Calculate the velocity and acceleration that can be achieved in
            // this linear segment
            for (size_t i = 0; i < this->dof_; ++i) {
                max_vel = std::min(max_vel, velocity_limits[i] / std::abs(tangent[i]));
                max_acc =
                    std::min(max_acc, acceleration_limits[i] / std::abs(tangent[i]));
            }

            end_pos = pre_pos + linear_seg->GetLength();
            i++;
            end_vel = i == num_of_segments ? vel_end : 0.0;
        }

        bezier_velocity = std::numeric_limits<double>::max();
        bezier_length = 0.0;

        // to find the bezier curve segment after the linear segment
        while (i < num_of_segments &&
               this->path_->GetPathSegmentByIndex(i)->GetPathSegType() !=
                   PathSegType::LinearSeg) {
            blend_seg = this->path_->GetPathSegmentByIndex(i);
            bezier_length += blend_seg->GetLength();
            // Calculate the velocity and acceleration that can be achieved in
            // this segment
            bezier_velocity = std::min(
                bezier_velocity, _ComputeSegmentMaxSVel(blend_seg, max_vel_list,
                                                        max_acc_list, max_jerk_list));
            end_vel = bezier_velocity;
            number_of_bezier_segs++;
            is_next_bezier_segment = true;
            i++;
        }

        if (is_next_bezier_segment) {
            // The curve's endpoint tangent and its neighboring line can
            // produce slightly different caps after coordinate rounding.
            // Choose the shared join speed from both geometric segments
            // before constructing either profile, preserving continuity
            // without increasing a requested velocity limit.
            if (is_cur_linear_segment)
                end_vel = std::min(end_vel, max_vel);
            if (i < num_of_segments) {
                const auto next_line = this->path_->GetPathSegmentByIndex(i);
                const auto tangent =
                    next_line->GetTangent(next_line->GetStartParameter());
                for (size_t joint = 0; joint < this->dof_; ++joint) {
                    if (tangent[joint] != 0.0)
                        end_vel = std::min(end_vel, velocity_limits[joint] /
                                                       std::abs(tangent[joint]));
                }
            }
        }

        if (is_cur_linear_segment) {
            holistic_motion::utility::LogDebug(
                "Compute trapezium profile: pre_pos:{}, end_pos:{}, "
                "pre_vel:{}, end_vel:{}, max_vel:{} max_acc:{}, "
                "pre_time:{}",
                pre_pos, end_pos, pre_vel, end_vel, max_vel, max_acc, pre_time);

            int seg_no = i - 1 - number_of_bezier_segs;
            bool res =
                _ComputeTrapeziumProfile(pre_pos, end_pos, pre_vel, end_vel, max_vel,
                                         max_acc, pre_time, linear_segs, seg_no);
            traj_segs.pop_back();

            if (!res) {
                holistic_motion::utility::LogWarning("Construct trapezium profile "
                                                     "failed!");
                return;
            }
            if (linear_segs.empty()) {
                holistic_motion::utility::LogWarning(
                    "Trapezium profile produced no trajectory segments");
                return;
            }
            if (linear_segs.back().seg_no == 0 &&
                std::abs(pre_vel - vel_init) > Epsilon) {
                holistic_motion::utility::LogWarning(" Construct trapezium profile "
                                                     "failed!");
                return;
            }

            traj_segs.splice(traj_segs.end(), linear_segs);
        }

        if (is_next_bezier_segment) {
            if (!std::isfinite(end_vel) || end_vel <= 0.0) {
                holistic_motion::utility::LogWarning(
                    "Bezier segment has no finite positive path velocity");
                return;
            }
            pre_seg = traj_segs.back();
            traj_segs.push_back(TrajectorySeg(i - 1, pre_time + bezier_length / end_vel,
                                              pre_pos + bezier_length, end_vel, 0.0,
                                              0.0));
        }
    }
    this->trajectory_segments_ = traj_segs;
    this->trajectory_pspline_ = this->InterpolateToPSpline(traj_segs);
    if (this->trajectory_pspline_->GetKnots().size() < 2 ||
        this->trajectory_pspline_->GetLastTimeStamp() <= 0.0) {
        holistic_motion::utility::LogWarning(
            "Trajectory interpolation produced no positive-duration segments");
        return;
    }
    if (!this->InitializePhasePathSegments())
        return;
    this->valid_ = true;
    if (!this->EnforceJointLimits(velocity_limits, acceleration_limits, jerk_limits)) {
        holistic_motion::utility::LogWarning(
            "Trajectory contains non-finite derivatives");
        return;
    }
    holistic_motion::utility::LogDebug("Constructing succeed!\n");
}

template <typename LieGroup>
double TrajectoryTrapezium<LieGroup>::_ComputeSegmentMaxSVel(
    const std::shared_ptr<PathSegmentBase<LieGroup>> &segment,
    const Eigen::VectorXd &velocity_limits, const Eigen::VectorXd &acceleration_limits,
    const Eigen::VectorXd &jerk_limits) {
    // for a general segment segment, especially Bezier5th (whose max
    // tangent/curvature is hard to calculated analytically) use s(t) = kt, k is
    // selected to comply with dq, ddq limits
    if (!segment)
        return 0.0;
    detail::PathSpeedLimit speed_limit;
    double s = segment->GetStartParameter();
    const double length = segment->GetLength();
    const double ep = s + length;
    if (!std::isfinite(s) || !std::isfinite(length) || !std::isfinite(ep) ||
        length < 0.0 || (length > 0.0 && ep <= s)) {
        return 0.0;
    }
    const double step = std::min(0.01, length);
    // Large coordinate units must not make preliminary cap sampling
    // unbounded. Preserve the ordinary 0.01 grid; long intervals use 4096
    // normalized steps before the separate composed-trajectory limit check.
    constexpr int maximum_intervals = 4096;
    const bool normalized_grid = length > 0.01 * maximum_intervals;
    const double start = s;
    int sample = 0;
    detail::SegmentEvaluationSampler<LieGroup> derivatives(*segment);

    while (true) {
        typename LieGroup::Tangent tangent, curvature, torsion;
        derivatives.Compute(s, tangent, curvature, torsion);
        for (size_t i = 0; i < this->dof_; i++) {
            if (!std::isfinite(tangent[i]) || !std::isfinite(curvature[i]) ||
                !std::isfinite(torsion[i])) {
                return 0.0;
            }
            speed_limit.AddVelocity(velocity_limits[i], tangent[i]);
            speed_limit.AddAcceleration(acceleration_limits[i], curvature[i]);
            speed_limit.AddJerk(jerk_limits[i], torsion[i]);
        }
        if (s == ep)
            break;
        ++sample;
        const double next =
            normalized_grid
                ? (sample == maximum_intervals
                       ? ep
                       : std::min(ep, start + length *
                                                  (sample / double(maximum_intervals))))
                : std::min(ep, s + step);
        // A fixed step may round back to the same large path parameter.
        // Reject the unsampleable interval instead of looping indefinitely.
        if (next <= s)
            return 0.0;
        s = next;
    }

    return speed_limit.Get();
}

template <typename LieGroup>
bool TrajectoryTrapezium<LieGroup>::_ComputeTrapeziumProfile(
    const double &q0, const double &q1, double v0, double &v1,
    const double &requested_max_velocity, double max_acceleration, const double &t0,
    std::list<TrajectorySeg> &traj_segs, const int &seg_no) {
    return detail::ComputeTrapeziumProfile(
        q0, q1, v0, v1, requested_max_velocity, max_acceleration, t0, traj_segs, seg_no);
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
