#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

#include "PathSegmentEvaluation.h"
#include "TrajectoryProfiles.h"
#include "TrajectorySampling.h"

namespace holistic_motion {
namespace robotics {

template <typename LieGroup>
TrajectoryDoubleS<LieGroup>::TrajectoryDoubleS(
        const std::shared_ptr<PathBase<LieGroup>> &path,
        const std::shared_ptr<TrajectoryConstraints> &constraints,
        const double &vel_init,
        const double &vel_end) {
    holistic_motion::utility::LogDebug("Constructing TrajectoryDoubleS...");
    if (!path) {
        holistic_motion::utility::LogWarning("The input path is null!");
        return;
    }
    if (!std::isfinite(vel_init) || !std::isfinite(vel_end) ||
        vel_init < 0.0 || vel_end < 0.0) {
        holistic_motion::utility::LogWarning(
                "Initial and final path velocities must be finite and non-negative");
        return;
    }
    this->path_ = path;
    auto path_type = this->path_->GetType();
    this->path_type_ = path_type;
    if (!(path_type == PathType::Bezier5thJointSpace ||
          path_type == PathType::Bezier5thCartesianSpace)) {
        holistic_motion::utility::LogWarning(
                "Trajectory DoubleS profile is used for "
                "Bezier5thJointSpace "
                "or Bezier5thCartesianSpace, but preset path type is {}",
                to_underlying_type(path_type));
        return;
    }

    if (!path->IsValid()) {
        holistic_motion::utility::LogWarning("The input path is not valid!");
        return;
    }
    auto path_length = path->GetLength();
    if (0 == path_length) {
        holistic_motion::utility::LogDebug(
                "The input path length: {}, skip trajectory build!",
                path_length);
        return;
    } else if (path_length < 0) {
        holistic_motion::utility::LogWarning(
                "The input path is not valid, path length: {}!", path_length);
        return;
    }

    double bezier_velocity{0.0};  ///< the velocity of bezier segment
    double bezier_length{0.0};    ///< the length of bezier segment

    TrajectorySeg pre_seg;
    double &pre_time = pre_seg.timestamp;  ///< start time
    double &pre_pos = pre_seg.pos;         ///< start position
    double &pre_vel = pre_seg.vel;         ///< start velocity
    double end_pos{0.0};                   // end position
    double end_vel{0.0};                   // end velocity

    double max_vel{0.0};   ///< max velocity
    double max_acc{0.0};   ///< max acceleration
    double max_jerk{0.0};  ///< max jerk

    std::shared_ptr<PathSegmentBase<LieGroup>> linear_seg;
    std::shared_ptr<PathSegmentBase<LieGroup>> blend_seg;
    std::list<TrajectorySeg> traj_segs, linear_segs;

    Eigen::VectorXd velocity_limits, acceleration_limits, jerk_limits;
    if (!this->GetLimitFromConstraintProfile(
                constraints, velocity_limits, acceleration_limits,
                jerk_limits)) {
        return;
    }

    // update the limits
    auto max_vel_list = velocity_limits;
    auto max_acc_list = acceleration_limits;
    auto max_jerk_list = jerk_limits;

    traj_segs.push_back(TrajectorySeg(0, .0, .0, vel_init, .0, .0));

    int i = 0;
    int num_of_segments = this->path_->GetNumOfPathSegments();
    holistic_motion::utility::LogDebug("Num of path segments: {}", num_of_segments);
    while (i < num_of_segments) {
        holistic_motion::utility::LogDebug("[{:4.2f}%] computing.....",
                                (double)i / num_of_segments * 100);
        bool is_cur_linear_segment = false;
        bool is_next_bezier_segment = false;

        linear_seg = this->path_->GetPathSegmentByIndex(i);

        double sp_of_linear_seg = linear_seg->GetStartParameter();
        pre_seg = traj_segs.back();

        int bezier_segs_number = 0;
        if ((is_cur_linear_segment = (linear_seg->GetPathSegType() ==
                                      PathSegType::LinearSeg))) {
            auto tangent = linear_seg->GetTangent(sp_of_linear_seg);
            max_vel = (std::numeric_limits<double>::max)();
            max_acc = max_vel;
            max_jerk = max_vel;
            // save the max limits
            for (size_t i = 0; i < this->dof_; ++i) {
                max_vel = std::min(max_vel,
                                   velocity_limits[i] / std::abs(tangent[i]));
                max_acc = std::min(
                        max_acc, acceleration_limits[i] / std::abs(tangent[i]));
                max_jerk = std::min(max_jerk,
                                    jerk_limits[i] / std::abs(tangent[i]));
            }

            end_pos = pre_pos + linear_seg->GetLength();
            i++;
            end_vel = (i == num_of_segments) ? vel_end : 0.0;
        }

        // need to find the next Bezier5th svel if possible
        bezier_velocity = std::numeric_limits<double>::max();
        bezier_length = 0.0;

        // to find the bezier curve segment after the linear segment
        while (i < num_of_segments &&
               this->path_->GetPathSegmentByIndex(i)->GetPathSegType() !=
                       PathSegType::LinearSeg) {
            blend_seg = this->path_->GetPathSegmentByIndex(i);
            bezier_length += blend_seg->GetLength();
            bezier_velocity = std::min(
                    bezier_velocity,
                    _ComputeSegmentMaxSVel(blend_seg, max_vel_list,
                                           max_acc_list, max_jerk_list));
            end_vel = bezier_velocity;
            is_next_bezier_segment = true;
            bezier_segs_number++;
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
                    "DoubleS profile:[ pre_pos:{}, end_pos:{}, "
                    "pre_vel:{}, end_vel:{}, max_vel:{} max_acc:{}, "
                    "max_jerk:{}, pre_time:{} ]",
                    pre_pos, end_pos, pre_vel, end_vel, max_vel, max_acc,
                    max_jerk, pre_time);
            bool allow_concave = false;
            int seg_no = i - 1 - bezier_segs_number;
            if (seg_no == 0 || i == num_of_segments) {  // initial/final segment
                allow_concave = true;
            }

            // compute doubleS profile
            bool res = _ComputeDoubleSProfile(
                    pre_pos, end_pos, pre_vel, end_vel, max_vel, max_acc,
                    max_jerk, pre_time, linear_segs, seg_no, allow_concave);

            if (linear_segs.empty()) {
                holistic_motion::utility::LogWarning(
                        "Double-S profile produced no trajectory segments");
                return;
            }

            if (linear_segs.back().seg_no == 0 &&
                std::abs(pre_vel - vel_init) > Epsilon) {
                holistic_motion::utility::LogWarning(
                        "Desired initial velocity {} is "
                        "not in limits [0.0, {}]",
                        vel_init, pre_vel);
                return;
            }

            // compute doubleS profile failed for pre_vel is too large
            traj_segs.pop_back();

            if (!res) {
                if (traj_segs.empty()) {
                    holistic_motion::utility::LogWarning(
                        "Initial velocity cannot be reached without an earlier "
                        "segment");
                    return;
                }
                traj_segs.back().vel = pre_vel;
                holistic_motion::utility::LogDebug(
                        "Compute doubleS profile with reverse max jerk");
                if (!_ReverseWithMaxJerk(traj_segs)) {
                    holistic_motion::utility::LogWarning(
                            "Compute doubleS profile failed!");
                    return;
                }
                if (!_AlignProfileAfter(traj_segs.back(), linear_segs)) return;
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
            traj_segs.push_back(
                    TrajectorySeg(i - 1, pre_time + bezier_length / end_vel,
                                  pre_pos + bezier_length, end_vel, 0.0, 0.0));
        }
    }
    holistic_motion::utility::LogDebug("[100%] computed complete! ");

    this->trajectory_segments_ = traj_segs;
    this->trajectory_pspline_ = this->InterpolateToPSpline(traj_segs);
    if (this->trajectory_pspline_->GetKnots().size() < 2 ||
        this->trajectory_pspline_->GetLastTimeStamp() <= 0.0) {
        holistic_motion::utility::LogWarning(
                "Trajectory interpolation produced no positive-duration segments");
        return;
    }
    if (!this->InitializePhasePathSegments()) return;
    this->valid_ = true;
    if (!this->EnforceJointLimits(
                velocity_limits, acceleration_limits, jerk_limits)) {
        holistic_motion::utility::LogWarning(
                "Trajectory contains non-finite derivatives");
        return;
    }
    holistic_motion::utility::LogDebug("Constructing succeed!\n");
}

template <typename LieGroup>
double TrajectoryDoubleS<LieGroup>::_ComputeSegmentMaxSVel(
        const std::shared_ptr<PathSegmentBase<LieGroup>> &segment,
        const Eigen::VectorXd &velocity_limits,
        const Eigen::VectorXd &acceleration_limits,
        const Eigen::VectorXd &jerk_limits) {
    // for a general segment segment, especially Bezier5th (whose max
    // tangent/curvature is hard to calculated analytically) use s(t) = kt, k is
    // selected to comply with dq, ddq limits
    if (!segment) return 0.0;
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
        if (s == ep) break;
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
        if (next <= s) return 0.0;
        s = next;
    }

    return speed_limit.Get();
}

template <typename LieGroup>
bool TrajectoryDoubleS<LieGroup>::_ComputeDoubleSProfile(
    const double &q0, const double &q1, double &start_velocity, double &end_velocity,
    const double &requested_max_velocity, double max_acceleration,
    const double &max_jerk, const double &t0, std::list<TrajectorySeg> &traj_seg,
    const int &seg_no, const bool &allow_concave) {
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

}  // namespace robotics
}  // namespace holistic_motion
