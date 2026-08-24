#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

#include <algorithm>

namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(TrajectoryTrapezium)

template <typename LieGroup>
TrajectoryTrapezium<LieGroup>::TrajectoryTrapezium(
        const std::shared_ptr<PathBase<LieGroup>> &path,
        const std::shared_ptr<TrajectoryConstraints> &constraints,
        const double &vel_init,
        const double &vel_end) {
    holistic_motion::utility::LogDebug("Constructing...");
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

    TrajectorySeg pre_seg;
    double bezier_velocity{0.0};  ///< the velocity of bezier segment
    double bezier_length{0.0};    ///< the length of bezier segment

    double &pre_time = pre_seg.timestamp;  ///< start time
    double &pre_pos = pre_seg.pos;         ///< start position
    double &pre_vel = pre_seg.vel;         ///< start velocity

    // double end_time{0.0}; ///< end time
    double end_pos{0.0};  ///< end position
    double end_vel{0.0};  ///< end velocity

    double max_vel{0.0};  ///< max velocity
    double max_acc{0.0};  ///< max acceleration
    // double max_jerk{0.0}; ///< max jerk (unuse)

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

    std::shared_ptr<PathSegmentBase<LieGroup>> linear_seg;
    std::shared_ptr<PathSegmentBase<LieGroup>> blend_seg;
    std::list<TrajectorySeg> traj_segs, linear_segs;

    // add begin point with init velocity...
    traj_segs.push_back(TrajectorySeg(0, .0, .0, vel_init, .0, .0));

    int i = 0;
    int num_of_segments = this->path_->GetNumOfPathSegments();
    while (i < num_of_segments)  // compute each path segment
    {
        holistic_motion::utility::LogDebug("[{:4.2f}%] computing.....",
                                (double)i / num_of_segments * 100);
        bool is_next_bezier_segment = false;
        bool is_cur_linear_segment = false;

        // get the last segments...
        pre_seg = traj_segs.back();
        linear_seg = this->path_->GetPathSegmentByIndex(i);

        int number_of_bezier_segs = 0;

        if ((is_cur_linear_segment = (linear_seg->GetPathSegType() ==
                                      PathSegType::LinearSeg))) {
            auto sp = linear_seg->GetStartParameter();
            auto tangent = linear_seg->GetTangent(sp);
            max_vel = (std::numeric_limits<double>::max)();
            max_acc = max_vel;
            // Calculate the velocity and acceleration that can be achieved in
            // this linear segment
            for (size_t i = 0; i < this->dof_; ++i) {
                max_vel = std::min(max_vel,
                                   velocity_limits[i] / std::abs(tangent[i]));
                max_acc = std::min(
                        max_acc, acceleration_limits[i] / std::abs(tangent[i]));
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
                    bezier_velocity,
                    _ComputeSegmentMaxSVel(blend_seg, max_vel_list,
                                           max_acc_list, max_jerk_list));
            end_vel = bezier_velocity;
            number_of_bezier_segs++;
            is_next_bezier_segment = true;
            i++;
        }

        if (is_cur_linear_segment) {
            holistic_motion::utility::LogDebug(
                    "Compute trapezium profile: pre_pos:{}, end_pos:{}, "
                    "pre_vel:{}, end_vel:{}, max_vel:{} max_acc:{}, "
                    "pre_time:{}",
                    pre_pos, end_pos, pre_vel, end_vel, max_vel, max_acc,
                    pre_time);

            int seg_no = i - 1 - number_of_bezier_segs;
            bool res = _ComputeTrapeziumProfile(pre_pos, end_pos, pre_vel,
                                                end_vel, max_vel, max_acc,
                                                pre_time, linear_segs, seg_no);
            traj_segs.pop_back();

            if (!res) {
                holistic_motion::utility::LogWarning(
                        "Construct trapezium profile "
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
                holistic_motion::utility::LogWarning(
                        " Construct trapezium profile "
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
            traj_segs.push_back(
                    TrajectorySeg(i - 1, pre_time + bezier_length / end_vel,
                                  pre_pos + bezier_length, end_vel, 0.0, 0.0));
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
double TrajectoryTrapezium<LieGroup>::_ComputeSegmentMaxSVel(
        const std::shared_ptr<PathSegmentBase<LieGroup>> &segment,
        const Eigen::VectorXd &velocity_limits,
        const Eigen::VectorXd &acceleration_limits,
        const Eigen::VectorXd &jerk_limits) {
    // for a general segment segment, especially Bezier5th (whose max
    // tangent/curvature is hard to calculated analytically) use s(t) = kt, k is
    // selected to comply with dq, ddq limits
    double m = std::numeric_limits<double>::max();
    double s = segment->GetStartParameter();
    double ep = segment->GetLength() + s;
    double step = 0.01;
    if (segment->GetLength() <= step) {
        step = segment->GetLength();
    }

    bool quit_loop = false;
    while (!quit_loop) {
        if (s >= ep) {
            s = ep;
            quit_loop = true;
        }
        auto tangent = segment->GetTangent(s);
        auto curvature = segment->GetCurvature(s);
        auto torsion = segment->GetTorsion(s);
        for (size_t i = 0; i < this->dof_; i++) {
            if (std::abs(tangent[i]) > Epsilon) {
                m = std::min(m, velocity_limits[i] / std::abs(tangent[i]));
            }
            if (std::abs(curvature[i]) > Epsilon) {
                m = std::min(m, std::sqrt(acceleration_limits[i] /
                                          std::abs(curvature[i])));
            }
            if (std::abs(torsion[i]) > Epsilon) {
                m = std::min(m, std::pow(jerk_limits[i] / std::abs(torsion[i]),
                                         1.0 / 3.0));
            }
        }
        s += step;
    }

    return m;
}

template <typename LieGroup>
bool TrajectoryTrapezium<LieGroup>::_ComputeTrapeziumProfile(
        const double &q0,
        const double &q1,
        double v0,
        double &v1,
        const double &max_velocity,
        double max_acceleration,
        const double &t0,
        std::list<TrajectorySeg> &traj_segs,
        const int &seg_no) {
    holistic_motion::utility::LogDebug(
            "Compute q0:{}, q1:{}, v0:{}, v1:{}, max_velocity:{} "
            "max_acceleration:{}, seg_no:{}",
            q0, q1, v0, v1, max_velocity, max_acceleration, seg_no);

    traj_segs.clear();
    if (!std::isfinite(q0) || !std::isfinite(q1) || !std::isfinite(v0) ||
        !std::isfinite(v1) || !std::isfinite(max_velocity) ||
        !std::isfinite(max_acceleration) || !std::isfinite(t0) || q1 < q0 ||
        v0 < -Epsilon || v1 < -Epsilon || max_velocity <= 0.0 ||
        max_acceleration <= 0.0) {
        holistic_motion::utility::LogWarning(
                "Invalid trapezium inputs: q0={}, q1={}, v0={}, v1={}, "
                "vmax={}, amax={}, t0={}",
                q0, q1, v0, v1, max_velocity, max_acceleration, t0);
        return false;
    }
    v0 = std::max(0.0, v0);
    v1 = std::max(0.0, v1);
    double h = q1 - q0;
    double delta_h = std::abs((v1 * v1 - v0 * v0) / (2.0 * max_acceleration));

    if (h < delta_h) {
        // Given parameters cannot be satisfied. q1 - q0 is too small.
        if (v1 > v0) {
            // update v1
            v1 = std::sqrt(v0 * v0 + 2.0 * h * max_acceleration);
            traj_segs.push_back(
                    TrajectorySeg(seg_no, t0, q0, v0, max_acceleration, 0.0));
            traj_segs.push_back(
                    TrajectorySeg(seg_no, (t0 + (v1 - v0) / max_acceleration),
                                  q1, v1, 0.0, 0.0));
            return true;
        } else {
            // The segment is too short to decelerate at the nominal scalar
            // acceleration. Build the distance-consistent profile with the
            // required acceleration; EnforceJointLimits subsequently applies
            // one global time scale, preserving velocity continuity across
            // neighboring linear and blend segments while restoring every
            // joint-space derivative limit.
            if (h <= Epsilon) return false;
            const double required_acceleration =
                    (v0 * v0 - v1 * v1) / (2.0 * h);
            if (!std::isfinite(required_acceleration) ||
                required_acceleration <= 0.0) {
                return false;
            }
            const double duration =
                    (v0 - v1) / required_acceleration;
            traj_segs.push_back(
                    TrajectorySeg(seg_no, t0, q0, v0,
                                  -required_acceleration, 0.0));
            traj_segs.push_back(
                    TrajectorySeg(seg_no, t0 + duration,
                                  q1, v1, 0.0, 0.0));
            return true;
        }
    } else {
        // Find the maximum velocity that the trajectory can achieve...
        double v_max_upbound =
                std::sqrt(h * max_acceleration + 0.5 * (v0 * v0 + v1 * v1));
        double v_lim =
                v_max_upbound >= max_velocity ? max_velocity : v_max_upbound;
        const double minimum_endpoint_velocity = std::min(v0, v1);
        const double velocity_tolerance =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max({1.0, std::abs(v_lim), std::abs(v0), std::abs(v1)});
        if (v_lim + velocity_tolerance < minimum_endpoint_velocity) {
            holistic_motion::utility::LogWarning(
                    "The target speed[{}, {}] "
                    "exceeds the limit speed[{}]"
                    " that the trajectory can achieve!",
                    v0, v1, v_lim);
            return false;
        }
        v_lim = std::max(v_lim, minimum_endpoint_velocity);
        double ta = std::abs(v_lim - v0) /
                    max_acceleration;  ///< acceleration period
        double td = std::abs(v1 - v_lim) /
                    max_acceleration;  ///< deceleration period
        double seg_duration = ta + td;
        double tc = 0.0;  ///< constant speed period

        // Determine whether the maximum speed of the trajectory exceeds the
        // limit speed
        if (v_max_upbound > max_velocity) {
            seg_duration = h / v_lim + v_lim / (2 * max_acceleration) *
                                               (std::pow((1 - v0 / v_lim), 2) +
                                                std::pow((1 - v1 / v_lim), 2));
            tc = seg_duration - ta - td;
        }
        holistic_motion::utility::LogDebug("Compute trapezium time, ta:{}, tc:{}, td:{}",
                                ta, tc, td);

        TrajectorySeg current_segment =
                TrajectorySeg(seg_no, t0, q0, v0, 0.0, 0.0);

        if (ta > Epsilon) {  // if there is acceleration period
            current_segment.acc = max_acceleration;
            traj_segs.push_back(current_segment);
            current_segment = _ComputeNextTrajStep(current_segment, ta, 0.0, seg_no);
        }
        if (tc > Epsilon) {  // if there is constant speed period
            current_segment.acc = 0.0;
            traj_segs.push_back(current_segment);
            current_segment = _ComputeNextTrajStep(current_segment, tc, 0.0, seg_no);
        }
        if (td > Epsilon) {  // if there is deceleration period
            current_segment.acc = -max_acceleration;
            traj_segs.push_back(current_segment);
        }
        traj_segs.push_back(
                TrajectorySeg(seg_no, t0 + seg_duration, q1, v1, 0.0, 0.0));
        return true;
    }
}

template <typename LieGroup>
TrajectorySeg TrajectoryTrapezium<LieGroup>::_ComputeNextTrajStep(
        const TrajectorySeg &traj_seg,
        const double &t,
        const double &next_jerk,
        const int &next_seg_no) {
    double acc = traj_seg.acc + traj_seg.jerk * t;
    double vel = traj_seg.vel + traj_seg.acc * t + 0.5 * traj_seg.jerk * t * t;
    double pos = traj_seg.pos + traj_seg.vel * t + 0.5 * traj_seg.acc * t * t +
                 traj_seg.jerk * std::pow(t, 3) / 6.0;
    return TrajectorySeg(next_seg_no, (traj_seg.timestamp + t), pos, vel, acc,
                         next_jerk);
}

}  // namespace robotics
}  // namespace holistic_motion
