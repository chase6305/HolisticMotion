#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

#include <algorithm>

#include "TrajectoryIntegration.h"
#include "TrajectorySampling.h"

namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(TrajectoryTrapezium)

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
    double m = std::numeric_limits<double>::max();
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

    while (true) {
        auto tangent = segment->GetTangent(s);
        auto curvature = segment->GetCurvature(s);
        auto torsion = segment->GetTorsion(s);
        for (size_t i = 0; i < this->dof_; i++) {
            if (!std::isfinite(tangent[i]) || !std::isfinite(curvature[i]) ||
                !std::isfinite(torsion[i])) {
                return 0.0;
            }
            if (tangent[i] != 0.0) {
                m = std::min(m, velocity_limits[i] / std::abs(tangent[i]));
            }
            if (curvature[i] != 0.0) {
                m = std::min(m, detail::SquareRootRatio(acceleration_limits[i],
                                                        std::abs(curvature[i])));
            }
            if (torsion[i] != 0.0) {
                m = std::min(
                    m, detail::CubeRootRatio(jerk_limits[i], std::abs(torsion[i])));
            }
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

    return m;
}

template <typename LieGroup>
bool TrajectoryTrapezium<LieGroup>::_ComputeTrapeziumProfile(
    const double &q0, const double &q1, double v0, double &v1,
    const double &requested_max_velocity, double max_acceleration, const double &t0,
    std::list<TrajectorySeg> &traj_segs, const int &seg_no) {
    double max_velocity = requested_max_velocity;
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
    // Curve and line tangents can disagree on the same speed cap by a few
    // ulps. Retain roundoff-equivalent endpoints instead of creating a ramp
    // too short to advance the timestamp. Always compare against the original
    // cap so the allowance cannot accumulate between endpoints.
    constexpr double speed_roundoff = 256.0 * std::numeric_limits<double>::epsilon();
    for (double endpoint : {v0, v1}) {
        if (endpoint > max_velocity &&
            endpoint - requested_max_velocity <= speed_roundoff * endpoint)
            max_velocity = endpoint;
    }
    const double h = q1 - q0;
    if (!std::isfinite(h))
        return false;
    // Factor the squared-speed difference to avoid cancellation and overflow
    // in the individual squares of nonzero endpoint speeds.
    const double average_velocity = 0.5 * v0 + 0.5 * v1;
    const double delta_h = (std::abs(v1 - v0) / max_acceleration) * average_velocity;

    if (h < delta_h) {
        if (h <= 0.0)
            return false;
        double end_velocity = v1;
        double acceleration = max_acceleration;
        if (v1 > v0) {
            // Lower an unreachable end speed. hypot avoids squaring v0;
            // the factored square root also avoids overflowing h * a.
            end_velocity =
                std::min(v1, std::hypot(v0, std::sqrt(h) * std::sqrt(max_acceleration) *
                                                std::sqrt(2.0)));
        }
        // Distance / average speed remains accurate when the velocity change
        // rounds away. The former (v1-v0)/a could become zero on a short path.
        const double duration = h / (0.5 * v0 + 0.5 * end_velocity);
        const double end_time = t0 + duration;
        if (!std::isfinite(duration) || duration <= 0.0 || !std::isfinite(end_time) ||
            end_time <= t0)
            return false;
        if (v0 > v1) {
            // Preserve the requested deceleration endpoint, even on a short
            // positive path. Global limit enforcement subsequently slows the
            // complete trajectory to accommodate this required acceleration.
            acceleration = (v1 - v0) / duration;
            if (!std::isfinite(acceleration))
                return false;
        }
        v1 = end_velocity;
        traj_segs.emplace_back(seg_no, t0, q0, v0, acceleration, 0.0);
        traj_segs.emplace_back(seg_no, end_time, q1, v1, 0.0, 0.0);
        return true;
    } else {
        // Find the maximum velocity that the trajectory can achieve...
        const double squared_peak = h * max_acceleration + 0.5 * (v0 * v0 + v1 * v1);
        const double inverse_sqrt_two = 1.0 / std::sqrt(2.0);
        // Keep ordinary rounding; use a scaled norm if squared intermediates
        // overflow or enter the subnormal range.
        const double v_max_upbound =
            std::isfinite(squared_peak) &&
                    squared_peak >= std::numeric_limits<double>::min()
                ? std::sqrt(squared_peak)
                : std::hypot(std::sqrt(h) * std::sqrt(max_acceleration),
                             std::hypot(v0 * inverse_sqrt_two, v1 * inverse_sqrt_two));
        double v_lim = v_max_upbound >= max_velocity ? max_velocity : v_max_upbound;
        const double minimum_endpoint_velocity = std::min(v0, v1);
        const double velocity_tolerance =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(v_lim), std::abs(v0), std::abs(v1)});
        if (v_lim + velocity_tolerance < minimum_endpoint_velocity) {
            holistic_motion::utility::LogWarning("The target speed[{}, {}] "
                                                 "exceeds the limit speed[{}]"
                                                 " that the trajectory can achieve!",
                                                 v0, v1, v_lim);
            return false;
        }
        v_lim = std::max(v_lim, minimum_endpoint_velocity);
        double ta = std::abs(v_lim - v0) / max_acceleration; ///< acceleration period
        double td = std::abs(v1 - v_lim) / max_acceleration; ///< deceleration period
        double tc = 0.0;                                     ///< constant speed period
        const bool reaches_speed_cap = v_max_upbound >= max_velocity;

        if (!reaches_speed_cap) {
            // Rationalize (v_peak-v_endpoint)/a. Positive elapsed time must
            // survive even when v_peak rounds to a nonzero endpoint speed.
            const double signed_transition_distance = std::copysign(delta_h, v1 - v0);
            ta = (h + signed_transition_distance) / (v_lim + v0);
            td = (h - signed_transition_distance) / (v_lim + v1);
        }

        // Determine whether the maximum speed of the trajectory exceeds the
        // limit speed
        if (reaches_speed_cap) {
            // Average speed gives the distance for either ramp direction.
            // An endpoint can exceed the internal speed cap; that ramp must
            // decelerate toward the cap instead of accelerating away from it.
            const double first_distance = 0.5 * (v0 + v_lim) * ta;
            const double last_distance = 0.5 * (v_lim + v1) * td;
            const double cruise_distance = h - first_distance - last_distance;
            const double distance_tolerance =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max({h, first_distance, last_distance});
            if (!std::isfinite(cruise_distance) ||
                cruise_distance < -distance_tolerance) {
                return false;
            }
            tc = std::max(0.0, cruise_distance) / v_lim;
        }
        if (!std::isfinite(ta) || !std::isfinite(tc) || !std::isfinite(td) ||
            !std::isfinite(t0 + ta + tc + td) || t0 + ta + tc + td <= t0)
            return false;
        holistic_motion::utility::LogDebug(
            "Compute trapezium time, ta:{}, tc:{}, td:{}", ta, tc, td);

        TrajectorySeg current_segment = TrajectorySeg(seg_no, t0, q0, v0, 0.0, 0.0);

        std::list<TrajectorySeg> phases;
        bool collapsed_phase = false;
        const auto append_phase = [&](double duration, double acceleration) {
            collapsed_phase = false;
            if (duration == 0.0)
                return true;
            current_segment.acc = acceleration;
            const auto next =
                _ComputeNextTrajStep(current_segment, duration, 0.0, seg_no);
            if (!std::isfinite(next.timestamp) || !std::isfinite(next.pos) ||
                !std::isfinite(next.vel) || !std::isfinite(next.acc))
                return false;
            if (next.timestamp <= current_segment.timestamp) {
                collapsed_phase = true;
                // Adjacent path segments can disagree on their speed cap by
                // a few ulps. Omit the resulting unrepresentable ramp only
                // when displacement is roundoff relative to this path interval
                // and the speed change is itself roundoff. Keep the current
                // state to preserve the requested start velocity. A larger
                // change must be represented by motion, not omitted here.
                const double speed_roundoff =
                    64.0 * std::numeric_limits<double>::epsilon() *
                    std::max(std::abs(current_segment.vel), std::abs(next.vel));
                const double position_roundoff =
                    64.0 * std::numeric_limits<double>::epsilon() * h;
                return std::abs(next.pos - current_segment.pos) <= position_roundoff &&
                       std::abs(next.vel - current_segment.vel) <= speed_roundoff;
            }
            phases.push_back(current_segment);
            current_segment = next;
            return true;
        };
        const auto append_cruise_transition = [&](double end_position,
                                                  double end_velocity,
                                                  double planned_end_time) {
            // Absorb an unrepresentable ramp into the adjacent cruise rather
            // than discarding its velocity change. The replacement is an
            // independently checked constant-acceleration phase.
            const double displacement = end_position - current_segment.pos;
            const double average_speed = 0.5 * current_segment.vel + 0.5 * end_velocity;
            if (!std::isfinite(displacement) || displacement <= 0.0 ||
                average_speed <= 0.0)
                return false;
            const double end_time =
                current_segment.timestamp + displacement / average_speed;
            const double elapsed = end_time - current_segment.timestamp;
            if (!std::isfinite(end_time) || !std::isfinite(elapsed) || elapsed <= 0.0)
                return false;
            const double acceleration = (end_velocity - current_segment.vel) / elapsed;
            constexpr double roundoff = 64.0 * std::numeric_limits<double>::epsilon();
            double time_budget =
                roundoff * std::max(std::abs(planned_end_time),
                                    std::abs(current_segment.timestamp));
            const double speed_change = std::abs(end_velocity - current_segment.vel);
            // Replacing a cruise and a collapsed ramp by a single ramp
            // changes their average speed. Its expected timing correction
            // depends on that speed change, not on an arbitrary ulp cutoff.
            // Bound the change using the longer of the planned and actual
            // spans; displacement and acceleration remain checked below.
            const double planned_elapsed =
                std::abs(planned_end_time - current_segment.timestamp);
            time_budget += std::max(elapsed, planned_elapsed) *
                           (0.5 * speed_change / average_speed);
            const double distance_error =
                std::abs(average_speed * elapsed - displacement);
            // The stored endpoints are absolute path coordinates. A short
            // interval far from zero also carries their rounding error; an
            // h-only budget can reject motion accurate to one coordinate ulp.
            // Do not scale this allowance by the absolute timestamp: a clock
            // too coarse to reproduce the displacement must still be rejected.
            const double coordinate_roundoff =
                std::numeric_limits<double>::epsilon() *
                std::max(std::abs(current_segment.pos), std::abs(end_position));
            const double distance_budget = roundoff * h + coordinate_roundoff;
            if (!std::isfinite(acceleration) ||
                std::abs(acceleration) > max_acceleration ||
                std::abs(end_time - planned_end_time) > time_budget ||
                !std::isfinite(distance_error) || distance_error > distance_budget)
                return false;
            current_segment.acc = acceleration;
            phases.push_back(current_segment);
            current_segment = TrajectorySeg(seg_no, end_time, end_position,
                                            end_velocity, acceleration, 0.0);
            return true;
        };
        const double first_acceleration =
            reaches_speed_cap ? std::copysign(max_acceleration, v_lim - v0)
                              : max_acceleration;
        const double last_acceleration =
            reaches_speed_cap ? std::copysign(max_acceleration, v1 - v_lim)
                              : -max_acceleration;
        bool has_cruise_phase = false;
        if (!append_phase(ta, first_acceleration)) {
            if (!collapsed_phase || tc <= 0.0)
                return false;
            const double last_distance = (0.5 * v_lim + 0.5 * v1) * td;
            if (!append_cruise_transition(q1 - last_distance, v_lim, t0 + ta + tc))
                return false;
            has_cruise_phase = true;
        } else {
            const auto prefix_size = phases.size();
            if (!append_phase(tc, 0.0))
                return false;
            has_cruise_phase = phases.size() > prefix_size;
        }
        if (!append_phase(td, last_acceleration)) {
            if (!collapsed_phase || !has_cruise_phase)
                return false;
            const double planned_end_time = current_segment.timestamp + td;
            current_segment = phases.back();
            phases.pop_back();
            if (!append_cruise_transition(q1, v1, planned_end_time))
                return false;
        }
        phases.emplace_back(seg_no, current_segment.timestamp, q1, v1, 0.0, 0.0);
        traj_segs.swap(phases);
        return true;
    }
}

template <typename LieGroup>
TrajectorySeg TrajectoryTrapezium<LieGroup>::_ComputeNextTrajStep(
    const TrajectorySeg &traj_seg, const double &t, const double &next_jerk,
    const int &next_seg_no) {
    double acc = traj_seg.acc + traj_seg.jerk * t;
    double vel = traj_seg.vel + traj_seg.acc * t +
                 detail::QuadraticContribution(traj_seg.jerk, t);
    double pos = traj_seg.pos + traj_seg.vel * t +
                 detail::QuadraticContribution(traj_seg.acc, t);
    if (traj_seg.jerk != 0.0)
        pos += detail::CubicJerkDisplacement(traj_seg.jerk, t);
    return TrajectorySeg(next_seg_no, (traj_seg.timestamp + t), pos, vel, acc,
                         next_jerk);
}

} // namespace robotics
} // namespace holistic_motion
