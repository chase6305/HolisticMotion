#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

#include "PathSegmentEvaluation.h"
#include "TrajectoryIntegration.h"
#include "TrajectorySampling.h"

namespace holistic_motion {
namespace robotics {

namespace {
constexpr int maximum_speed_iterations = std::numeric_limits<double>::max_exponent -
                                         std::numeric_limits<double>::min_exponent +
                                         std::numeric_limits<double>::digits;

std::array<double, 2> JerkRampTimes(double velocity_change, double acceleration,
                                    double jerk) {
    const double triangular = detail::SquareRootRatio(velocity_change, jerk);
    const double saturated = acceleration / jerk;
    return triangular < saturated
               ? std::array<double, 2>{triangular, 2.0 * triangular}
               : std::array<double, 2>{saturated,
                                       saturated + velocity_change / acceleration};
}
} // namespace

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(TrajectoryDoubleS)

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
    detail::SegmentDerivativeSampler<LieGroup> derivatives(*segment);

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
    // Publish adjusted endpoint speeds only with a complete usable profile.
    double v0 = start_velocity;
    double v1 = end_velocity;
    double max_velocity = requested_max_velocity;
    holistic_motion::utility::LogDebug(
        "Compute q0:{}, q1:{}, v0:{}, v1:{}, max_velocity:{} "
        "max_acceleration:{}, max_jerk:{}, seg_no:{}, allow_concave:{}",
        q0, q1, v0, v1, max_velocity, max_acceleration, max_jerk, seg_no,
        allow_concave);

    traj_seg.clear();
    if (!std::isfinite(q0) || !std::isfinite(q1) || !std::isfinite(v0) ||
        !std::isfinite(v1) || !std::isfinite(max_velocity) ||
        !std::isfinite(max_acceleration) || !std::isfinite(max_jerk) ||
        !std::isfinite(t0) || q1 <= q0 || v0 < -Epsilon || v1 < -Epsilon ||
        max_velocity <= 0.0 || max_acceleration <= 0.0 || max_jerk <= 0.0) {
        holistic_motion::utility::LogWarning(
                "Invalid Double-S inputs: q0={}, q1={}, v0={}, v1={}, "
                "vmax={}, amax={}, jmax={}, t0={}",
                q0, q1, v0, v1, max_velocity, max_acceleration, max_jerk, t0);
        return false;
    }
    v0 = std::max(0.0, v0);
    v1 = std::max(0.0, v1);
    // Adjacent curve and line tangents can compute the same speed cap a few
    // ulps apart. Preserve the endpoint speed within a relative roundoff
    // budget instead of needlessly rebuilding the entire preceding profile.
    // The composed trajectory still receives the global joint-limit check.
    constexpr double speed_roundoff = 128.0 * std::numeric_limits<double>::epsilon();
    for (double endpoint : {v0, v1}) {
        if (endpoint > max_velocity &&
            endpoint - requested_max_velocity <= speed_roundoff * endpoint)
            max_velocity = endpoint;
    }
    double acc_init = .0;
    double profile_jerk = max_jerk;

    // duration
    ///< jerk_accel_time constant duration of the jerk in the acceleration phase
    ///< ta acceleration period
    ///< tv constant speed period
    ///< jerk_decel_time constant duration of the jerk in the deceleration phase
    ///< td deceleration period
    ///< delta to compute
    double jerk_accel_time{0.0}, ta{0.0}, tv{0.0};
    double jerk_decel_time{0.0}, td{0.0}, delta{0.0}, jerk_time{0.0};
    // A concave phase can retain an above-cap endpoint, but reducing an
    // infeasible start speed below still requires upstream backtracking.
    bool need_reduce_v0 = !allow_concave && v0 > max_velocity;
    // const double v0_init = v0;

    // const double max_acceleration_const = max_acceleration;
    const double lower_scale = 0.9;

    if (!allow_concave) {
        v0 = v0 > max_velocity ? max_velocity : v0;
        v1 = v1 > max_velocity ? max_velocity : v1;
    }

    // The minimum-distance transition has zero endpoint acceleration and
    // either triangular or acceleration-limited jerk phases. With the lower
    // endpoint speed fixed, this distance grows monotonically with the higher
    // speed, so solve its feasible bound directly instead of relaxing speeds
    // by a fixed percentage until an absolute velocity tolerance is reached.
    const auto reachable = [&](double first, double second) {
        if (first == second) return true;
        const double difference = std::abs(second - first);
        const double duration =
            JerkRampTimes(difference, max_acceleration, max_jerk)[1];
        return (0.5 * first + 0.5 * second) * duration <= q1 - q0;
    };
    if (!reachable(v0, v1)) {
        const bool reduce_start = v0 > v1;
        const double fixed = std::min(v0, v1);
        double feasible_speed = fixed;
        double infeasible_speed = std::max(v0, v1);
        for (int iteration = 0; iteration < maximum_speed_iterations; ++iteration) {
            const double middle =
                    feasible_speed + 0.5 * (infeasible_speed - feasible_speed);
            if (middle == feasible_speed || middle == infeasible_speed) break;
            if (reachable(fixed, middle))
                feasible_speed = middle;
            else
                infeasible_speed = middle;
        }
        if (reduce_start) {
            v0 = feasible_speed;
            need_reduce_v0 = true;
        } else {
            v1 = feasible_speed;
        }
    }

    if (need_reduce_v0 && !allow_concave) {
        holistic_motion::utility::LogDebug(
                "Need to reduce {}th segment V0( init speed )!", seg_no);
    }

    const auto initial_ramp =
        JerkRampTimes(std::abs(max_velocity - v0), max_acceleration, max_jerk);
    jerk_accel_time = initial_ramp[0];
    ta = initial_ramp[1];
    const auto final_ramp =
        JerkRampTimes(std::abs(max_velocity - v1), max_acceleration, max_jerk);
    jerk_decel_time = final_ramp[0];
    td = final_ramp[1];

    // compute constant speed period
    tv = (q1 - q0) / max_velocity - 0.5 * ta * (1 + v0 / max_velocity) -
         0.5 * td * (1 + v1 / max_velocity);

    // The no-cruise formulas below describe a convex speed peak. They do
    // not represent a valley between two above-cap endpoints when there is
    // too little distance to reach the cap. Reject that unsupported boundary
    // condition instead of integrating toward the wrong terminal speed.
    if (allow_concave && v0 > max_velocity && v1 > max_velocity && tv <= 0.0) {
        holistic_motion::utility::LogWarning(
            "Double-S cannot reach the speed cap between above-cap endpoints");
        return false;
    }

    /// < 2.the second case, if tv <= 0.0, there is no constant speed period
    constexpr int maximum_acceleration_reductions = 1000;
    int acceleration_reductions = 0;
    bool solved_convex_peak = false;
    if (tv <= 0.0 && v0 == 0.0 && v1 == 0.0) {
        // Rest-to-rest motion has a closed form. Taking roots before dividing
        // retains finite durations even when the squared/fourth-power
        // intermediates of the expanded formula are not representable.
        const double triangular =
            detail::CubeRootRatio(q1 - q0, max_jerk) / std::cbrt(2.0);
        const double saturated = max_acceleration / max_jerk;
        jerk_accel_time = jerk_decel_time = std::min(triangular, saturated);
        ta = td =
            triangular <= saturated
                ? 2.0 * triangular
                : 0.5 * saturated +
                      std::hypot(0.5 * saturated,
                                 detail::SquareRootRatio(q1 - q0, max_acceleration));
        tv = 0.0;
        solved_convex_peak = true;
    }
    if (tv <= 0.0 && !solved_convex_peak && v0 <= max_velocity && v1 <= max_velocity) {
        // Ramp distance increases monotonically with its peak speed. Solve
        // that bounded scalar problem directly, without fourth powers or
        // repeatedly reducing the available acceleration.
        const double base_velocity = std::max(v0, v1);
        const auto ramp = [&](double endpoint, double increment) {
            const double difference = (base_velocity - endpoint) + increment;
            return JerkRampTimes(difference, max_acceleration, max_jerk);
        };
        const auto distance = [&](double increment) {
            return (0.5 * v0 + 0.5 * base_velocity + 0.5 * increment) *
                       ramp(v0, increment)[1] +
                   (0.5 * v1 + 0.5 * base_velocity + 0.5 * increment) *
                       ramp(v1, increment)[1];
        };
        // Solve the increment, not the absolute peak. A short moving
        // interval may need a positive jerk time even when adding its speed
        // increment to an endpoint rounds back to that endpoint.
        double low = 0.0, high = max_velocity - base_velocity;
        const double length = q1 - q0;
        for (int iteration = 0; iteration < maximum_speed_iterations; ++iteration) {
            const double middle = low + 0.5 * (high - low);
            if (middle == low || middle == high)
                break;
            if (distance(middle) <= length)
                low = middle;
            else
                high = middle;
        }
        const auto first = ramp(v0, low), last = ramp(v1, low);
        jerk_accel_time = first[0];
        ta = first[1];
        jerk_decel_time = last[0];
        td = last[1];
        const double remaining = length - distance(low);
        tv = remaining > 64.0 * std::numeric_limits<double>::epsilon() * length
                 ? remaining / (base_velocity + low)
                 : 0.0;
        if (base_velocity + low == base_velocity) {
            // The speed peak itself is indistinguishable from an endpoint.
            // Fit the endpoint transition across the available distance,
            // reducing jerk if necessary instead of introducing a vanishing
            // opposite ramp and an unrepresentable residual cruise.
            const double duration = length / (0.5 * v0 + 0.5 * v1);
            const double change = std::abs(v1 - v0);
            if (change == 0.0) {
                ta = td = jerk_accel_time = jerk_decel_time = 0.0;
                tv = duration;
            } else {
                const double average_acceleration = change / duration;
                const bool triangular = average_acceleration <= 0.5 * max_acceleration;
                const double jerk_duration =
                    triangular
                        ? 0.5 * duration
                        : std::min(0.5 * duration,
                                   std::max(max_acceleration / max_jerk,
                                            duration - change / max_acceleration));
                const double acceleration =
                    std::min(max_acceleration, change / (duration - jerk_duration));
                const double jerk = std::min(max_jerk, acceleration / jerk_duration);
                const double achieved_change =
                    (jerk * jerk_duration) * (duration - jerk_duration);
                constexpr double roundoff =
                    64.0 * std::numeric_limits<double>::epsilon();
                if (std::isfinite(duration) && duration > 0.0 && jerk_duration > 0.0 &&
                    std::isfinite(jerk) && jerk > 0.0 &&
                    std::isfinite(achieved_change) &&
                    std::abs(achieved_change - change) <= roundoff * change) {
                    profile_jerk = jerk;
                    ta = v1 > v0 ? duration : 0.0;
                    td = v1 < v0 ? duration : 0.0;
                    jerk_accel_time = v1 > v0 ? jerk_duration : 0.0;
                    jerk_decel_time = v1 < v0 ? jerk_duration : 0.0;
                    tv = 0.0;
                }
            }
        }
        solved_convex_peak = true;
    }
    while (tv <= 0.0 && !solved_convex_peak) {
        jerk_accel_time = jerk_decel_time = jerk_time =
                max_acceleration / max_jerk;
        delta = std::sqrt(
                std::pow(max_acceleration, 4) / std::pow(max_jerk, 2) +
                2 * (v0 * v0 + v1 * v1) +
                max_acceleration *
                        (4 * (q1 - q0) -
                         2 * max_acceleration / max_jerk * (v0 + v1)));
        ta = (max_acceleration * max_acceleration / max_jerk - 2 * v0 + delta) /
             (2 * max_acceleration);
        td = (max_acceleration * max_acceleration / max_jerk - 2 * v1 + delta) /
             (2 * max_acceleration);
        // Rationalize the nearly equal subtraction for short motions at
        // nonzero speed. The expanded numerator keeps the small displacement.
        const double acceleration_term =
                max_acceleration * max_acceleration / max_jerk;
        if (2.0 * v0 > acceleration_term) {
            ta = ((v1 - v0) * (v1 + v0 - acceleration_term) +
                  2.0 * max_acceleration * (q1 - q0)) /
                 (max_acceleration * (delta + 2.0 * v0 - acceleration_term));
        }
        if (2.0 * v1 > acceleration_term) {
            td = ((v0 - v1) * (v0 + v1 - acceleration_term) +
                  2.0 * max_acceleration * (q1 - q0)) /
                 (max_acceleration * (delta + 2.0 * v1 - acceleration_term));
        }
        tv = 0.0;

        holistic_motion::utility::LogDebug(
                "There is no constant speed period: jerk_accel_time:{}, ta:{}, "
                "tv:{}, "
                "td:{}, jerk_decel_time: {}, duration:{}",
                jerk_accel_time, ta, tv, td, jerk_decel_time, ta + tv + td);

        // ta or td is negative during the recursion process
        if (ta < 0.0)  // if ta < 0.0, there is only a deceleration period
        {
            td = 2 * (q1 - q0) / (v1 + v0);
            jerk_decel_time =
                    (max_jerk * (q1 - q0) -
                     std::sqrt(std::abs(max_jerk *
                                        (max_jerk * std::pow(q1 - q0, 2) +
                                         std::pow(v1 + v0, 2) * (v1 - v0))))) /
                    (max_jerk * (v1 + v0));
            ta = jerk_accel_time = 0.0;
            break;
        }

        if (td < 0.0)  // if td < 0.0, there is only a acceleration period
        {
            ta = 2 * (q1 - q0) / (v1 + v0);
            jerk_accel_time =
                    (max_jerk * (q1 - q0) -
                     std::sqrt(std::abs(max_jerk *
                                        (max_jerk * std::pow(q1 - q0, 2) -
                                         std::pow(v1 + v0, 2) * (v1 - v0))))) /
                    (max_jerk * (v1 + v0));
            td = jerk_decel_time = 0.0;
            break;
        }
        // TODO: need to adjust the maximum acceleration ratio to improve the
        // calculation success rate
        if ((ta < 2 * jerk_time) || (td < 2 * jerk_time)) {
            max_acceleration *= lower_scale;  // amax is not reached, set a
                                              // lower amax limit;
            if (++acceleration_reductions > maximum_acceleration_reductions ||
                !std::isfinite(max_acceleration) ||
                max_acceleration <= std::numeric_limits<double>::min()) {
                return false;
            }
            holistic_motion::utility::LogDebug("max_acceleration lower to [{}]",
                                               max_acceleration);
        } else {
            break;
        }
    }

    holistic_motion::utility::LogDebug(
            "DoubleS time, jerk_accel_time:{}, ta:{}, tv:{}, "
            "jerk_decel_time:{}, td:{} "
            "delta:{}, jerk_time:{}, duration:{}",
            jerk_accel_time, ta, tv, jerk_decel_time, td, delta, jerk_time,
            ta + tv + td);

    holistic_motion::utility::LogDebug("Begin to add trajectory seg...");

    std::array<double, 7> durations{
            jerk_accel_time, ta - 2.0 * jerk_accel_time, jerk_accel_time, tv,
            jerk_decel_time, td - 2.0 * jerk_decel_time, jerk_decel_time};
    const double phase_tolerance =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(ta), std::abs(td), std::abs(tv)});
    for (double &duration : durations) {
        if (!std::isfinite(duration) || duration < -phase_tolerance)
            return false;
        // Roundoff at a triangular profile can make its zero-acceleration
        // plateau a few ulps negative. Do not create backwards timestamps.
        duration = std::max(0.0, duration);
    }
    // Subtracting twice the jerk-ramp time from a triangular phase can
    // leave a positive residue as well as a negative one. Normalize both
    // signs before integration: a representable residue here can collapse
    // when backtracking rebases the profile at a larger timestamp.
    constexpr double plateau_roundoff = 64.0 * std::numeric_limits<double>::epsilon();
    if (durations[1] <= plateau_roundoff * std::abs(ta))
        durations[1] = 0.0;
    if (durations[5] <= plateau_roundoff * std::abs(td))
        durations[5] = 0.0;
    const double acceleration_jerk = v0 > max_velocity ? -profile_jerk : profile_jerk;
    const double deceleration_jerk = v1 > max_velocity ? profile_jerk : -profile_jerk;
    const std::array<double, 7> next_jerks{
        0.0, -acceleration_jerk, 0.0, deceleration_jerk,
        0.0, -deceleration_jerk, 0.0};
    std::list<TrajectorySeg> phases;
    phases.emplace_back(seg_no, t0, q0, v0, acc_init, acceleration_jerk);
    for (std::size_t phase = 0; phase < durations.size(); ++phase) {
        const auto &previous = phases.back();
        auto next = _ComputeNextTrajStep(previous, durations[phase],
                                        next_jerks[phase], seg_no);
        if (!std::isfinite(next.timestamp) || !std::isfinite(next.pos) ||
            !std::isfinite(next.vel) || !std::isfinite(next.acc)) {
            return false;
        }
        if (next.timestamp <= previous.timestamp &&
            (next.pos != previous.pos || next.vel != previous.vel ||
             next.acc != previous.acc)) {
            // Subtracting twice the jerk time from a triangular ramp can
            // leave a roundoff-sized positive plateau. If that plateau cannot
            // advance time, retain its jerk switch with an unchanged state.
            // Bound both the duration cancellation and the state error;
            // genuine plateaus and all moving jerk phases remain rejected.
            constexpr double roundoff =
                64.0 * std::numeric_limits<double>::epsilon();
            const auto same_to_roundoff = [](double first, double second) {
                return std::abs(first - second) <=
                       roundoff * std::max(std::abs(first), std::abs(second));
            };
            // Plateau time also contains a speed difference divided by a.
            // Include that propagated uncertainty after a backtracked speed
            // update, without letting large world coordinates set the budget.
            const double duration_roundoff =
                roundoff * (phase == 1 ? ta : td) +
                (roundoff * std::max({std::abs(v0), std::abs(v1), max_velocity})) /
                    max_acceleration;
            const bool rounding_plateau =
                (phase == 1 || phase == 5) && previous.jerk == 0.0 &&
                std::isfinite(duration_roundoff) &&
                durations[phase] <= duration_roundoff &&
                next.acc == previous.acc &&
                std::abs(next.pos - previous.pos) <= roundoff * (q1 - q0) &&
                same_to_roundoff(next.vel, previous.vel);
            if (!rounding_plateau) return false;
            next.pos = previous.pos;
            next.vel = previous.vel;
        }
        // Keep all eight states, including zero-duration jerk transitions:
        // backtracking uses this fixed layout to recover each linear segment.
        phases.push_back(next);
    }
    if (q1 > q0 && phases.back().timestamp <= t0) return false;
    traj_seg.swap(phases);
    start_velocity = v0;
    end_velocity = v1;

    holistic_motion::utility::LogDebug("Finsh to Add trajectory seg!");

    return !need_reduce_v0;
}

template <typename LieGroup>
TrajectorySeg TrajectoryDoubleS<LieGroup>::_ComputeNextTrajStep(
    const TrajectorySeg &segment, const double &t, const double &next_jerk,
    const int &next_seg_no) {
    double acc = segment.acc + segment.jerk * t;
    double vel = segment.vel + segment.acc * t +
                 detail::QuadraticContribution(segment.jerk, t);
    double pos = segment.pos + segment.vel * t +
                 detail::QuadraticContribution(segment.acc, t);
    if (segment.jerk != 0.0)
        pos += detail::CubicJerkDisplacement(segment.jerk, t);
    // holistic_motion::utility::LogDebug("TrajectorySeg << seg_no:{}, t:{}, pos:{},
    // vel:{}, acc:{}", next_seg_no, (segment.timestamp + t), pos, vel, acc,
    //                      next_jerk);
    return TrajectorySeg(next_seg_no, (segment.timestamp + t), pos, vel, acc,
                         next_jerk);
}

template <typename LieGroup>
bool TrajectoryDoubleS<LieGroup>::_AlignProfileAfter(
    const TrajectorySeg &previous, std::list<TrajectorySeg> &profile) {
    if (profile.empty() || !std::isfinite(previous.timestamp) ||
        !std::isfinite(previous.pos) || !std::isfinite(previous.vel)) {
        return false;
    }
    const auto &first = profile.front();
    double gap = first.pos - previous.pos;
    if (!std::isfinite(gap)) return false;
    // Integration can end a few ulps beyond the intended boundary. Do not
    // turn that roundoff into a backwards constant-speed bridge.
    const double position_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max(std::abs(first.pos), std::abs(previous.pos));
    if (gap < -position_tolerance) return false;
    gap = std::max(0.0, gap);
    if (gap > 0.0 && previous.vel <= 0.0) return false;
    const double bridge_duration = gap == 0.0 ? 0.0 : gap / previous.vel;
    const double start = previous.timestamp + bridge_duration;
    if (!std::isfinite(start) || (gap > 0.0 && start <= previous.timestamp)) {
        return false;
    }

    // Rebase elapsed times instead of adding a potentially huge offset. Stage
    // all timestamps so a failure near the end leaves the input untouched.
    std::vector<double> timestamps;
    timestamps.reserve(profile.size());
    const TrajectorySeg *last = nullptr;
    for (const auto &step : profile) {
        const double timestamp = start + (step.timestamp - first.timestamp);
        if (!std::isfinite(timestamp) || !std::isfinite(step.pos) ||
            !std::isfinite(step.vel) || !std::isfinite(step.acc) ||
            !std::isfinite(step.jerk)) {
            return false;
        }
        if (last && (step.timestamp < last->timestamp ||
                     timestamp < timestamps.back() ||
                     (timestamp == timestamps.back() &&
                      (step.pos != last->pos || step.vel != last->vel ||
                       step.acc != last->acc)))) {
            return false;
        }
        timestamps.push_back(timestamp);
        last = &step;
    }
    if (profile.back().timestamp > first.timestamp &&
        timestamps.back() <= start)
        return false;
    auto timestamp = timestamps.begin();
    for (auto &step : profile) step.timestamp = *timestamp++;
    return true;
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

}  // namespace robotics
}  // namespace holistic_motion
