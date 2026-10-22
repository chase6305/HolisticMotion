#include "TrajectoryProfiles.h"
#include "TrajectorySampling.h"

#include <algorithm>
#include <array>

namespace holistic_motion::robotics {

namespace {
constexpr int maximum_speed_iterations =
    std::numeric_limits<double>::max_exponent -
    std::numeric_limits<double>::min_exponent +
    std::numeric_limits<double>::digits;

std::array<double, 2> JerkRampTimes(double velocity_change, double acceleration,
                                    double jerk) {
    const double triangular = detail::SquareRootRatio(velocity_change, jerk);
    const double saturated = acceleration / jerk;
    return triangular < saturated
               ? std::array<double, 2>{triangular, 2.0 * triangular}
               : std::array<double, 2>{saturated, saturated + velocity_change /
                                                                  acceleration};
}

// Refit the two jerk-limited ramps on represented absolute timestamps.
// Rounded-up phase durations leave room under the derivative limits. Solve
// their shared peak speed from the requested displacement, then validate the
// actual stored cubic coefficients before publishing all eight states.
bool FitRoundedProfile(const std::array<double, 7> &durations, double q0,
                       double q1, double v0, double v1, double vmax,
                       double amax, double jmax, double t0, int seg_no,
                       std::list<TrajectorySeg> &result) {
    if (result.size() != 8 || v0 > vmax || v1 > vmax)
        return false;
    using Wide = long double;
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const Wide h = static_cast<Wide>(q1) - q0;
    const Wide position_budget =
        64 * eps * h + eps * std::max(std::abs(q0), std::abs(q1));
    double planned_end = t0;
    for (double duration : durations)
        planned_end += duration;
    const Wide time_budget =
        64 * eps * std::max(std::abs(t0), std::abs(planned_end));
    int extension_phase = 3;
    if (durations[3] == 0.0) {
        extension_phase = 6;
        while (extension_phase > 0 && durations[extension_phase] == 0.0)
            --extension_phase;
    }
    for (int attempt = 0; attempt < 12; ++attempt) {
        const int extra = attempt < 8 ? attempt : 7 - attempt;
        if (extra < 0 && extension_phase != 3)
            continue;
        std::array<double, 8> times{};
        std::array<Wide, 7> dt{};
        times[0] = t0;
        bool valid = true;
        for (int i = 0; i < 7; ++i) {
            times[i + 1] = times[i] + durations[i];
            if (times[i + 1] - times[i] < durations[i])
                times[i + 1] = std::nextafter(
                    times[i + 1], std::numeric_limits<double>::infinity());
            if (i == extension_phase && durations[i] > 0)
                for (int k = 0; k < std::abs(extra); ++k)
                    times[i + 1] = std::nextafter(
                        times[i + 1],
                        extra > 0 ? std::numeric_limits<double>::infinity()
                                  : -std::numeric_limits<double>::infinity());
            dt[i] = times[i + 1] - times[i];
            valid &= std::isfinite(times[i + 1]) &&
                     (durations[i] == 0 ? dt[i] == 0 : dt[i] > 0);
        }
        if (!valid || std::abs(static_cast<Wide>(times.back()) - planned_end) >
                          time_budget)
            continue;
        struct RampWeights {
            Wide duration, effective_acceleration_time, velocity_weight;
        };
        const auto weights = [&](int i) {
            const Wide x = dt[i], y = dt[i + 1], z = dt[i + 2];
            const Wide effective = x / 2 + y + z / 2;
            const Wide weight = effective == 0
                                    ? 0
                                    : (x * x / 6 + x * y / 2 + y * y / 2 +
                                       (x / 2 + y) * z + z * z / 3) /
                                          effective;
            return RampWeights{x + y + z, effective, weight};
        };
        const auto first = weights(0), last = weights(4);
        // A ramp's displacement is v_start * (duration - weight) +
        // v_end * weight. Solve the sum of both ramps and the cruise for
        // their shared speed, retaining either fixed speed if a ramp is absent.
        const Wide fitted =
            first.effective_acceleration_time == 0 ? static_cast<Wide>(v0)
            : last.effective_acceleration_time == 0
                ? static_cast<Wide>(v1)
                : (h - v0 * (first.duration - first.velocity_weight) -
                   v1 * last.velocity_weight) /
                      (first.velocity_weight + dt[3] + last.duration -
                       last.velocity_weight);
        // Tiny endpoint speed differences can be narrower than one clock
        // ulp permits. Project the fitted speed onto the derivative-feasible
        // interval, then independently check the resulting displacement.
        const auto change_limit = [&](int phase, const RampWeights &ramp) {
            return ramp.effective_acceleration_time *
                   std::min(static_cast<Wide>(amax),
                            static_cast<Wide>(jmax) *
                                std::min(dt[phase], dt[phase + 2]));
        };
        const auto speed_interval = [](Wide endpoint, Wide change) {
            Wide low = endpoint - change, high = endpoint + change;
            // Round inward before subtracting a large endpoint again to
            // recover a tiny speed change. Otherwise that cancellation can
            // put a nominal interval endpoint beyond the jerk cap.
            if (low < endpoint)
                low = std::nextafter(low, endpoint);
            if (high > endpoint)
                high = std::nextafter(high, endpoint);
            return std::array<Wide, 2>{low, high};
        };
        const auto first_interval = speed_interval(v0, change_limit(0, first));
        const auto last_interval = speed_interval(v1, change_limit(4, last));
        const Wide minimum_speed =
            std::max({Wide(0), first_interval[0], last_interval[0]});
        const Wide maximum_speed =
            std::min({Wide(vmax), first_interval[1], last_interval[1]});
        if (!std::isfinite(fitted) || minimum_speed > maximum_speed)
            continue;
        const Wide peak = std::clamp(fitted, minimum_speed, maximum_speed);
        const Wide up = first.effective_acceleration_time == 0
                            ? 0
                            : (peak - v0) / first.effective_acceleration_time;
        const Wide down = last.effective_acceleration_time == 0
                              ? 0
                              : (v1 - peak) / last.effective_acceleration_time;
        if (std::abs(up) > amax || std::abs(down) > amax)
            continue;
        std::array<Wide, 8> acc{0, up, up, 0, 0, down, down, 0};
        std::array<double, 8> jerk{};
        for (int i = 0; i < 7; ++i) {
            if (dt[i] == 0) {
                if (acc[i] != acc[i + 1])
                    valid = false;
                continue;
            }
            jerk[i] = static_cast<double>((acc[i + 1] - acc[i]) / dt[i]);
            if (!std::isfinite(jerk[i]) || std::abs(jerk[i]) > jmax)
                valid = false;
        }
        if (!valid)
            continue;
        std::array<Wide, 8> velocity{}, position{};
        velocity[0] = v0;
        position[0] = q0;
        for (int i = 0; i < 7; ++i) {
            velocity[i + 1] = velocity[i] + acc[i] * dt[i] +
                              (acc[i + 1] - acc[i]) * dt[i] / 2;
            position[i + 1] = position[i] + velocity[i] * dt[i] +
                              acc[i] * dt[i] * dt[i] / 2 +
                              (acc[i + 1] - acc[i]) * dt[i] * dt[i] / 6;
        }
        velocity[3] = velocity[4] = peak;
        velocity[7] = v1;
        position[7] = q1;
        std::array<TrajectorySeg, 8> candidate;
        for (int i = 0; i < 8; ++i)
            candidate[i] = TrajectorySeg(
                seg_no, times[i],
                std::clamp(static_cast<double>(position[i]), q0, q1),
                static_cast<double>(velocity[i]), static_cast<double>(acc[i]),
                jerk[i]);
        // The terminal position is prescribed. Carry it through absent
        // trailing phases, then validate the preceding positive phase against
        // that same endpoint. A zero-duration jerk switch must change no state.
        for (int i = 6; i >= 0; --i) {
            if (dt[i] == 0) {
                candidate[i].pos = candidate[i + 1].pos;
                candidate[i].vel = candidate[i + 1].vel;
                candidate[i].acc = candidate[i + 1].acc;
            }
        }
        auto p = candidate.begin(), n = std::next(p);
        for (int i = 0; i < 7; ++i, ++p, ++n) {
            const Wide c2 = p->acc / 2.0, c3 = p->jerk / 6.0, t = dt[i];
            const Wide predicted_q = p->pos + t * (p->vel + t * (c2 + t * c3));
            const Wide predicted_v = p->vel + t * (2 * c2 + t * 3 * c3);
            const Wide predicted_a = 2 * c2 + t * 6 * c3;
            valid &= std::isfinite(n->pos) && std::isfinite(n->vel) &&
                     std::isfinite(n->acc) && n->pos >= p->pos && n->vel >= 0 &&
                     n->vel <= vmax && std::abs(n->acc) <= amax &&
                     std::abs(predicted_q - n->pos) <= position_budget &&
                     std::abs(predicted_v - n->vel) <= 64 * eps * vmax &&
                     std::abs(predicted_a - n->acc) <= 64 * eps * amax;
            const Wide velocity_budget = 64 * eps * vmax;
            valid &= predicted_v >= -velocity_budget &&
                     predicted_v <= vmax + velocity_budget &&
                     std::abs(predicted_a) <= amax * (1 + 64 * eps);
            if (c3 != 0) {
                const Wide stationary = -c2 / (3 * c3);
                if (stationary > 0 && stationary < t) {
                    const Wide speed =
                        p->vel + stationary * (2 * c2 + 3 * c3 * stationary);
                    valid &= speed >= -velocity_budget &&
                             speed <= vmax + velocity_budget;
                }
            }
            if (dt[i] == 0)
                valid &=
                    p->pos == n->pos && p->vel == n->vel && p->acc == n->acc;
        }
        if (valid) {
            // All states are validated; reuse the existing eight list nodes.
            std::copy(candidate.begin(), candidate.end(), result.begin());
            return true;
        }
    }
    return false;
}
} // namespace

bool detail::ComputeDoubleSProfile(const double &q0, const double &q1,
                                   double &start_velocity, double &end_velocity,
                                   const double &requested_max_velocity,
                                   double max_acceleration,
                                   const double &max_jerk, const double &t0,
                                   std::list<TrajectorySeg> &traj_seg,
                                   const int &seg_no,
                                   const bool &allow_concave) {
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
    constexpr double speed_roundoff =
        128.0 * std::numeric_limits<double>::epsilon();
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
        if (first == second)
            return true;
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
        for (int iteration = 0; iteration < maximum_speed_iterations;
             ++iteration) {
            const double middle =
                feasible_speed + 0.5 * (infeasible_speed - feasible_speed);
            if (middle == feasible_speed || middle == infeasible_speed)
                break;
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
                : 0.5 * saturated + std::hypot(0.5 * saturated,
                                               detail::SquareRootRatio(
                                                   q1 - q0, max_acceleration));
        tv = 0.0;
        solved_convex_peak = true;
    }
    if (tv <= 0.0 && !solved_convex_peak && v0 <= max_velocity &&
        v1 <= max_velocity) {
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
        for (int iteration = 0; iteration < maximum_speed_iterations;
             ++iteration) {
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
                const bool triangular =
                    average_acceleration <= 0.5 * max_acceleration;
                const double jerk_duration =
                    triangular
                        ? 0.5 * duration
                        : std::min(
                              0.5 * duration,
                              std::max(max_acceleration / max_jerk,
                                       duration - change / max_acceleration));
                const double acceleration = std::min(
                    max_acceleration, change / (duration - jerk_duration));
                const double jerk =
                    std::min(max_jerk, acceleration / jerk_duration);
                const double achieved_change =
                    (jerk * jerk_duration) * (duration - jerk_duration);
                constexpr double roundoff =
                    64.0 * std::numeric_limits<double>::epsilon();
                if (std::isfinite(duration) && duration > 0.0 &&
                    jerk_duration > 0.0 && std::isfinite(jerk) && jerk > 0.0 &&
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
                (4 * (q1 - q0) - 2 * max_acceleration / max_jerk * (v0 + v1)));
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
        if (ta < 0.0) // if ta < 0.0, there is only a deceleration period
        {
            td = 2 * (q1 - q0) / (v1 + v0);
            jerk_decel_time =
                (max_jerk * (q1 - q0) -
                 std::sqrt(
                     std::abs(max_jerk * (max_jerk * std::pow(q1 - q0, 2) +
                                          std::pow(v1 + v0, 2) * (v1 - v0))))) /
                (max_jerk * (v1 + v0));
            ta = jerk_accel_time = 0.0;
            break;
        }

        if (td < 0.0) // if td < 0.0, there is only a acceleration period
        {
            ta = 2 * (q1 - q0) / (v1 + v0);
            jerk_accel_time =
                (max_jerk * (q1 - q0) -
                 std::sqrt(
                     std::abs(max_jerk * (max_jerk * std::pow(q1 - q0, 2) -
                                          std::pow(v1 + v0, 2) * (v1 - v0))))) /
                (max_jerk * (v1 + v0));
            td = jerk_decel_time = 0.0;
            break;
        }
        // TODO: need to adjust the maximum acceleration ratio to improve the
        // calculation success rate
        if ((ta < 2 * jerk_time) || (td < 2 * jerk_time)) {
            max_acceleration *= lower_scale; // amax is not reached, set a
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
    constexpr double plateau_roundoff =
        64.0 * std::numeric_limits<double>::epsilon();
    if (durations[1] <= plateau_roundoff * std::abs(ta))
        durations[1] = 0.0;
    if (durations[5] <= plateau_roundoff * std::abs(td))
        durations[5] = 0.0;
    const double acceleration_jerk =
        v0 > max_velocity ? -profile_jerk : profile_jerk;
    const double deceleration_jerk =
        v1 > max_velocity ? profile_jerk : -profile_jerk;
    const std::array<double, 7> next_jerks{
        0.0, -acceleration_jerk, 0.0, deceleration_jerk,
        0.0, -deceleration_jerk, 0.0};
    std::list<TrajectorySeg> phases;
    phases.emplace_back(seg_no, t0, q0, v0, acc_init, acceleration_jerk);
    bool rounded_jerk_phase = false;
    for (std::size_t phase = 0; phase < durations.size(); ++phase) {
        const auto &previous = phases.back();
        auto next = detail::AdvanceTrajectoryStep(previous, durations[phase],
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
                (roundoff *
                 std::max({std::abs(v0), std::abs(v1), max_velocity})) /
                    max_acceleration;
            const bool rounding_plateau =
                (phase == 1 || phase == 5) && previous.jerk == 0.0 &&
                std::isfinite(duration_roundoff) &&
                durations[phase] <= duration_roundoff &&
                next.acc == previous.acc &&
                std::abs(next.pos - previous.pos) <= roundoff * (q1 - q0) &&
                same_to_roundoff(next.vel, previous.vel);
            if (!rounding_plateau)
                return false;
            next.pos = previous.pos;
            next.vel = previous.vel;
        }
        // A long elapsed time can quantize a short jerk ramp enough to
        // invalidate its stored acceleration transition. Fit against the
        // actual clock only when that error exceeds the arithmetic budget.
        const double clock_error =
            (next.timestamp - previous.timestamp) - durations[phase];
        rounded_jerk_phase |=
            std::abs(previous.jerk * clock_error) >
            64.0 * std::numeric_limits<double>::epsilon() * max_acceleration;
        // Keep all eight states, including zero-duration jerk transitions:
        // backtracking uses this fixed layout to recover each linear segment.
        phases.push_back(next);
    }
    if (q1 > q0 && phases.back().timestamp <= t0)
        return false;
    if (rounded_jerk_phase) {
        FitRoundedProfile(durations, q0, q1, v0, v1, max_velocity,
                          max_acceleration, max_jerk, t0, seg_no, phases);
    }
    traj_seg.swap(phases);
    start_velocity = v0;
    end_velocity = v1;

    holistic_motion::utility::LogDebug("Finsh to Add trajectory seg!");

    return !need_reduce_v0;
}

bool detail::AlignDoubleSProfileAfter(const TrajectorySeg &previous,
                                      std::list<TrajectorySeg> &profile) {
    if (profile.empty() || !std::isfinite(previous.timestamp) ||
        !std::isfinite(previous.pos) || !std::isfinite(previous.vel)) {
        return false;
    }
    const auto &first = profile.front();
    double gap = first.pos - previous.pos;
    if (!std::isfinite(gap))
        return false;
    // Integration can end a few ulps beyond the intended boundary. Do not
    // turn that roundoff into a backwards constant-speed bridge.
    const double position_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max(std::abs(first.pos), std::abs(previous.pos));
    if (gap < -position_tolerance)
        return false;
    gap = std::max(0.0, gap);
    if (gap > 0.0 && previous.vel <= 0.0)
        return false;
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
    std::array<double, 7> durations{};
    double velocity_cap = 0.0, acceleration_cap = 0.0, jerk_cap = 0.0;
    bool rounded_ramp = false;
    if (profile.size() == 8) {
        std::size_t index = 0;
        for (auto it = profile.begin(); it != profile.end(); ++it, ++index) {
            velocity_cap = std::max(velocity_cap, it->vel);
            acceleration_cap = std::max(acceleration_cap, std::abs(it->acc));
            jerk_cap = std::max(jerk_cap, std::abs(it->jerk));
            const auto next = std::next(it);
            if (next == profile.end())
                break;
            durations[index] = next->timestamp - it->timestamp;
            const double represented =
                timestamps[index + 1] - timestamps[index];
            rounded_ramp |=
                std::abs(it->jerk * (represented - durations[index])) >
                64.0 * std::numeric_limits<double>::epsilon() *
                    std::max(std::abs(it->acc), std::abs(next->acc));
        }
    }
    auto timestamp = timestamps.begin();
    for (auto &step : profile)
        step.timestamp = *timestamp++;
    // Backtracking can move an already fitted ramp onto a coarser clock.
    // Retain its attained derivative caps while fitting the new knot grid.
    if (rounded_ramp && velocity_cap > 0.0 && acceleration_cap > 0.0 &&
        jerk_cap > 0.0) {
        FitRoundedProfile(durations, profile.front().pos, profile.back().pos,
                          profile.front().vel, profile.back().vel, velocity_cap,
                          acceleration_cap, jerk_cap, start,
                          profile.front().seg_no, profile);
    }
    return true;
}

} // namespace holistic_motion::robotics
