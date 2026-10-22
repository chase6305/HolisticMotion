#include "TrajectoryProfiles.h"

#include <algorithm>
#include <array>

namespace holistic_motion::robotics {

namespace {
// Try a pair of constant-acceleration phases on the represented clock only
// after the ordinary collapsed-ramp repairs fail. Their shared velocity is
// free; both requested endpoint velocities and positions remain fixed.
bool FitRoundedClockProfile(double q0, double q1, double v0, double v1,
                            double velocity_cap, double acceleration_cap,
                            double start, double ta, double tc, double td,
                            int segment, std::list<TrajectorySeg> &output) {
    const double length = q1 - q0;
    const double ramp_end = start + ta;
    const double cruise_end = ramp_end + tc;
    const double planned_end = cruise_end + td;
    const bool first_collapsed = ta > 0.0 && ramp_end == start;
    const bool last_collapsed = td > 0.0 && planned_end == cruise_end;
    if ((!first_collapsed && !last_collapsed) || tc <= 0.0 ||
        !std::isfinite(length) || length <= 0.0 ||
        !std::isfinite(planned_end) || planned_end <= start)
        return false;

    double planned_middle = first_collapsed ? cruise_end : ramp_end;
    if (planned_middle <= start || planned_middle >= planned_end)
        planned_middle = start + 0.5 * (planned_end - start);
    const auto neighbours = [](double value) {
        const double up =
            std::nextafter(value, std::numeric_limits<double>::infinity());
        const double down =
            std::nextafter(value, -std::numeric_limits<double>::infinity());
        std::array<double, 9> result{value, up, down};
        for (std::size_t step = 2; step <= 4; ++step) {
            result[2 * step - 1] = std::nextafter(
                result[2 * step - 3], std::numeric_limits<double>::infinity());
            result[2 * step] = std::nextafter(
                result[2 * step - 2], -std::numeric_limits<double>::infinity());
        }
        return result;
    };
    constexpr double roundoff = 64.0 * std::numeric_limits<double>::epsilon();
    const double time_budget =
        roundoff * std::max(std::abs(start), std::abs(planned_end));
    const double maximum_speed = std::max({velocity_cap, v0, v1});
    const double velocity_budget = roundoff * maximum_speed;
    const auto ends = neighbours(planned_end);
    const auto middles = neighbours(planned_middle);
    // Preserve the two-step search's first successful result. Only if it
    // fails, inspect the remaining pairs within four floating-point steps.
    // An endpoint-peak retry must retain the old search, so it cannot
    // preempt a successful two-step fit of the original cap excursion.
    const unsigned passes = velocity_cap > std::max(v0, v1) ? 2 : 1;
    for (unsigned pass = 0; pass < passes; ++pass) {
        const std::size_t count = pass == 0 ? 5 : 9;
        for (std::size_t end_index = 0; end_index < count; ++end_index) {
            const double end = ends[end_index];
            if (!std::isfinite(end) ||
                std::abs(end - planned_end) > time_budget)
                continue;
            for (std::size_t middle_index = 0; middle_index < count;
                 ++middle_index) {
                if (count == 9 && end_index < 5 && middle_index < 5)
                    continue;
                const double middle = middles[middle_index];
                const double first = middle - start;
                const double second = end - middle;
                const double total = end - start;
                if (!std::isfinite(first) || !std::isfinite(second) ||
                    !std::isfinite(total) || first <= 0.0 || second <= 0.0)
                    continue;
                // h = (v0 + vm) * first / 2 + (vm + v1) * second / 2.
                // Normalize durations before weighting endpoint speeds, and
                // avoid doubling a potentially large mean speed before
                // cancellation.
                const double mean = length / total;
                const double weighted_ends =
                    v0 * (first / total) + v1 * (second / total);
                const double fitted = mean + (mean - weighted_ends);
                const std::array<double, 3> speeds{
                    fitted,
                    std::nextafter(fitted,
                                   std::numeric_limits<double>::infinity()),
                    std::nextafter(fitted,
                                   -std::numeric_limits<double>::infinity())};
                for (double speed : speeds) {
                    if (!std::isfinite(speed) || speed < 0.0 ||
                        speed > maximum_speed)
                        continue;
                    const double first_acceleration = (speed - v0) / first;
                    const double second_acceleration = (v1 - speed) / second;
                    if (!std::isfinite(first_acceleration) ||
                        !std::isfinite(second_acceleration) ||
                        std::abs(first_acceleration) > acceleration_cap ||
                        std::abs(second_acceleration) > acceleration_cap ||
                        2.0 * (first_acceleration / 2.0) !=
                            first_acceleration ||
                        2.0 * (second_acceleration / 2.0) !=
                            second_acceleration)
                        continue;
                    // The spline stores acceleration / 2. Reject a lossy
                    // subnormal coefficient instead of validating another
                    // motion.
                    const double first_distance =
                        v0 * first + detail::QuadraticContribution(
                                         first_acceleration, first);
                    const double position = q0 + first_distance;
                    const double second_distance =
                        speed * second + detail::QuadraticContribution(
                                             second_acceleration, second);
                    const double first_speed = v0 + first_acceleration * first;
                    const double last_speed =
                        speed + second_acceleration * second;
                    if (!std::isfinite(position) ||
                        !std::isfinite(first_distance) ||
                        !std::isfinite(second_distance) ||
                        !std::isfinite(first_speed) ||
                        !std::isfinite(last_speed) || position < q0 ||
                        position > q1 || first_speed < 0.0 ||
                        last_speed < 0.0 || first_speed > maximum_speed ||
                        last_speed > maximum_speed ||
                        std::abs(first_speed - speed) > velocity_budget ||
                        std::abs(last_speed - v1) > velocity_budget)
                        continue;
                    const double first_budget =
                        roundoff * length +
                        std::numeric_limits<double>::epsilon() *
                            std::max(std::abs(q0), std::abs(position));
                    const double second_budget =
                        roundoff * length +
                        std::numeric_limits<double>::epsilon() *
                            std::max(std::abs(position), std::abs(q1));
                    if (std::abs(first_distance - (position - q0)) >
                            first_budget ||
                        std::abs(second_distance - (q1 - position)) >
                            second_budget)
                        continue;
                    std::list<TrajectorySeg> candidate;
                    candidate.emplace_back(segment, start, q0, v0,
                                           first_acceleration, 0.0);
                    candidate.emplace_back(segment, middle, position, speed,
                                           second_acceleration, 0.0);
                    candidate.emplace_back(segment, end, q1, v1, 0.0, 0.0);
                    output.swap(candidate);
                    return true;
                }
            }
        }
    }
    return false;
}
// Reconstruct an accepted profile whose ramps used ideal durations instead
// of the differences between their rounded absolute timestamps.
bool FitRoundedRamps(double q0, double q1, double v0, double v1,
                     double velocity_cap, double acceleration_cap, double start,
                     double ta, double tc, double td, int segment,
                     std::list<TrajectorySeg> &output) {
    using Wide = long double;
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const Wide length = static_cast<Wide>(q1) - q0;
    const Wide position_budget =
        64 * eps * length + eps * std::max(std::abs(q0), std::abs(q1));
    const double speed_cap = std::max({velocity_cap, v0, v1});
    const std::array<double, 3> durations{ta, tc, td};
    const double planned_end = ((start + ta) + tc) + td;
    const Wide time_budget =
        64 * eps * std::max(std::abs(start), std::abs(planned_end));
    const int extension_phase = tc > 0 ? 1 : td > 0 ? 2 : 0;
    for (int extra = 0; extra < 8; ++extra) {
        std::array<double, 4> times{start};
        std::array<Wide, 3> dt{};
        bool valid = true;
        for (int i = 0; i < 3; ++i) {
            times[i + 1] = times[i] + durations[i];
            if (times[i + 1] - times[i] < durations[i])
                times[i + 1] = std::nextafter(
                    times[i + 1], std::numeric_limits<double>::infinity());
            if (i == extension_phase && durations[i] > 0)
                for (int k = 0; k < extra; ++k)
                    times[i + 1] = std::nextafter(
                        times[i + 1], std::numeric_limits<double>::infinity());
            dt[i] = times[i + 1] - times[i];
            valid &= std::isfinite(times[i + 1]) &&
                     (durations[i] == 0 ? dt[i] == 0 : dt[i] > 0);
        }
        if (!valid || std::abs(static_cast<Wide>(times.back()) - planned_end) >
                          time_budget)
            continue;
        const Wide fitted = dt[0] == 0 ? static_cast<Wide>(v0)
                            : dt[2] == 0
                                ? static_cast<Wide>(v1)
                                : (length - v0 * dt[0] / 2 - v1 * dt[2] / 2) /
                                      (dt[0] / 2 + dt[1] + dt[2] / 2);
        const Wide first_change = static_cast<Wide>(acceleration_cap) * dt[0];
        const Wide last_change = static_cast<Wide>(acceleration_cap) * dt[2];
        const Wide minimum_speed =
            std::max({Wide(0), v0 - first_change, v1 - last_change});
        const Wide maximum_speed =
            std::min({Wide(speed_cap), v0 + first_change, v1 + last_change});
        if (!std::isfinite(fitted) || minimum_speed > maximum_speed)
            continue;
        const Wide peak = std::clamp(fitted, minimum_speed, maximum_speed);
        const double up =
            dt[0] == 0 ? 0 : static_cast<double>((peak - v0) / dt[0]);
        const double down =
            dt[2] == 0 ? 0 : static_cast<double>((v1 - peak) / dt[2]);
        if (!std::isfinite(up) || !std::isfinite(down) ||
            std::abs(up) > acceleration_cap ||
            std::abs(down) > acceleration_cap)
            continue;
        const Wide first_position = q0 + (v0 / 2 + peak / 2) * dt[0];
        const Wide second_position = first_position + peak * dt[1];
        std::array<double, 4> positions{q0, static_cast<double>(first_position),
                                        static_cast<double>(second_position),
                                        q1};
        for (double &position : positions)
            position = std::clamp(position, q0, q1);
        if (dt[2] == 0)
            positions[2] = q1;
        if (dt[1] == 0)
            positions[1] = positions[2];
        const std::array<double, 4> velocities{v0, static_cast<double>(peak),
                                               static_cast<double>(peak), v1};
        const std::array<double, 3> accelerations{up, 0, down};
        for (int i = 0; i < 3; ++i) {
            const Wide coefficient = accelerations[i] / 2.0;
            const Wide q =
                positions[i] + dt[i] * (velocities[i] + dt[i] * coefficient);
            const Wide v = velocities[i] + dt[i] * 2 * coefficient;
            valid &= std::isfinite(positions[i + 1]) &&
                     positions[i + 1] >= positions[i] &&
                     std::abs(q - positions[i + 1]) <= position_budget &&
                     std::abs(v - velocities[i + 1]) <= 64 * eps * speed_cap;
            if (dt[i] == 0)
                valid &= positions[i] == positions[i + 1] &&
                         velocities[i] == velocities[i + 1];
        }
        if (!valid)
            continue;
        std::array<TrajectorySeg, 4> states;
        std::size_t count = 0;
        for (int i = 0; i < 3; ++i)
            if (dt[i] > 0)
                states[count++] =
                    TrajectorySeg(segment, times[i], positions[i],
                                  velocities[i], accelerations[i], 0.0);
        states[count++] = TrajectorySeg(segment, times[3], q1, v1, 0.0, 0.0);
        if (count < 2)
            continue;
        // Commit only after every numeric check. Equal phase counts can reuse
        // the accepted profile's nodes; otherwise retain allocation rollback.
        if (output.size() == count) {
            std::copy_n(states.begin(), count, output.begin());
        } else {
            std::list<TrajectorySeg> candidate(states.begin(),
                                               states.begin() + count);
            output.swap(candidate);
        }
        return true;
    }
    return false;
}
} // namespace

bool detail::ComputeTrapeziumProfile(const double &q0, const double &q1,
                                     double v0, double &v1,
                                     const double &requested_max_velocity,
                                     double max_acceleration, const double &t0,
                                     std::list<TrajectorySeg> &traj_segs,
                                     const int &seg_no) {
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
    constexpr double speed_roundoff =
        256.0 * std::numeric_limits<double>::epsilon();
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
    const double delta_h =
        (std::abs(v1 - v0) / max_acceleration) * average_velocity;

    if (h < delta_h) {
        if (h <= 0.0)
            return false;
        double end_velocity = v1;
        double acceleration = max_acceleration;
        if (v1 > v0) {
            // Lower an unreachable end speed. hypot avoids squaring v0;
            // the factored square root also avoids overflowing h * a.
            end_velocity = std::min(
                v1, std::hypot(v0, std::sqrt(h) * std::sqrt(max_acceleration) *
                                       std::sqrt(2.0)));
        }
        // Distance / average speed remains accurate when the velocity change
        // rounds away. The former (v1-v0)/a could become zero on a short path.
        const double duration = h / (0.5 * v0 + 0.5 * end_velocity);
        const double end_time = t0 + duration;
        if (!std::isfinite(duration) || duration <= 0.0 ||
            !std::isfinite(end_time) || end_time <= t0)
            return false;
        if (v0 > v1) {
            // Preserve the requested deceleration endpoint, even on a short
            // positive path. Global limit enforcement subsequently slows the
            // complete trajectory to accommodate this required acceleration.
            acceleration = (v1 - v0) / duration;
            if (!std::isfinite(acceleration))
                return false;
        }
        const double clock_error = (end_time - t0) - duration;
        const double velocity_budget =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max({max_velocity, v0, end_velocity});
        if (std::abs(acceleration * clock_error) > velocity_budget &&
            FitRoundedRamps(q0, q1, v0, end_velocity, max_velocity,
                            std::abs(acceleration), t0, duration, 0.0, 0.0,
                            seg_no, traj_segs)) {
            v1 = end_velocity;
            return true;
        }
        v1 = end_velocity;
        traj_segs.emplace_back(seg_no, t0, q0, v0, acceleration, 0.0);
        traj_segs.emplace_back(seg_no, end_time, q1, v1, 0.0, 0.0);
        return true;
    } else {
        // Find the maximum velocity that the trajectory can achieve...
        const double squared_peak =
            h * max_acceleration + 0.5 * (v0 * v0 + v1 * v1);
        const double inverse_sqrt_two = 1.0 / std::sqrt(2.0);
        // Keep ordinary rounding; use a scaled norm if squared intermediates
        // overflow or enter the subnormal range.
        const double v_max_upbound =
            std::isfinite(squared_peak) &&
                    squared_peak >= std::numeric_limits<double>::min()
                ? std::sqrt(squared_peak)
                : std::hypot(
                      std::sqrt(h) * std::sqrt(max_acceleration),
                      std::hypot(v0 * inverse_sqrt_two, v1 * inverse_sqrt_two));
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
        double ta =
            std::abs(v_lim - v0) / max_acceleration; ///< acceleration period
        double td =
            std::abs(v1 - v_lim) / max_acceleration; ///< deceleration period
        double tc = 0.0;                             ///< constant speed period
        const bool reaches_speed_cap = v_max_upbound >= max_velocity;

        if (!reaches_speed_cap) {
            // Rationalize (v_peak-v_endpoint)/a. Positive elapsed time must
            // survive even when v_peak rounds to a nonzero endpoint speed.
            const double signed_transition_distance =
                std::copysign(delta_h, v1 - v0);
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

        TrajectorySeg current_segment =
            TrajectorySeg(seg_no, t0, q0, v0, 0.0, 0.0);

        std::list<TrajectorySeg> phases;
        bool collapsed_phase = false;
        bool rounded_ramp = false;
        const auto append_phase = [&](double duration, double acceleration) {
            collapsed_phase = false;
            if (duration == 0.0)
                return true;
            current_segment.acc = acceleration;
            const auto next = detail::AdvanceTrajectoryStep(
                current_segment, duration, 0.0, seg_no);
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
                return std::abs(next.pos - current_segment.pos) <=
                           position_roundoff &&
                       std::abs(next.vel - current_segment.vel) <=
                           speed_roundoff;
            }
            const double clock_error =
                (next.timestamp - current_segment.timestamp) - duration;
            rounded_ramp |= std::abs(acceleration * clock_error) >
                            64.0 * std::numeric_limits<double>::epsilon() *
                                std::max({max_velocity, v0, v1});
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
            const double average_speed =
                0.5 * current_segment.vel + 0.5 * end_velocity;
            if (!std::isfinite(displacement) || displacement <= 0.0 ||
                average_speed <= 0.0)
                return false;
            const double end_time =
                current_segment.timestamp + displacement / average_speed;
            const double elapsed = end_time - current_segment.timestamp;
            if (!std::isfinite(end_time) || !std::isfinite(elapsed) ||
                elapsed <= 0.0)
                return false;
            const double acceleration =
                (end_velocity - current_segment.vel) / elapsed;
            constexpr double roundoff =
                64.0 * std::numeric_limits<double>::epsilon();
            double time_budget =
                roundoff * std::max(std::abs(planned_end_time),
                                    std::abs(current_segment.timestamp));
            const double speed_change =
                std::abs(end_velocity - current_segment.vel);
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
                !std::isfinite(distance_error) ||
                distance_error > distance_budget)
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
        const auto retry_with_lower_peak = [&]() {
            // A cap just above an endpoint can require a ramp shorter than
            // one clock ulp. If merging cannot preserve its displacement,
            // recompute with that endpoint as the peak instead. This lowers
            // the requested cap and removes the unrepresentable excursion;
            // it does not discard a boundary velocity or relax error budgets.
            // The new cap equals an endpoint, so this retry cannot recurse.
            const double endpoint_peak = std::max(v0, v1);
            if (endpoint_peak <= 0.0 || endpoint_peak >= max_velocity)
                return false;
            std::list<TrajectorySeg> candidate;
            double candidate_end_velocity = v1;
            if (!detail::ComputeTrapeziumProfile(
                    q0, q1, v0, candidate_end_velocity, endpoint_peak,
                    max_acceleration, t0, candidate, seg_no) ||
                candidate_end_velocity != v1 || candidate.size() < 2)
                return false;
            // Recheck every replacement phase using its stored timestamps.
            // Replanning must not bypass the failed merge's distance budget.
            for (auto it = candidate.begin(); std::next(it) != candidate.end();
                 ++it) {
                const auto next = std::next(it);
                const double elapsed = next->timestamp - it->timestamp;
                const double displacement = next->pos - it->pos;
                const double integrated =
                    it->vel * elapsed +
                    detail::QuadraticContribution(it->acc, elapsed);
                const double budget =
                    64.0 * std::numeric_limits<double>::epsilon() * h +
                    std::numeric_limits<double>::epsilon() *
                        std::max(std::abs(it->pos), std::abs(next->pos));
                if (!std::isfinite(integrated) ||
                    std::abs(integrated - displacement) > budget)
                    return false;
            }
            traj_segs.swap(candidate);
            return true;
        };
        const auto recover_collapsed_ramp = [&]() {
            if (retry_with_lower_peak())
                return true;
            return FitRoundedClockProfile(q0, q1, v0, v1, max_velocity,
                                          max_acceleration, t0, ta, tc, td,
                                          seg_no, traj_segs);
        };
        bool has_cruise_phase = false;
        if (!append_phase(ta, first_acceleration)) {
            if (!collapsed_phase || tc <= 0.0)
                return false;
            const double last_distance = (0.5 * v_lim + 0.5 * v1) * td;
            if (!append_cruise_transition(q1 - last_distance, v_lim,
                                          t0 + ta + tc))
                return recover_collapsed_ramp();
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
                return recover_collapsed_ramp();
        }
        phases.emplace_back(seg_no, current_segment.timestamp, q1, v1, 0.0,
                            0.0);
        if (rounded_ramp) {
            FitRoundedRamps(q0, q1, v0, v1, max_velocity, max_acceleration, t0,
                            ta, tc, td, seg_no, phases);
        }
        traj_segs.swap(phases);
        return true;
    }
}

} // namespace holistic_motion::robotics
