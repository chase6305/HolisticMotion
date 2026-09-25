#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>

#include "holistic_motion/trajectory/TrajectoryTrapezium.h"
#include "jerk_integration_checks.h"
#include "segment_sampling_checks.h"

using namespace holistic_motion::robotics;

namespace {
struct ProfileProbe : TrajectoryTrapezium<Rn<double, 2>> {
    ProfileProbe() : TrajectoryTrapezium(nullptr, nullptr) { this->dof_ = 2; }
    using TrajectoryTrapezium<Rn<double, 2>>::_ComputeSegmentMaxSVel;
    using TrajectoryTrapezium<Rn<double, 2>>::_ComputeTrapeziumProfile;
    using TrajectoryTrapezium<Rn<double, 2>>::_ComputeNextTrajStep;
};

void CheckProfile(double length, double v0, double v1, double vmax, double amax,
                  double q0 = 0.0, double t0 = 0.0) {
    ProfileProbe probe;
    std::list<TrajectorySeg> phases;
    const double requested_v1 = v1;
    const double q1 = q0 + length;
    if (!probe._ComputeTrapeziumProfile(q0, q1, v0, v1, vmax, amax, t0, phases, 0) ||
        phases.size() < 2) {
        throw std::runtime_error("missing trapezoidal profile");
    }
    const double position_tolerance = 1e-12 * std::max(1.0, length);
    const double velocity_tolerance = 1e-12 * std::max({1.0, vmax, v0, v1});
    if (phases.front().timestamp != t0 || phases.front().pos != q0 ||
        phases.front().vel != v0 || v1 != requested_v1 || phases.back().pos != q1 ||
        phases.back().vel != v1) {
        throw std::runtime_error("profile changed boundary conditions");
    }
    for (auto it = phases.begin(), next = std::next(it); next != phases.end();
         ++it, ++next) {
        const double dt = next->timestamp - it->timestamp;
        if (!std::isfinite(dt) || dt <= 0.0 || !std::isfinite(it->acc) ||
            std::abs(it->acc) > amax) {
            throw std::runtime_error("invalid trapezoidal phase");
        }
        // Integrate each constant-acceleration phase independently, including
        // its endpoint; reported boundary states must agree with the motion.
        const double position = it->pos + it->vel * dt + 0.5 * it->acc * dt * dt;
        const double velocity = it->vel + it->acc * dt;
        // Subtracting absolute timestamps quantizes a short phase duration;
        // acceleration amplifies that rounding when integrating velocity.
        const double time_tolerance =
            4.0 * std::numeric_limits<double>::epsilon() *
            std::max({std::abs(it->timestamp), std::abs(next->timestamp), dt});
        const double phase_velocity_tolerance =
            velocity_tolerance + std::abs(it->acc) * time_tolerance;
        if (std::abs(position - next->pos) > position_tolerance ||
            std::abs(velocity - next->vel) > phase_velocity_tolerance) {
            throw std::runtime_error("phase integration misses the reported boundary");
        }
        if (std::min(it->vel, velocity) < -phase_velocity_tolerance ||
            std::max(it->vel, velocity) >
                std::max({vmax, v0, v1}) + phase_velocity_tolerance) {
            throw std::runtime_error("phase has an incorrect acceleration direction");
        }
    }
}

void CheckCollapsedRampBesideCruise() {
    // Recorded scalar intervals from the four remaining randomized failures.
    // A late start makes their final or initial ramp shorter than one clock ulp.
    for (const auto &p :
         {std::array<double, 6>{66.5193344135276, 96.60372628140267, 0.2265367688528383,
                                0.29291630408370545, 0.29291630408371444,
                                1.387343068500515},
          std::array<double, 6>{68.40790943864458, 104.53683930167746,
                                0.2585331439917136, 0.26440221377150974,
                                0.26440221377150525, 0.341581502663684},
          std::array<double, 6>{74.47442128085522, 116.91582397283716,
                                0.2991828509450422, 0.28837733837942814,
                                0.29918285094504693, 0.6504015001320413},
          std::array<double, 6>{7.141729974195082, 27.283268837390622,
                                0.13244950878744685, 0.13502812165066638,
                                0.13502812165066883, 0.6376673769478273}}) {
        CheckProfile(p[1] - p[0], p[2], p[3], p[4], p[5], p[0], 1000.0);
    }
    // Both ends need a sub-clock velocity adjustment. The surviving cruise
    // must actually carry the net change, not silently hold its initial speed.
    ProfileProbe probe;
    double end_velocity = 0.3 + 1e-14;
    std::list<TrajectorySeg> phases;
    if (!probe._ComputeTrapeziumProfile(10.0, 40.0, 0.3, end_velocity, 0.3 + 2e-14, 1.0,
                                        1000.0, phases, 0) ||
        phases.size() != 2 || phases.front().acc <= 0.0 || phases.front().vel != 0.3 ||
        phases.back().vel != end_velocity) {
        throw std::runtime_error("merged cruise must retain the velocity change");
    }
    CheckProfile(30.0, 0.3, 0.3 + 1e-14, 0.3 + 2e-14, 1.0, 10.0, 1000.0);

    // A merged phase whose rounded duration cannot reproduce its displacement
    // must still fail transactionally, even when its speeds are close.
    end_velocity = 0.5 + 1e-12;
    if (probe._ComputeTrapeziumProfile(10.0, 12.0, 0.5, end_velocity, 0.5 + 2e-12, 1.0,
                                       1e6, phases, 0) ||
        !phases.empty() || end_velocity != 0.5 + 1e-12) {
        throw std::runtime_error("merged cruise bypassed displacement checks");
    }
}

void CheckCollapsedRampAtNonzeroPosition() {
    // Recorded from a short interval late in a blended path. Its displacement
    // error is below one coordinate ulp but exceeds an h-only relative budget.
    constexpr double q0 = 51.523257784451715;
    constexpr double q1 = 51.675437373741296;
    constexpr double v0 = 0.8398822958553492;
    constexpr double requested_v1 = 0.8398822958553513;
    constexpr double vmax = 0.8398822958553549;
    constexpr double amax = 0.9354261944068154;
    constexpr double t0 = 74.28314124735722;
    CheckProfile(q1 - q0, v0, requested_v1, vmax, amax, q0, t0);
    ProfileProbe probe;
    std::list<TrajectorySeg> phases;
    double v1 = requested_v1;
    if (!probe._ComputeTrapeziumProfile(q0, q1, v0, v1, vmax, amax, t0, phases, 0))
        throw std::runtime_error("coordinate roundoff rejected a merged cruise");
    for (auto it = phases.begin(), next = std::next(it); next != phases.end();
         ++it, ++next) {
        const long double dt = next->timestamp - it->timestamp;
        const long double position = static_cast<long double>(it->pos) +
                                     it->vel * dt + 0.5L * it->acc * dt * dt;
        const long double coordinate_ulp =
            std::nextafter(next->pos, std::numeric_limits<double>::infinity()) -
            next->pos;
        if (std::abs(position - next->pos) > coordinate_ulp)
            throw std::runtime_error("merged cruise exceeds coordinate precision");
    }
}

void CheckLowerPeakPreservesBoundariesAtCoarseClock() {
    // A recorded rejected interval has a 2.18e-14 ramp on a 5.68e-14 clock.
    // Reducing the peak to the faster endpoint removes that excursion while
    // retaining both endpoint speeds and the existing acceleration limit.
    constexpr double q0 = 14.159136934492288;
    constexpr double q1 = 14.199071545761711;
    constexpr double fast = 0.29546205743865284;
    constexpr double slow = 0.03750188018187762;
    constexpr double cap = 0.2954620574387062;
    constexpr double acceleration = 2.443973585440111;
    constexpr double start = 273.568908829043;
    for (double clock : {0.0, start}) {
        CheckProfile(q1 - q0, fast, slow, cap, acceleration, q0, clock);
        ProfileProbe probe;
        std::list<TrajectorySeg> phases;
        double end_velocity = slow;
        if (!probe._ComputeTrapeziumProfile(q0, q1, fast, end_velocity, cap,
                                            acceleration, clock, phases, 0))
            throw std::runtime_error("lower peak did not recover a feasible profile");
        for (auto it = phases.begin(); it != phases.end(); ++it) {
            if (it->vel > cap)
                throw std::runtime_error("lower peak increased the speed cap");
            const auto next = std::next(it);
            if (next == phases.end())
                break;
            const long double dt = next->timestamp - it->timestamp;
            const long double position = static_cast<long double>(it->pos) +
                                         it->vel * dt + 0.5L * it->acc * dt * dt;
            const double budget =
                64.0 * std::numeric_limits<double>::epsilon() * (q1 - q0) +
                std::numeric_limits<double>::epsilon() *
                    std::max(std::abs(it->pos), std::abs(next->pos));
            if (std::abs(position - next->pos) > budget)
                throw std::runtime_error("lower peak bypassed displacement checks");
        }
    }
}

void CheckShortTransition(double length, double v0, double requested_v1) {
    ProfileProbe probe;
    std::list<TrajectorySeg> phases;
    double v1 = requested_v1;
    const double vmax = std::max({1.0, v0, requested_v1});
    if (!probe._ComputeTrapeziumProfile(0.0, length, v0, v1, vmax, 1.0, 0.0, phases,
                                        0) ||
        phases.size() != 2) {
        throw std::runtime_error("missing short velocity transition");
    }
    const auto &start = phases.front();
    const auto &end = phases.back();
    const double dt = end.timestamp - start.timestamp;
    if (!std::isfinite(dt) || dt <= 0.0 || !std::isfinite(start.acc)) {
        throw std::runtime_error("short transition has no finite elapsed time");
    }
    const long double time = dt;
    const long double position =
        static_cast<long double>(start.vel) * time + 0.5L * start.acc * time * time;
    const long double velocity = start.vel + static_cast<long double>(start.acc) * time;
    if (std::abs(position - length) > 1e-12L * length ||
        std::abs(velocity - end.vel) > 1e-14L || end.pos != length || end.vel != v1) {
        throw std::runtime_error("short transition integration misses its endpoint");
    }
    if (v0 > requested_v1) {
        if (v1 != requested_v1 || start.acc >= -1.0)
            throw std::runtime_error("short deceleration must preserve endpoint speed");
    } else if (v1 < v0 || v1 > requested_v1 || start.acc != 1.0) {
        throw std::runtime_error(
            "short acceleration must lower the reachable end speed");
    }
}

void CheckUnrepresentableShortTransitionIsRejected() {
    ProfileProbe probe;
    for (double t0 : {1e20, std::numeric_limits<double>::max()}) {
        double v1 = 0.8;
        std::list<TrajectorySeg> phases;
        if (probe._ComputeTrapeziumProfile(0.0, 1e-6, 0.5, v1, 1.0, 1.0, t0, phases,
                                           0) ||
            !phases.empty() || v1 != 0.8) {
            throw std::runtime_error(
                "unrepresentable transition must fail without a partial "
                "result");
        }
    }
}

void CheckNonzeroEndpointPeak(double length, double speed, double vmax, double amax) {
    ProfileProbe probe;
    double end_speed = speed;
    std::list<TrajectorySeg> phases;
    if (!probe._ComputeTrapeziumProfile(0.0, length, speed, end_speed, vmax, amax, 0.0,
                                        phases, 0) ||
        phases.size() < 2) {
        throw std::runtime_error("moving endpoints must retain positive travel time");
    }
    const long double h = length, v = speed, limit = vmax, a = amax;
    const long double peak = std::sqrt(a * h + v * v);
    const long double expected_duration =
        peak <= limit ? 2.0L * h / (peak + v)
                      : h / limit + (limit - v) * (limit - v) / (a * limit);
    if (std::abs(phases.back().timestamp - expected_duration) >
            1e-12L * expected_duration ||
        end_speed != speed) {
        throw std::runtime_error(
            "moving endpoint timing must match the analytic profile");
    }
    for (auto it = phases.begin(), next = std::next(it); next != phases.end();
         ++it, ++next) {
        const long double dt = next->timestamp - it->timestamp;
        const long double position =
            it->pos + static_cast<long double>(it->vel) * dt + 0.5L * it->acc * dt * dt;
        const long double velocity = it->vel + static_cast<long double>(it->acc) * dt;
        if (!std::isfinite(dt) || dt <= 0.0L ||
            std::abs(position - next->pos) > 1e-12L * h ||
            std::abs(velocity - next->vel) > 1e-12L * limit ||
            std::abs(it->acc) > amax) {
            throw std::runtime_error("moving endpoint phase has inconsistent motion");
        }
    }
}

void CheckUnrepresentableGeneralPhasesAreRejected() {
    ProfileProbe probe;
    // All phases collapse in the first case; the short ramps in the second.
    for (const auto &input : {std::array<double, 3>{1e20, 1.0, 1.0},
                              std::array<double, 3>{0x1p52, 2.01, 100.0}}) {
        double v1 = 0.0;
        std::list<TrajectorySeg> phases;
        if (probe._ComputeTrapeziumProfile(0.0, input[1], 0.0, v1, 1.0, input[2],
                                           input[0], phases, 0) ||
            !phases.empty()) {
            throw std::runtime_error(
                "collapsed moving phases must not return a valid prefix");
        }
    }
}

void CheckMergedCruiseTimingCorrection() {
    // A recorded short ramp changes the average speed by more than 256 ulps.
    // Its corrected elapsed time is accurate in distance and acceleration.
    CheckProfile(43.8266138539926 - 40.461565697748156, 0.001714266295441807,
                  0.0014801148204472974, 0.0017142662954422409,
                  0.022815250329534954, 40.461565697748156, 10728.812330648614);
    // When the last deceleration collapses after a very long cruise, retain
    // the complete stopping motion by spreading it across that cruise. The
    // physical checks must validate every phase, not accept a valid prefix.
    CheckProfile(1e20, 0.0, 0.0, 1.0, 100.0);
    ProfileProbe probe;
    std::list<TrajectorySeg> phases;
    double v1 = 0.0;
    if (!probe._ComputeTrapeziumProfile(0.0, 1e20, 0.0, v1, 1.0, 100.0, 0.0,
                                        phases, 0) ||
        phases.size() != 3 || phases.back().timestamp != 2e20 ||
        phases.back().vel != 0.0 || phases.back().pos != 1e20) {
        throw std::runtime_error("long merged cruise must retain its stopping phase");
    }
}

void CheckLongZeroJerkStep() {
    ProfileProbe probe;
    const TrajectorySeg initial(0, 0.0, 3.0, 1e-110, 2e-220, 0.0);
    const auto next = probe._ComputeNextTrajStep(initial, 1e110, 5.0, 9);
    if (!std::isfinite(next.pos) || std::abs(next.pos - 5.0) > 1e-14 ||
        std::abs(next.vel / 1e-110 - 3.0) > 1e-14 || next.acc != initial.acc ||
        next.timestamp != 1e110 || next.jerk != 5.0 || next.seg_no != 9) {
        throw std::runtime_error(
            "zero jerk must not multiply an overflowing time cube");
    }
}

void CheckRoundedCapTransitions() {
    ProfileProbe probe;
    for (double scale : {1e-100, 1.0, 1e100}) {
        for (bool at_start : {false, true}) {
            const double cap = std::nextafter(0.5, 0.0) * scale;
            const double v0 = (at_start ? 0.5 : 0.25) * scale;
            double v1 = (at_start ? 0.25 : 0.5) * scale;
            const double requested_v1 = v1;
            std::list<TrajectorySeg> phases;
            if (!probe._ComputeTrapeziumProfile(10.0 * scale, 11.0 * scale, v0, v1, cap,
                                                scale, 10.0, phases, 0) ||
                phases.size() < 2 || phases.front().vel != v0 ||
                phases.back().vel != requested_v1 || v1 != requested_v1) {
                throw std::runtime_error(
                    "rounded cap transition rejected a representable profile");
            }
            for (auto it = phases.begin(), next = std::next(it); next != phases.end();
                 ++it, ++next) {
                const double dt = next->timestamp - it->timestamp;
                const double position =
                    it->pos + it->vel * dt + 0.5 * it->acc * dt * dt;
                const double velocity = it->vel + it->acc * dt;
                if (!std::isfinite(dt) || dt <= 0.0 ||
                    std::abs(position - next->pos) > 1e-13 * scale ||
                    std::abs(velocity - next->vel) > 1e-13 * scale ||
                    std::abs(it->acc) > scale) {
                    throw std::runtime_error(
                        "rounded cap transition changed physical motion");
                }
            }
        }
        // Small physical units must not turn a meaningful velocity change
        // into rounding noise when its absolute timestamp cannot advance.
        double v1 = 0.0;
        std::list<TrajectorySeg> phases;
        if (probe._ComputeTrapeziumProfile(10.0 * scale, 11.0 * scale, 0.5 * scale, v1,
                                           0.25 * scale, scale, 1e20, phases, 0) ||
            !phases.empty()) {
            throw std::runtime_error("unrepresentable motion was accepted");
        }
    }
}

void CheckJerkRange() {
    ProfileProbe probe;
    const auto integrate = [&](const TrajectorySeg &initial, double time,
                               double next_jerk, int next_seg_no) {
        return probe._ComputeNextTrajStep(initial, time, next_jerk, next_seg_no);
    };
    CheckJerkIntegrationRange(integrate);
    CheckQuadraticIntegrationRange(integrate);
}

void CheckCurveSampling() {
    ProfileProbe probe;
    sampling_checks::CheckSegmentSampling(
        [&](const auto &segment, const auto &v, const auto &a, const auto &j) {
            return probe._ComputeSegmentMaxSVel(segment, v, a, j);
        });
}
} // namespace

int main() {
    int failures = 0;
    for (const auto &parameters :
         {std::array<double, 5>{0.1, 0.0, 0.0, 1e10, 1e15},
          std::array<double, 5>{0.100005, 0.0, 0.0, 1.0, 10.0},
          std::array<double, 5>{0.1, 0.0, 0.0, 1.0, 1e6},
          std::array<double, 5>{1.0, 0.8, 0.1, 0.5, 1.0},
          std::array<double, 5>{1.0, 0.1, 0.8, 0.5, 1.0},
          std::array<double, 5>{19.0, 0.0, 0.2678885027613014, 0.2678885027613005,
                                0.2585985702225886},
          std::array<double, 5>{6.510572995333455, 0.0, 0.8072406861546729,
                                0.8072406861546723, 0.827492861910945},
          std::array<double, 5>{0.2643455806903485, 0.0, 0.0037290323588947765,
                                0.0037290323588949166, 0.02204043382988225}}) {
        try {
            CheckProfile(parameters[0], parameters[1], parameters[2], parameters[3],
                         parameters[4]);
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            ++failures;
        }
    }
    for (double length : {1e-6, 1e-12, 1e-18}) {
        for (const auto &endpoints :
             {std::array<double, 2>{0.5, 0.8}, std::array<double, 2>{0.8, 0.1}}) {
            try {
                CheckShortTransition(length, endpoints[0], endpoints[1]);
            } catch (const std::exception &error) {
                std::cerr << error.what() << '\n';
                ++failures;
            }
        }
    }
    try {
        CheckShortTransition(1.0, 1e155, 1.1e155);
        CheckUnrepresentableShortTransitionIsRejected();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        ++failures;
    }
    for (const auto &parameters :
         {std::array<double, 4>{1e-18, 0.5, 1.0, 1.0},
          std::array<double, 4>{1e-18, 0.5, 0.5, 1.0},
          std::array<double, 4>{1.0, 1e155, 2e155, 1.0},
          std::array<double, 4>{1e-200, 1e-200, 2e-200, 1e-200}}) {
        try {
            CheckNonzeroEndpointPeak(parameters[0], parameters[1], parameters[2],
                                     parameters[3]);
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            ++failures;
        }
    }
    for (const auto check :
         {CheckUnrepresentableGeneralPhasesAreRejected, CheckLongZeroJerkStep,
          CheckJerkRange, CheckCurveSampling, CheckRoundedCapTransitions,
          CheckCollapsedRampBesideCruise, CheckCollapsedRampAtNonzeroPosition,
          CheckLowerPeakPreservesBoundariesAtCoarseClock,
          CheckMergedCruiseTimingCorrection}) {
        try {
            check();
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            ++failures;
        }
    }
    return failures ? 1 : 0;
}
