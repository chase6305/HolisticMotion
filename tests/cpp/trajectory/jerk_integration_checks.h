#pragma once

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include "holistic_motion/trajectory/Types.h"

template <typename Integrate>
void CheckJerkIntegrationRange(Integrate integrate) {
    using holistic_motion::robotics::TrajectorySeg;
    struct Case {
        double jerk;
        double time;
    };
    for (const auto& input :
         {Case{1e-300, 1e110}, Case{-1e-300, 1e110}, Case{1e300, 1e-110},
          Case{-1e300, 1e-110}, Case{1e307, 4.0}, Case{-1e307, 4.0}}) {
        const TrajectorySeg initial(0, 0.0, 0.0, 0.0, 0.0, input.jerk);
        const auto result = integrate(initial, input.time, 7.0, 3);
        const long double time = input.time;
        const long double jerk = input.jerk;
        const long double expected_position = jerk / 6 * time * time * time;
        const long double expected_velocity = jerk * time * time / 2;
        const long double expected_acceleration = jerk * time;
        const auto close = [](double actual, long double expected) {
            return std::isfinite(actual) &&
                   std::abs(actual - expected) <= 1e-14L * std::abs(expected);
        };
        if (!close(result.pos, expected_position) ||
            !close(result.vel, expected_velocity) ||
            !close(result.acc, expected_acceleration) ||
            result.timestamp != input.time || result.jerk != 7.0 ||
            result.seg_no != 3)
            throw std::runtime_error(
                "jerk integration lost a representable state");
    }
    // A true displacement overflow must remain nonfinite for profile
    // validation.
    const TrajectorySeg overflowing(0, 0.0, 0.0, 0.0, 0.0,
                                    std::numeric_limits<double>::max());
    if (std::isfinite(integrate(overflowing, 4.0, 0.0, 0).pos))
        throw std::runtime_error("true jerk displacement overflow was hidden");
}

template <typename Integrate>
void CheckQuadraticIntegrationRange(Integrate integrate) {
    using holistic_motion::robotics::TrajectorySeg;
    // Exact powers of two provide references without first halving a
    // subnormal coefficient. Odd multiples expose rounding as well as zero.
    constexpr int time_exponent = 500;
    constexpr int coefficient_exponent =
        std::numeric_limits<double>::min_exponent -
        std::numeric_limits<double>::digits;
    const double time = std::scalbn(1.0, time_exponent);
    for (double multiple : {1.0, 3.0, -1.0, -3.0}) {
        const double coefficient =
            multiple * std::numeric_limits<double>::denorm_min();
        const double linear =
            std::scalbn(multiple, time_exponent + coefficient_exponent);
        const double quadratic =
            std::scalbn(multiple, 2 * time_exponent + coefficient_exponent - 1);
        const double cubic = std::scalbn(
            multiple / 6.0, 3 * time_exponent + coefficient_exponent);
        const auto close = [](double actual, double expected) {
            return std::isfinite(actual) &&
                   std::abs(actual - expected) <= 1e-14 * std::abs(expected);
        };
        const TrajectorySeg accelerating(0, 0.0, 0.0, 0.0, coefficient, 0.0);
        const auto acceleration_result = integrate(accelerating, time, 7.0, 3);
        const TrajectorySeg jerking(0, 0.0, 0.0, 0.0, 0.0, coefficient);
        const auto jerk_result = integrate(jerking, time, 7.0, 3);
        if (!close(acceleration_result.pos, quadratic) ||
            !close(acceleration_result.vel, linear) ||
            acceleration_result.acc != coefficient ||
            !close(jerk_result.pos, cubic) ||
            !close(jerk_result.vel, quadratic) ||
            !close(jerk_result.acc, linear))
            throw std::runtime_error(
                "halving a subnormal coefficient lost motion");
        for (const auto& result : {acceleration_result, jerk_result}) {
            if (result.timestamp != time || result.jerk != 7.0 ||
                result.seg_no != 3)
                throw std::runtime_error(
                    "quadratic integration changed metadata");
        }
    }
    const TrajectorySeg overflowing(0, 0.0, 0.0, 0.0,
                                    std::numeric_limits<double>::max(), 0.0);
    if (std::isfinite(integrate(overflowing, 4.0, 0.0, 0).pos))
        throw std::runtime_error(
            "true quadratic displacement overflow was hidden");
}
