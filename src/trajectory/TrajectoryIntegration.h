#pragma once

#include <cmath>

namespace holistic_motion::robotics::detail {

inline double QuadraticContribution(double coefficient, double time) {
    const double half_coefficient = 0.5 * coefficient;
    const double first_product = half_coefficient * time;
    const double result = first_product * time;
    if (coefficient == 0.0 || time == 0.0 || !std::isfinite(coefficient) ||
        !std::isfinite(time) ||
        (std::isnormal(half_coefficient) && std::isnormal(first_product) &&
         std::isnormal(result))) {
        return result;
    }

    // Halving a subnormal coefficient can erase or round it before a large
    // time amplifies it. Apply the factor of one half to the exponent instead.
    int coefficient_exponent = 0;
    int time_exponent = 0;
    const double coefficient_mantissa =
        std::frexp(coefficient, &coefficient_exponent);
    const double time_mantissa = std::frexp(time, &time_exponent);
    return std::scalbn(coefficient_mantissa * time_mantissa * time_mantissa,
                       coefficient_exponent + 2 * time_exponent - 1);
}

inline double CubicJerkDisplacement(double jerk, double time) {
    if (jerk == 0.0) return 0.0;
    const double time_cubed = std::pow(time, 3);
    const double product = jerk * time_cubed;
    // Preserve the existing rounding for ordinary inputs. Nonfinite inputs
    // still propagate to the caller's state validation.
    if (!std::isfinite(jerk) || !std::isfinite(time) ||
        (std::isnormal(time_cubed) && std::isnormal(product))) {
        return product / 6.0;
    }

    // Keep intermediate magnitudes bounded even when time^3 or jerk*time^3
    // is unrepresentable. Only scale the final displacement back to double.
    int jerk_exponent = 0;
    int time_exponent = 0;
    const double jerk_mantissa = std::frexp(jerk, &jerk_exponent);
    const double time_mantissa = std::frexp(time, &time_exponent);
    return std::scalbn(
        jerk_mantissa * time_mantissa * time_mantissa * time_mantissa / 6.0,
        jerk_exponent + 3 * time_exponent);
}

}  // namespace holistic_motion::robotics::detail
