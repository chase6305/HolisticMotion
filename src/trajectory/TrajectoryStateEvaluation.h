#pragma once

#include <array>
#include <cmath>

#include <Eigen/Core>

// Shared chain-rule arithmetic for state queries and local sampling workspaces.
// Keep these helpers local to each implementation unit; neither the expressions
// nor their extreme-value fallback paths depend on which caller uses them.
namespace holistic_motion::robotics {
namespace {
template <unsigned Power, typename Tangent>
Tangent ScaleByPower(const Tangent &value, double speed, double speed_power) {
    if (std::isnormal(speed_power) || speed == 0.0)
        return value * speed_power;
    // A speed power may overflow or underflow even when its product with a
    // geometric derivative is representable. Starting from that derivative
    // keeps every intermediate between it and the final result in magnitude.
    Tangent result = value;
    for (unsigned order = 0; order < Power; ++order)
        result *= speed;
    return result;
}

template <typename Tangent>
Tangent ScaleMixedJerkExtreme(const Tangent &curvature, double speed,
                              double acceleration, double factor) {
    if (!std::isfinite(speed) || !std::isfinite(acceleration) ||
        !curvature.Coeffs().allFinite())
        return curvature * factor;
    // Accumulate exponents separately when the scalar product is outside the
    // normal range. The geometric coefficient can restore a finite result.
    // Unlike extended precision, this also works when long double is double.
    int speed_exponent = 0, acceleration_exponent = 0;
    const double speed_mantissa = std::frexp(speed, &speed_exponent);
    const double acceleration_mantissa =
        std::frexp(acceleration, &acceleration_exponent);
    Tangent result = curvature;
    for (Eigen::Index index = 0; index < result.Coeffs().size(); ++index) {
        int curvature_exponent = 0;
        const double curvature_mantissa =
            std::frexp(curvature.Coeffs()[index], &curvature_exponent);
        result.Coeffs()[index] = std::scalbn(
            3.0 * curvature_mantissa * speed_mantissa * acceleration_mantissa,
            curvature_exponent + speed_exponent + acceleration_exponent);
    }
    return result;
}

template <typename Tangent>
Tangent ScaleMixedJerk(const Tangent &curvature, double speed, double acceleration) {
    if (speed == 0.0 || acceleration == 0.0)
        return Tangent::ZeroHelper();
    const double factor = 3.0 * speed * acceleration;
    if (std::isnormal(factor))
        return curvature * factor;
    return ScaleMixedJerkExtreme(curvature, speed, acceleration, factor);
}

template <typename State, typename Tangent>
inline void ComposeDerivatives(State &state, const std::array<double, 4> &jet,
                               const Tangent &tangent, const Tangent &curvature,
                               const Tangent &torsion, double time_scale) {
    const double inverse_scale = 1.0 / time_scale;
    const double speed_squared = jet[1] * jet[1];

    state.velocity = tangent * jet[1] * inverse_scale;
    state.acceleration =
        (tangent * jet[2] + ScaleByPower<2>(curvature, jet[1], speed_squared)) *
        inverse_scale * inverse_scale;
    state.jerk = (tangent * jet[3] + ScaleMixedJerk(curvature, jet[1], jet[2]) +
                  ScaleByPower<3>(torsion, jet[1], speed_squared * jet[1])) *
                 inverse_scale * inverse_scale * inverse_scale;
}

} // namespace
} // namespace holistic_motion::robotics
