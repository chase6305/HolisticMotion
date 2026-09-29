#pragma once

#include <array>
#include <cmath>

#include <Eigen/Core>

// Shared chain-rule arithmetic for state queries and local sampling workspaces.
// Inline template definitions can be shared by the internal batch evaluator
// without changing the arithmetic or introducing per-sample dynamic dispatch.
namespace holistic_motion::robotics {
namespace detail {
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

template <typename State, typename Tangent>
inline void
ComposeLinearDerivatives(State &state, const std::array<double, 4> &jet,
                         const Tangent &tangent, double time_scale) {
    if (!std::isfinite(jet[1]) || !std::isfinite(jet[2])) {
        const auto zero = Tangent::ZeroHelper();
        ComposeDerivatives(state, jet, tangent, zero, zero, time_scale);
        return;
    }
    // Finite clock derivatives keep the line's higher geometric terms zero,
    // even when speed powers or the mixed factor overflow. Preserve their signs
    // and the original additions so zero-valued state components stay
    // identical.
    const double mixed_zero = jet[1] == 0.0 || jet[2] == 0.0
                                  ? 0.0
                                  : std::copysign(0.0, jet[1] * jet[2]);
    const double cubic_zero = std::copysign(0.0, jet[1]);
    const double inverse_scale = 1.0 / time_scale;
    // Evaluate coefficient expressions directly, retaining each multiplication
    // and zero addition without materializing intermediate tangent vectors.
    state.velocity.Coeffs() = tangent.Coeffs() * jet[1] * inverse_scale;
    state.acceleration.Coeffs() = ((tangent.Coeffs().array() * jet[2] + 0.0) *
                                   inverse_scale * inverse_scale)
                                      .matrix();
    state.jerk.Coeffs() =
        ((tangent.Coeffs().array() * jet[3] + mixed_zero + cubic_zero) *
         inverse_scale * inverse_scale * inverse_scale)
            .matrix();
}

} // namespace detail
} // namespace holistic_motion::robotics
