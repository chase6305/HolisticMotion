#pragma once

#include <cmath>

namespace holistic_motion::robotics::detail {

// A whole-turn shift must preserve the physical angle after rounding back to
// double. Comparing remainders alone loses the original phase at large angles.
inline bool SameAnglePhase(double first, double second) {
    constexpr double tolerance = 1e-10;
    return std::isfinite(first) && std::isfinite(second) &&
           std::abs(std::sin(first) - std::sin(second)) <= tolerance &&
           std::abs(std::cos(first) - std::cos(second)) <= tolerance;
}

} // namespace holistic_motion::robotics::detail
