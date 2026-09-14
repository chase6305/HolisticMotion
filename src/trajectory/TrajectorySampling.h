#pragma once

#include <cmath>

namespace holistic_motion::robotics::detail {

// Called with positive finite limits and absolute nonzero derivatives.
// Preserve ordinary rounding, but take roots before division when the ratio
// overflows or underflows. The root of that ratio may still be representable.
inline double SquareRootRatio(double numerator, double denominator) {
    const double ratio = numerator / denominator;
    if (std::isnormal(ratio)) return std::sqrt(ratio);
    return std::sqrt(numerator) / std::sqrt(denominator);
}

inline double CubeRootRatio(double numerator, double denominator) {
    const double ratio = numerator / denominator;
    if (std::isnormal(ratio)) return std::pow(ratio, 1.0 / 3.0);
    return std::cbrt(numerator) / std::cbrt(denominator);
}

}  // namespace holistic_motion::robotics::detail
