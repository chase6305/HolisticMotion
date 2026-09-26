#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

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

// Positive roots preserve ordering. Reduce normal ratios across all samples
// before taking a root once; exceptional ratios still use the stable fallback.
class PathSpeedLimit {
public:
    void AddVelocity(double limit, double derivative) {
        if (derivative != 0.0)
            speed_ = std::min(speed_, limit / std::abs(derivative));
    }

    void AddAcceleration(double limit, double derivative) {
        if (derivative == 0.0)
            return;
        const double magnitude = std::abs(derivative);
        const double ratio = limit / magnitude;
        if (std::isnormal(ratio))
            acceleration_ratio_ = std::min(acceleration_ratio_, ratio);
        else
            speed_ = std::min(speed_, SquareRootRatio(limit, magnitude));
    }

    void AddJerk(double limit, double derivative) {
        if (derivative == 0.0)
            return;
        const double magnitude = std::abs(derivative);
        const double ratio = limit / magnitude;
        if (std::isnormal(ratio))
            jerk_ratio_ = std::min(jerk_ratio_, ratio);
        else
            speed_ = std::min(speed_, CubeRootRatio(limit, magnitude));
    }

    double Get() const {
        return std::min({speed_, std::sqrt(acceleration_ratio_),
                         std::pow(jerk_ratio_, 1.0 / 3.0)});
    }

private:
    double speed_{std::numeric_limits<double>::max()};
    double acceleration_ratio_{std::numeric_limits<double>::infinity()};
    double jerk_ratio_{std::numeric_limits<double>::infinity()};
};

}  // namespace holistic_motion::robotics::detail
