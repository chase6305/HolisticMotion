#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "holistic_motion/kinematics/Types.h"

#include "../utility/AnglePhase.h"
#include "JointMotion.h"

namespace holistic_motion::robotics::detail {

constexpr double kTwoPi = 2.0 * M_PI;

inline double CoordinatePeriod(const JointNode &node) {
    const double norm = detail::JointAxisNorm(node.axis);
    return norm > 0.0 && std::isfinite(norm) ? kTwoPi / norm : 0.0;
}

inline double CoordinateAngle(double joint, const JointNode &node) {
    return joint * detail::JointAxisNorm(node.axis);
}

inline bool SameCoordinateRotation(double first, double second, const JointNode &node) {
    const double first_angle = CoordinateAngle(first, node);
    const double second_angle = CoordinateAngle(second, node);
    return std::isfinite(first_angle) && std::isfinite(second_angle) &&
           SameAnglePhase(first_angle, second_angle);
}

inline double PeriodicAnchor(double joint, const JointNode &node, double period) {
    double anchor = std::remainder(joint, period);
    if (!SameCoordinateRotation(anchor, joint, node)) {
        const double angle = CoordinateAngle(joint, node);
        if (!std::isfinite(angle))
            return std::numeric_limits<double>::quiet_NaN();
        anchor = std::atan2(std::sin(angle), std::cos(angle)) * (period / kTwoPi);
    }
    return anchor;
}

inline bool NearestPeriodicValue(double lower, double upper, double joint,
                                 double reference, double &value, const JointNode &node,
                                 double period) {
    if (!std::isfinite(lower) || !std::isfinite(upper))
        return false;
    const bool inside = joint >= lower && joint <= upper;
    if (inside && std::abs(reference - joint) < 0.5 * period) {
        value = joint;
        return true;
    }
    double anchor = joint;
    if (!inside) {
        anchor = PeriodicAnchor(joint, node, period);
    }
    const long double minimum =
        std::ceil((static_cast<long double>(lower) - anchor) / period);
    const long double maximum =
        std::floor((static_cast<long double>(upper) - anchor) / period);
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum)
        return false;
    const long double nearest =
        std::round((static_cast<long double>(reference) - anchor) / period);
    const long double turn = std::clamp(nearest, minimum, maximum);
    constexpr long double kMaxExactTurn = 4503599627370495.0L;
    if (!std::isfinite(turn) || std::abs(turn) > kMaxExactTurn)
        return false;
    value = turn == 0.0L ? anchor : std::fma(static_cast<double>(turn), period, anchor);
    return value >= lower && value <= upper &&
           (value == joint || SameCoordinateRotation(value, joint, node));
}

} // namespace holistic_motion::robotics::detail
