#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_DECASTELJAU_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_DECASTELJAU_H_

#include <stdexcept>
#include <vector>

#include "holistic_motion/manif/impl/lie_group_base.h"

namespace holistic_motion::robotics {

/// Sample piecewise geodesic Bezier curves with shared segment endpoints.
/// The legacy degree parameter is the number of control points per complete
/// segment (at least two). A partial final segment uses the remaining points;
/// closed curves append the first point to that segment. Each segment excludes
/// its initial point and includes its final point. Two controls produce
/// k_interp samples; larger segments produce k_interp times their control count.
template <typename LieGroup>
std::vector<typename LieGroup::LieGroup>
Decasteljau(const std::vector<LieGroup> &trajectory, const unsigned int degree,
            const unsigned int k_interp, const bool closed_curve = false) {
    HOLISTIC_MOTION_CHECK(trajectory.size() >= 2, "Input trajectory requires at least two points!");
    HOLISTIC_MOTION_CHECK(degree >= 2 && degree <= trajectory.size(),
                          "Control count must lie between two and the input size!");
    HOLISTIC_MOTION_CHECK(k_interp > 0, "k_interp must be greater than zero!");
    using Group = typename LieGroup::LieGroup;
    std::vector<Group> curve, work;
    work.reserve(degree);
    const auto append_segment = [&](std::size_t start, std::size_t count, bool closing) {
        const std::size_t controls = count + (closing ? 1 : 0);
        const std::size_t multiplier = controls == 2 ? 1 : controls;
        if (k_interp > curve.max_size() / multiplier)
            throw std::length_error("Bezier sample count is too large");
        const std::size_t samples = k_interp * multiplier;
        if (samples > curve.max_size() - curve.size())
            throw std::length_error("Bezier sample count is too large");
        for (std::size_t sample = 0; sample < samples; ++sample) {
            if (sample + 1 == samples) {
                curve.emplace_back(closing ? trajectory.front() : trajectory[start + count - 1]);
                continue;
            }
            work.clear();
            for (std::size_t point = 0; point < count; ++point)
                work.emplace_back(trajectory[start + point]);
            if (closing)
                work.emplace_back(trajectory.front());
            const typename Group::Scalar t =
                typename Group::Scalar(sample + 1) / typename Group::Scalar(samples);
            for (std::size_t remaining = controls - 1; remaining > 0; --remaining)
                for (std::size_t point = 0; point < remaining; ++point)
                    work[point] = work[point].Rplus(work[point + 1].Rminus(work[point]) * t);
            curve.push_back(work.front());
        }
    };
    std::size_t start = 0;
    while (degree <= trajectory.size() - start) {
        append_segment(start, degree, false);
        start += degree - 1;
    }
    const std::size_t remaining = trajectory.size() - start;
    if (closed_curve || remaining > 1)
        append_segment(start, remaining, closed_curve);
    return curve;
}

} // namespace holistic_motion::robotics

#endif
