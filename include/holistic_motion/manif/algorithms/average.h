#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_AVERAGE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_AVERAGE_H_

#include "holistic_motion/manif/impl/lie_group_base.h"

namespace holistic_motion::robotics {

/// Iterative mean of nearby group elements. eps bounds the squared update norm.
/// See Arsigny et al., Bi-invariant Means in Lie Groups, section 4.2, and
/// Manton, A globally convergent numerical algorithm for computing the center
/// of mass on compact Lie groups.
template <template <typename LieGroup, typename... Args> class Container, typename LieGroup,
          typename... Args>
LieGroup
AverageBiinvariant(const Container<LieGroup, Args...> &points,
                   typename LieGroup::Scalar eps = Constants<typename LieGroup::Scalar>::eps,
                   int max_iterations = 20) {
    using Scalar = typename LieGroup::Scalar;
    using Tangent = typename LieGroup::Tangent;
    HOLISTIC_MOTION_CHECK(!points.empty(), "Points container is empty !");
    LieGroup average = *points.begin();
    if (points.size() == 1)
        return average;
    const Scalar weight = Scalar(1) / Scalar(points.size());
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Tangent update;
        update.SetZero();
        for (const auto &point : points)
            update += point - average;
        update *= weight;
        if (update.Coeffs().squaredNorm() < eps)
            break;
        average += update;
    }
    return average;
}

/// Jacobian-weighted mean of nearby group elements, with a squared update bound.
template <template <typename LieGroup, typename... Args> class Container, typename LieGroup,
          typename... Args>
LieGroup Average(const Container<LieGroup, Args...> &points,
                 typename LieGroup::Scalar eps = Constants<typename LieGroup::Scalar>::eps,
                 int max_iterations = 20) {
    using Scalar = typename LieGroup::Scalar;
    using Tangent = typename LieGroup::Tangent;
    HOLISTIC_MOTION_CHECK(!points.empty(), "Points container is empty !");
    LieGroup average = *points.begin();
    if (points.size() == 1)
        return average;
    const Scalar weight = Scalar(1) / Scalar(points.size());
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Tangent update;
        update.SetZero();
        for (const auto &point : points) {
            const Tangent error = average.Between(point).Log();
            // Jr(error) * error = error, so Jr^T * Jr * error = Jr^T * error.
            update.Coeffs().noalias() += error.Rjac().transpose() * error.Coeffs();
        }
        update *= weight;
        if (update.Coeffs().squaredNorm() < eps)
            break;
        average += update;
    }
    return average;
}

/// Left Frechet iteration for nearby elements; eps bounds the squared metric update.
/// Reference: Arsigny et al., Bi-invariant Means in Lie Groups, page 38.
template <template <typename LieGroup, typename... Args> class Container, typename LieGroup,
          typename... Args>
LieGroup
AverageFrechetLeft(const Container<LieGroup, Args...> &points,
                   typename LieGroup::Scalar eps = Constants<typename LieGroup::Scalar>::eps,
                   int max_iterations = 20) {
    using Scalar = typename LieGroup::Scalar;
    using Tangent = typename LieGroup::Tangent;
    HOLISTIC_MOTION_CHECK(!points.empty(), "Points container is empty !");
    LieGroup average = *points.begin();
    if (points.size() == 1)
        return average;
    const Scalar weight = Scalar(1) / Scalar(points.size());
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        const LieGroup previous = average;
        const Tangent origin = previous.Log();
        const auto jacobian = origin.Ljac();
        Tangent update;
        update.SetZero();
        for (const auto &point : points)
            update += jacobian * (point - previous) * weight;
        average = previous + (origin.Ljacinv() * update);
        const Tangent displacement = jacobian * (average - previous);
        if (displacement.Coeffs().squaredNorm() < eps)
            break;
    }
    return average;
}

/// Right Frechet iteration for nearby elements; eps bounds the squared metric update.
template <template <typename LieGroup, typename... Args> class Container, typename LieGroup,
          typename... Args>
LieGroup
AverageFrechetRight(const Container<LieGroup, Args...> &points,
                    typename LieGroup::Scalar eps = Constants<typename LieGroup::Scalar>::eps,
                    int max_iterations = 20) {
    using Scalar = typename LieGroup::Scalar;
    using Tangent = typename LieGroup::Tangent;
    HOLISTIC_MOTION_CHECK(!points.empty(), "Points container is empty !");
    LieGroup average = *points.begin();
    if (points.size() == 1)
        return average;
    const Scalar weight = Scalar(1) / Scalar(points.size());
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        const LieGroup previous = average;
        const Tangent origin = previous.Log();
        const auto jacobian = origin.Rjac();
        Tangent update;
        update.SetZero();
        for (const auto &point : points)
            update += jacobian * point.Lminus(previous) * weight;
        average = previous.Lplus(origin.Rjacinv() * update);
        const Tangent displacement = jacobian * average.Lminus(previous);
        if (displacement.Coeffs().squaredNorm() < eps)
            break;
    }
    return average;
}

} // namespace holistic_motion::robotics

#endif
