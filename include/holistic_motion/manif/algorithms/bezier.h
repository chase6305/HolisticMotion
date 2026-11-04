#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_BEZIER_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_BEZIER_H_

#include "holistic_motion/manif/algorithms/decasteljau.h"

namespace holistic_motion::robotics {

/// Compatibility entry point for the common geodesic Bezier sampler.
/// degree counts control points per segment; see Decasteljau().
template <typename LieGroup>
std::vector<typename LieGroup::LieGroup>
ComputeBezierCurve(const std::vector<LieGroup> &control_points, const unsigned int degree,
                   const unsigned int k_interp) {
    return Decasteljau(control_points, degree, k_interp);
}

} // namespace holistic_motion::robotics

#endif
