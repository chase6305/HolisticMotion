#pragma once

#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

namespace holistic_motion::robotics {

/// Canonical name for the acceleration-limited trapezoidal profile.
/// TrajectoryTrapezium remains available for source compatibility.
template <typename LieGroup>
using TrajectoryTrapezoidal = TrajectoryTrapezium<LieGroup>;

}  // namespace holistic_motion::robotics
