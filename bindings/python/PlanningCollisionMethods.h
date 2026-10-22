#pragma once

#include <array>

namespace holistic_motion::python {

inline constexpr std::array<const char *, 3> kOptimizerCollisionMethods{
    "from_sphere_collision_model", "from_collision_joints",
    "from_sphere_collision_joints"};
inline constexpr std::array<const char *, 4> kPlannerCollisionMethods{
    "from_sphere_collision_model", "from_collision_model", "from_collision_joints",
    "from_sphere_collision_joints"};

} // namespace holistic_motion::python
