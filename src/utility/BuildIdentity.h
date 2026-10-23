#pragma once

#include "BuildIdentityConfig.h"

namespace holistic_motion::utility::detail {

// Resolve through the linked core library, rather than assuming its objects
// came from the same build as a Python extension's headers.
const char *CoreBuildIdentity() noexcept;

} // namespace holistic_motion::utility::detail

namespace holistic_motion::robotics::collision::detail {

const char *CollisionBuildIdentity() noexcept;

} // namespace holistic_motion::robotics::collision::detail
