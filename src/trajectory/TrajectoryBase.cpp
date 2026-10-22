#include "TrajectoryBaseImpl.h"
#include "TrajectoryInstantiationBatches.h"

namespace holistic_motion::robotics {
HOLISTIC_MOTION_TRAJECTORY_TYPES_1_TO_8(HOLISTIC_MOTION_INSTANTIATE_TRAJECTORY_CLASS,
                                        TrajectoryBase)
} // namespace holistic_motion::robotics
