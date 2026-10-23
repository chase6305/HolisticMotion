#include "PathSegmentImpl.h"
#include "TrajectoryInstantiationBatches.h"

namespace holistic_motion::robotics {
HOLISTIC_MOTION_TRAJECTORY_TYPES_9_TO_16(HOLISTIC_MOTION_INSTANTIATE_TRAJECTORY_CLASS,
                                         PathSegmentBase)
HOLISTIC_MOTION_TRAJECTORY_TYPES_9_TO_16(HOLISTIC_MOTION_INSTANTIATE_TRAJECTORY_CLASS,
                                         PathSegLinear)
HOLISTIC_MOTION_TRAJECTORY_TYPES_9_TO_16(HOLISTIC_MOTION_INSTANTIATE_TRAJECTORY_CLASS,
                                         PathSegBezierCurve2nd)
HOLISTIC_MOTION_TRAJECTORY_TYPES_9_TO_16(HOLISTIC_MOTION_INSTANTIATE_TRAJECTORY_CLASS,
                                         PathSegBezierCurve5th)
} // namespace holistic_motion::robotics
