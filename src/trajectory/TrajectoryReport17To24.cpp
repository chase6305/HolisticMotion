#include "TrajectoryInstantiationBatches.h"
#include "TrajectoryReportImpl.h"

namespace holistic_motion::robotics {
#define INSTANTIATE_REPORT(group, type)                                                \
    template group<type>::ConstraintReport group<type>::GetConstraintReport(           \
        std::size_t) const;
HOLISTIC_MOTION_TRAJECTORY_TYPES_17_TO_24(INSTANTIATE_REPORT, TrajectoryBase)
#undef INSTANTIATE_REPORT
} // namespace holistic_motion::robotics
