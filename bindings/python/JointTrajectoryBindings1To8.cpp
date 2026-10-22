#include "JointTrajectoryBindings.h"

namespace holistic_motion::python::trajectory_detail {

JointTrajectoryFactoryBlock
BindJointTrajectories1To8(py::module_ &module,
                          const std::shared_ptr<const ReportKeys> &keys) {
    return BindJointTrajectoryRange<1>(module, keys,
                                       std::make_index_sequence<8>{});
}

} // namespace holistic_motion::python::trajectory_detail
