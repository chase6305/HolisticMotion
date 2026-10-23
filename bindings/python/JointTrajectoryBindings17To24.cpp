#include "JointTrajectoryBindings.h"

namespace holistic_motion::python::trajectory_detail {

JointTrajectoryFactoryBlock
BindJointTrajectories17To24(py::module_ &module,
                            const std::shared_ptr<const ReportKeys> &keys) {
    return BindJointTrajectoryRange<17>(module, keys,
                                        std::make_index_sequence<8>{});
}

} // namespace holistic_motion::python::trajectory_detail
