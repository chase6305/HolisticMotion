#include "TrajectoryBindingHelpers.h"

namespace holistic_motion::python {

using namespace trajectory_detail;

void BindTrajectories(py::module_ &module) {
    py::class_<TrajectoryConstraints, std::shared_ptr<TrajectoryConstraints>>(
        module, "TrajectoryConstraints")
        .def(
            py::init([](const py::object &velocity,
                        const py::object &acceleration,
                        const py::object &jerk) {
                return std::make_shared<TrajectoryConstraints>(
                    CheckedRealEigen<Eigen::VectorXd>(velocity, "max_velocity"),
                    CheckedRealEigen<Eigen::VectorXd>(acceleration,
                                                      "max_acceleration"),
                    CheckedRealEigen<Eigen::VectorXd>(jerk, "max_jerk"));
            }),
            py::arg("max_velocity"), py::arg("max_acceleration"),
            py::arg("max_jerk"))
        .def_property_readonly("valid", &TrajectoryConstraints::IsValid);

    const auto report_keys = std::make_shared<const ReportKeys>();
    const std::array<JointTrajectoryFactoryBlock, 4> blocks{
        BindJointTrajectories1To8(module, report_keys),
        BindJointTrajectories9To16(module, report_keys),
        BindJointTrajectories17To24(module, report_keys),
        BindJointTrajectories25To32(module, report_keys)};
    std::array<JointTrajectoryFactory, 32> factories;
    for (std::size_t index = 0; index < blocks.size(); ++index)
        std::copy(blocks[index].begin(), blocks[index].end(),
                  factories.begin() + 8 * index);
    const auto joint_factory =
        [factories](const Eigen::MatrixXd &waypoints,
                    const Eigen::VectorXd &max_velocity,
                    const Eigen::VectorXd &max_acceleration,
                    const Eigen::VectorXd &max_jerk, double blend_tolerance,
                    double minimum_duration, const std::string &profile) {
            if (waypoints.cols() < 1 || waypoints.cols() > 32)
                throw std::invalid_argument(
                    "joint trajectory supports waypoint DOF from 1 through 32");
            return factories[static_cast<std::size_t>(waypoints.cols() - 1)](
                waypoints, max_velocity, max_acceleration, max_jerk,
                blend_tolerance, minimum_duration, profile);
        };
    module.def("JointTrajectory", CheckedTrajectoryFactory(joint_factory),
               py::arg("waypoints"), py::arg("max_velocity"),
               py::arg("max_acceleration"), py::arg("max_jerk"),
               py::arg("blend_tolerance") = 0.0,
               py::arg("minimum_duration") = 0.0,
               py::arg("profile") = "double_s",
               "Construct a 1-32 DoF joint/configuration trajectory from its "
               "waypoint columns.");
    module.def(
        "RnTrajectory", CheckedTrajectoryFactory(joint_factory),
        py::arg("waypoints"), py::arg("max_velocity"),
        py::arg("max_acceleration"), py::arg("max_jerk"),
        py::arg("blend_tolerance") = 0.0, py::arg("minimum_duration") = 0.0,
        py::arg("profile") = "double_s",
        "Construct an R^n trajectory and infer n (1-32) from the waypoint "
        "columns.");
    module.def("BaseTrajectory", CheckedTrajectoryFactory(factories[2]),
               py::arg("waypoints"), py::arg("max_velocity"),
               py::arg("max_acceleration"), py::arg("max_jerk"),
               py::arg("blend_tolerance") = 0.0,
               py::arg("minimum_duration") = 0.0,
               py::arg("profile") = "double_s",
               "Construct an [x, y, yaw] mobile-base trajectory.");
    BindCartesianTrajectory(module, report_keys);
}

} // namespace holistic_motion::python
