#include "Bindings.h"
#include "RealInputs.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "holistic_motion/robot/JointPositionMap.h"

namespace holistic_motion::python {

void BindCoordinates(pybind11::module_ &module) {
    using robotics::JointCoordinate;
    using robotics::JointPositionMap;
    py::class_<JointCoordinate>(module, "JointCoordinate")
        .def_readonly("name", &JointCoordinate::name)
        .def_readonly("idx_q", &JointCoordinate::idx_q)
        .def_readonly("nq", &JointCoordinate::nq)
        .def_readonly("idx_v", &JointCoordinate::idx_v)
        .def_readonly("nv", &JointCoordinate::nv);
    py::class_<JointPositionMap>(module, "JointPositionMap")
        .def_property_readonly("nq", &JointPositionMap::GetConfigurationSize)
        .def_property_readonly("nv", &JointPositionMap::GetVelocitySize)
        .def_property_readonly("joints",
                               [](const JointPositionMap &self) {
                                   return std::vector<JointCoordinate>(
                                       self.GetJoints());
                               })
        .def(
            "configuration_with_joint_positions",
            [](const JointPositionMap &self, const py::object &reference,
               const py::object &positions) {
                return self.ConfigurationWithJointPositions(
                    CheckedRealEigen<Eigen::VectorXd>(reference, "reference"),
                    CheckedRealEigen<Eigen::VectorXd>(positions,
                                                      "joint_positions"));
            },
            py::arg("reference"), py::arg("joint_positions"))
        .def(
            "joint_positions",
            [](const JointPositionMap &self, const py::object &configuration) {
                return self.JointPositions(CheckedRealEigen<Eigen::VectorXd>(
                    configuration, "configuration"));
            },
            py::arg("configuration"))
        .def(
            "joint_velocities",
            [](const JointPositionMap &self, const py::object &velocity) {
                return self.JointVelocities(
                    CheckedRealEigen<Eigen::VectorXd>(velocity, "velocity"));
            },
            py::arg("velocity"));
}

} // namespace holistic_motion::python
