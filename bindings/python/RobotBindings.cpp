#include "BindingTransforms.h"
#include "Bindings.h"
#include "RealInputs.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "holistic_motion/robot/Robot.h"

namespace holistic_motion::python {
using namespace robotics;

void BindRobotTypes(py::module_ &module) {
    py::enum_<JointType>(module, "JointType")
        .value("UNKNOWN", JointType::UNKNOWN)
        .value("REVOLUTE", JointType::REVOLUTE)
        .value("PRISMATIC", JointType::PRISMATIC)
        .value("CONTINUOUS", JointType::CONTINUOUS)
        .value("FLOATING", JointType::FLOATING)
        .value("PLANAR", JointType::PLANAR)
        .value("FIXED", JointType::FIXED);

    py::class_<JointLimit, std::shared_ptr<JointLimit>>(module, "JointLimit")
        .def_readonly("lower", &JointLimit::lower_limit)
        .def_readonly("upper", &JointLimit::upper_limit)
        .def_readonly("max_velocity", &JointLimit::max_velocity)
        .def_readonly("max_acceleration", &JointLimit::max_acceleration)
        .def_readonly("max_jerk", &JointLimit::max_jerk)
        .def_readonly("max_effort", &JointLimit::max_effort);

    py::class_<Joint, std::shared_ptr<Joint>>(module, "Joint")
        .def_readonly("name", &Joint::name)
        .def_readonly("parent_link", &Joint::parent_link)
        .def_readonly("child_link", &Joint::child_link)
        .def_readonly("joint_type", &Joint::joint_type)
        .def_readonly("axis", &Joint::axis)
        .def_readonly("limit", &Joint::limit)
        .def_readonly("mimic_joint", &Joint::mimic_joint)
        .def_readonly("mimic_multiplier", &Joint::mimic_multiplier)
        .def_readonly("mimic_offset", &Joint::mimic_offset)
        .def_property_readonly("origin", [](const Joint &self) {
            return self.origin_pose.GetTransform();
        });

    py::class_<Inertial>(module, "Inertial")
        .def_readonly("present", &Inertial::present)
        .def_readonly("mass", &Inertial::mass)
        .def_readonly("inertia", &Inertial::inertia)
        .def_property_readonly("origin", [](const Inertial &self) {
            return self.origin.GetTransform();
        });

    py::enum_<GeometryType>(module, "GeometryType")
        .value("MESH", GeometryType::MESH)
        .value("BOX", GeometryType::BOX)
        .value("CYLINDER", GeometryType::CYLINDER)
        .value("SPHERE", GeometryType::SPHERE);

    py::class_<VisualGeometry>(module, "VisualGeometry")
        .def_readonly("name", &VisualGeometry::name)
        .def_readonly("type", &VisualGeometry::type)
        .def_readonly("mesh_path", &VisualGeometry::mesh_path)
        .def_readonly("material_name", &VisualGeometry::material_name)
        .def_readonly("texture_path", &VisualGeometry::texture_path)
        .def_readonly("color", &VisualGeometry::color)
        .def_readonly("has_color", &VisualGeometry::has_color)
        .def_readonly("scale", &VisualGeometry::scale)
        .def_readonly("size", &VisualGeometry::size)
        .def_readonly("radius", &VisualGeometry::radius)
        .def_readonly("length", &VisualGeometry::length)
        .def_property_readonly("origin", [](const VisualGeometry &self) {
            return self.origin.GetTransform();
        });

    py::class_<Link, std::shared_ptr<Link>>(module, "Link")
        .def_readonly("name", &Link::name)
        .def_readonly("inertial", &Link::inertial)
        .def_readonly("visuals", &Link::visuals)
        .def_readonly("parent_joint", &Link::parent_joint)
        .def_readonly("child_joints", &Link::child_joints);
}

void BindRobot(py::module_ &module) {
    py::class_<Robot, std::shared_ptr<Robot>>(module, "Robot")
        .def_property_readonly("nq", &Robot::GetModelDoF)
        .def_property_readonly("nv", &Robot::GetModelDoF)
        .def_property_readonly("joint_coordinates", &Robot::GetJointCoordinates)
        .def("create_joint_position_map", &Robot::CreateJointPositionMap,
             py::arg("joint_names"))
        .def("chain_joint_names", &Robot::GetChainJointNames,
             py::arg("base_link"), py::arg("tip_link"))
        .def(py::init<const std::string &, bool>(), py::arg("urdf_path"),
             py::arg("load_visuals") = false)
        .def_property_readonly("name", &Robot::GetName)
        .def_property_readonly("urdf_path", &Robot::GetURDFPath)
        .def_property_readonly("root_link_name", &Robot::GetRootLinkName)
        .def_property_readonly("dof", &Robot::GetDoF)
        .def_property_readonly("model_dof", &Robot::GetModelDoF)
        .def_property_readonly("joints", &Robot::GetJoints)
        .def_property_readonly("actuated_joints", &Robot::GetActuatedJoints)
        .def_property_readonly("all_actuated_joints",
                               &Robot::GetAllActuatedJoints)
        .def_property_readonly("links", &Robot::GetLinks)
        .def("get_link", &Robot::GetLink, py::arg("name"))
        .def("get_joint", &Robot::GetJoint, py::arg("name"))
        .def("create_kinematics", &Robot::CreateKinematics,
             py::arg("base_link"), py::arg("tip_link"))
        .def("create_fep_kinematics", &Robot::CreateFEPKinematics,
             py::arg("base_link"), py::arg("tip_link"))
        .def_property_readonly("kinematics", &Robot::GetKinematics)
        .def_property_readonly("has_kinematics", &Robot::HasKinematics)
        .def_property_readonly("visuals_loaded", &Robot::HasVisualsLoaded)
        .def("load_visuals", &Robot::LoadVisuals);
}

} // namespace holistic_motion::python
