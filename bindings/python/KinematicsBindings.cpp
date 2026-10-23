#include "BindingTransforms.h"
#include "Bindings.h"
#include "RealInputs.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/kinematics/OPWKinematics.h"
#include "holistic_motion/kinematics/URKinematics.h"

namespace holistic_motion::python {
using namespace robotics;

void BindKinematics(py::module_ &module) {
    py::enum_<KinematicsFamily>(module, "KinematicsFamily")
        .value("CUSTOM", KinematicsFamily::CUSTOM)
        .value("NUMERICAL", KinematicsFamily::NUMERICAL)
        .value("OPW", KinematicsFamily::OPW)
        .value("UR", KinematicsFamily::UR)
        .value("SRS", KinematicsFamily::SRS)
        .value("FEP", KinematicsFamily::FEP);
    py::enum_<IKAlgorithm>(module, "IKAlgorithm")
        .value("CUSTOM", IKAlgorithm::CUSTOM)
        .value("NUMERICAL", IKAlgorithm::NUMERICAL)
        .value("ANALYTIC", IKAlgorithm::ANALYTIC);
    py::enum_<IKSelection>(module, "IKSelection")
        .value("NEAREST", IKSelection::NEAREST)
        .value("ALL", IKSelection::ALL);
    py::enum_<IKSolveStatus>(module, "IKSolveStatus")
        .value("SUCCESS", IKSolveStatus::SUCCESS)
        .value("INVALID_INPUT", IKSolveStatus::INVALID_INPUT)
        .value("NO_SOLUTION", IKSolveStatus::NO_SOLUTION);
    py::class_<IKSolveReport>(module, "IKSolveReport")
        .def_readonly("status", &IKSolveReport::status)
        .def_readonly("algorithm", &IKSolveReport::algorithm)
        .def_readonly("selection", &IKSolveReport::selection)
        .def_readonly("solutions", &IKSolveReport::solutions)
        .def_readonly("distances", &IKSolveReport::distances)
        .def_property_readonly("success", &IKSolveReport::Success);
    py::class_<KinematicsCapabilities>(module, "KinematicsCapabilities")
        .def_readonly("family", &KinematicsCapabilities::family)
        .def_readonly("default_ik_algorithm",
                      &KinematicsCapabilities::default_ik_algorithm)
        .def_readonly("dof", &KinematicsCapabilities::dof)
        .def_readonly("supports_jacobian", &KinematicsCapabilities::supports_jacobian)
        .def_readonly("supports_redundancy",
                      &KinematicsCapabilities::supports_redundancy)
        .def_readonly("supports_closed_form_ik",
                      &KinematicsCapabilities::supports_closed_form_ik)
        .def_readonly("supports_batch_fk", &KinematicsCapabilities::supports_batch_fk)
        .def_readonly("cuda_compiled", &KinematicsCapabilities::cuda_compiled)
        .def_readonly("supports_concurrent_queries",
                      &KinematicsCapabilities::supports_concurrent_queries);

    py::class_<SE3d>(module, "SE3")
        .def(py::init<>())
        .def(py::init([](const py::object &transform) {
            return SE3d(
                CheckedRealEigen<Eigen::Matrix4d>(transform, "transform"));
        }))
        .def("matrix", &SE3d::GetTransform)
        .def("inverse", &SE3d::Inverse);

    py::class_<KinematicsBase, std::shared_ptr<KinematicsBase>>(module,
                                                                "AnalyticKinematics")
        .def("dof", &KinematicsBase::GetDOF)
        .def_property_readonly("capabilities", &KinematicsBase::GetCapabilities)
        .def(
            "solve_ik",
            [](const KinematicsBase &self, const py::object &target_input,
               const py::object &seed_input, IKSelection selection) {
                const auto target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                const auto seed = CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");
                if (!IsRigidTransform(target)) {
                    IKSolveReport report;
                    report.status = IKSolveStatus::INVALID_INPUT;
                    report.algorithm = self.GetCapabilities().default_ik_algorithm;
                    report.selection = selection;
                    return report;
                }
                // Legacy custom solvers inherit conservative capabilities and
                // may rely on the calling interpreter. Opt in only when the
                // solver declares concurrent read-only query support.
                if (!self.GetCapabilities().supports_concurrent_queries)
                    return self.SolveIK(SE3d(target), seed, selection);
                py::gil_scoped_release release;
                return self.SolveIK(SE3d(target), seed, selection);
            },
            py::arg("target"), py::arg("seed"),
            py::arg("selection") = IKSelection::NEAREST)
        .def(
            "forward",
            [](const KinematicsBase &self, const py::object &joints_input) {
                Eigen::VectorXd joints =
                    CheckedRealEigen<Eigen::VectorXd>(joints_input, "joints");

                SE3d pose;
                if (!self.GetFK(joints, pose)) {
                    throw py::value_error("invalid analytic FK joint vector");
                }
                return pose.GetTransform();
            },
            py::arg("joints"))
        .def(
            "solve_all",
            [](const KinematicsBase &self, const py::object &target_input,
               const py::object &seed_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                IkRtn solutions;
                std::vector<double> distances;
                if (!self.GetIK(SE3d(target), solutions, seed, distances)) {
                    return std::vector<Eigen::VectorXd>{};
                }
                return solutions.ik_joints;
            },
            py::arg("target"), py::arg("seed"))
        .def(
            "inverse",
            [](const KinematicsBase &self, const py::object &target_input,
               const py::object &seed_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                IkRtn solutions;
                double distance = 0.0;
                if (!self.GetNearestIK(SE3d(target), solutions, seed,
                                       distance) ||
                    solutions.ik_joints.empty()) {
                    throw py::value_error("analytic inverse kinematics failed");
                }
                return solutions.ik_joints.front();
            },
            py::arg("target"), py::arg("seed"));

    // Preserve the historical base name while making its role explicit for
    // generic numerical and analytic consumers alike.
    module.attr("KinematicsBase") = module.attr("AnalyticKinematics");

    py::class_<OPWParameters>(module, "OPWParameters")
        .def(py::init<>())
        .def_property("a1", ReadParameter(&OPWParameters::a1),
                      WriteRealParameter(&OPWParameters::a1, "a1"))
        .def_property("a2", ReadParameter(&OPWParameters::a2),
                      WriteRealParameter(&OPWParameters::a2, "a2"))
        .def_property("b", ReadParameter(&OPWParameters::b),
                      WriteRealParameter(&OPWParameters::b, "b"))
        .def_property("c1", ReadParameter(&OPWParameters::c1),
                      WriteRealParameter(&OPWParameters::c1, "c1"))
        .def_property("c2", ReadParameter(&OPWParameters::c2),
                      WriteRealParameter(&OPWParameters::c2, "c2"))
        .def_property("c3", ReadParameter(&OPWParameters::c3),
                      WriteRealParameter(&OPWParameters::c3, "c3"))
        .def_property("c4", ReadParameter(&OPWParameters::c4),
                      WriteRealParameter(&OPWParameters::c4, "c4"))
        .def_property("offsets", ReadParameter(&OPWParameters::offsets),
                      WriteRealParameter(&OPWParameters::offsets, "offsets"))
        .def_property("rotation_directions",
                      ReadParameter(&OPWParameters::rotation_directions),
                      WriteRealParameter(&OPWParameters::rotation_directions,
                                         "rotation_directions"))
        .def_property(
            "base_transform",
            [](const OPWParameters &self) {
                return self.T_b_ob.GetTransform();
            },
            [](OPWParameters &self, const py::object &input) {
                const Eigen::Matrix4d value =
                    CheckedRealEigen<Eigen::Matrix4d>(input, "transform");
                RequireRigidTransform(value, "transform");
                self.T_b_ob = SE3d(value);
            })
        .def_property(
            "tool_transform",
            [](const OPWParameters &self) {
                return self.T_oe_e.GetTransform();
            },
            [](OPWParameters &self, const py::object &input) {
                const Eigen::Matrix4d value =
                    CheckedRealEigen<Eigen::Matrix4d>(input, "transform");
                RequireRigidTransform(value, "transform");
                self.T_oe_e = SE3d(value);
            });

    py::class_<URParameters>(module, "URParameters")
        .def(py::init<>())
        .def_property("d1", ReadParameter(&URParameters::d1),
                      WriteRealParameter(&URParameters::d1, "d1"))
        .def_property("a2", ReadParameter(&URParameters::a2),
                      WriteRealParameter(&URParameters::a2, "a2"))
        .def_property("a3", ReadParameter(&URParameters::a3),
                      WriteRealParameter(&URParameters::a3, "a3"))
        .def_property("d4", ReadParameter(&URParameters::d4),
                      WriteRealParameter(&URParameters::d4, "d4"))
        .def_property("d5", ReadParameter(&URParameters::d5),
                      WriteRealParameter(&URParameters::d5, "d5"))
        .def_property("d6", ReadParameter(&URParameters::d6),
                      WriteRealParameter(&URParameters::d6, "d6"))
        .def_property("offsets", ReadParameter(&URParameters::offsets),
                      WriteRealParameter(&URParameters::offsets, "offsets"))
        .def_property("rotation_directions",
                      ReadParameter(&URParameters::rotation_directions),
                      WriteRealParameter(&URParameters::rotation_directions,
                                         "rotation_directions"))
        .def_property(
            "base_transform",
            [](const URParameters &self) { return self.T_b_ob.GetTransform(); },
            [](URParameters &self, const py::object &input) {
                const Eigen::Matrix4d value =
                    CheckedRealEigen<Eigen::Matrix4d>(input, "transform");
                RequireRigidTransform(value, "transform");
                self.T_b_ob = SE3d(value);
            })
        .def_property(
            "tool_transform",
            [](const URParameters &self) { return self.T_oe_e.GetTransform(); },
            [](URParameters &self, const py::object &input) {
                const Eigen::Matrix4d value =
                    CheckedRealEigen<Eigen::Matrix4d>(input, "transform");
                RequireRigidTransform(value, "transform");
                self.T_oe_e = SE3d(value);
            });

    auto valid_directions = [](const auto &params) {
        return std::all_of(params.rotation_directions.begin(),
                           params.rotation_directions.end(),
                           [](signed char direction) {
                               return direction == 1 || direction == -1;
                           });
    };
    auto valid_offsets = [](const auto &params) {
        return std::all_of(params.offsets.begin(), params.offsets.end(),
                           [](double value) { return std::isfinite(value); });
    };
    auto make_joint_nodes = [](const Eigen::VectorXd &lower,
                               const Eigen::VectorXd &upper) {
        if (lower.size() != 6 || upper.size() != 6 || !lower.allFinite() ||
            !upper.allFinite() || (lower.array() > upper.array()).any()) {
            throw py::value_error(
                "joint limits must be finite length-6 arrays");
        }
        std::vector<JointNode> nodes;
        for (int index = 0; index < 6; ++index) {
            nodes.emplace_back(SE3d(), Eigen::Vector3d::UnitZ(),
                               JointType::REVOLUTE, lower[index], upper[index]);
        }
        return nodes;
    };
    py::class_<OPWKinematics, KinematicsBase, std::shared_ptr<OPWKinematics>>(
        module, "OPWKinematics")
        .def(py::init([make_joint_nodes, valid_directions,
                       valid_offsets](const OPWParameters &params,
                                      const py::object &lower_input,
                                      const py::object &upper_input) {
                 const Eigen::VectorXd lower =
                     CheckedRealEigen<Eigen::VectorXd>(lower_input,
                                                       "lower_limits");
                 const Eigen::VectorXd upper =
                     CheckedRealEigen<Eigen::VectorXd>(upper_input,
                                                       "upper_limits");
                 const bool geometry_valid =
                     std::isfinite(params.a1) && std::isfinite(params.a2) &&
                     std::isfinite(params.b) && std::isfinite(params.c1) &&
                     std::isfinite(params.c2) && std::isfinite(params.c3) &&
                     std::isfinite(params.c4) && std::abs(params.c2) > 1e-12 &&
                     std::hypot(params.a2, params.c3) > 1e-12;
                 if (!geometry_valid || !valid_directions(params) ||
                     !valid_offsets(params)) {
                     throw py::value_error(
                         "invalid OPW geometry, offsets, or axis directions");
                 }
                 auto solver = std::make_shared<OPWKinematics>(
                     params, make_joint_nodes(lower, upper));
                 solver->DisableRobotConfig();
                 return solver;
             }),
             py::arg("parameters"), py::arg("lower_limits"),
             py::arg("upper_limits"));
    py::class_<URKinematics, KinematicsBase, std::shared_ptr<URKinematics>>(
        module, "URKinematics")
        .def(py::init([make_joint_nodes, valid_directions,
                       valid_offsets](const URParameters &params,
                                      const py::object &lower_input,
                                      const py::object &upper_input) {
                 const Eigen::VectorXd lower =
                     CheckedRealEigen<Eigen::VectorXd>(lower_input,
                                                       "lower_limits");
                 const Eigen::VectorXd upper =
                     CheckedRealEigen<Eigen::VectorXd>(upper_input,
                                                       "upper_limits");
                 const bool geometry_valid =
                     std::isfinite(params.d1) && std::isfinite(params.a2) &&
                     std::isfinite(params.a3) && std::isfinite(params.d4) &&
                     std::isfinite(params.d5) && std::isfinite(params.d6) &&
                     std::abs(params.a2) > 1e-12 && std::abs(params.a3) > 1e-12;
                 if (!geometry_valid || !valid_directions(params) ||
                     !valid_offsets(params)) {
                     throw py::value_error(
                         "invalid UR geometry, offsets, or axis directions");
                 }
                 auto solver = std::make_shared<URKinematics>(
                     params, make_joint_nodes(lower, upper));
                 solver->DisableRobotConfig();
                 return solver;
             }),
             py::arg("parameters"), py::arg("lower_limits"),
             py::arg("upper_limits"));

    py::class_<NumericalKinematics, KinematicsBase,
               std::shared_ptr<NumericalKinematics>>(module,
                                                     "NumericalKinematics")
        .def_property(
            "home_joints", &NumericalKinematics::GetHomeJoints,
            [](NumericalKinematics &self, const py::object &joints_input) {
                Eigen::VectorXd joints =
                    CheckedRealEigen<Eigen::VectorXd>(joints_input, "joints");

                if (!self.SetHomeJoints(joints)) {
                    throw py::value_error(
                        "home_joints must match DOF and be finite");
                }
            })
        .def_property_readonly("joint_limits",
                               [](const NumericalKinematics &self) {
                                   Eigen::VectorXd upper;
                                   Eigen::VectorXd lower;
                                   self.GetJointLimits(upper, lower);
                                   return py::make_tuple(lower, upper);
                               })
        .def(
            "is_reachable",
            [](const NumericalKinematics &self,
               const py::object &target_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");

                if (!IsRigidTransform(target))
                    return false;
                return self.IsReachable(SE3d(target));
            },
            py::arg("target"))
        .def("forward",
             [](const NumericalKinematics &self,
                const py::object &joints_input) {
                 Eigen::VectorXd joints =
                     CheckedRealEigen<Eigen::VectorXd>(joints_input, "joints");

                 SE3d pose;
                 if (!self.GetFK(joints, pose)) {
                     throw py::value_error(
                         "joint vector does not match robot DOF");
                 }
                 return pose.GetTransform();
             })
        .def(
            "forward_all",
            [](const NumericalKinematics &self,
               const py::object &joints_input) {
                Eigen::VectorXd joints =
                    CheckedRealEigen<Eigen::VectorXd>(joints_input, "joints");

                std::vector<SE3d> poses;
                if (!self.GetAllFK(joints, poses)) {
                    throw py::value_error(
                        "joint vector does not match robot DOF");
                }
                std::vector<Eigen::Matrix4d> matrices;
                matrices.reserve(poses.size());
                for (const auto &pose : poses) {
                    matrices.push_back(pose.GetTransform());
                }
                return matrices;
            },
            py::arg("joints"))
        .def(
            "forward_user",
            [](const NumericalKinematics &self,
               const py::object &joints_input) {
                Eigen::VectorXd joints =
                    CheckedRealEigen<Eigen::VectorXd>(joints_input, "joints");

                SE3d pose;
                if (!self.GetFK(joints, pose)) {
                    throw py::value_error(
                        "joint vector does not match robot DOF");
                }
                return self.ApplyUserFrame(pose).GetTransform();
            },
            py::arg("joints"))
        .def(
            "inverse",
            [](const NumericalKinematics &self, const py::object &target_input,
               const py::object &seed_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                IkRtn solutions;
                double distance = 0.0;
                if (!self.GetNearestIK(SE3d(target), solutions, seed,
                                       distance) ||
                    solutions.ik_joints.empty()) {
                    throw py::value_error(
                        "inverse kinematics did not converge");
                }
                return solutions.ik_joints.front();
            },
            py::arg("target"), py::arg("seed"))
        .def(
            "inverse_user",
            [](const NumericalKinematics &self, const py::object &target_input,
               const py::object &seed_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                IkRtn solutions;
                double distance = 0.0;
                const SE3d base_target = self.RemoveUserFrame(SE3d(target));
                if (!self.GetNearestIK(base_target, solutions, seed,
                                       distance) ||
                    solutions.ik_joints.empty()) {
                    throw py::value_error(
                        "inverse kinematics did not converge");
                }
                return solutions.ik_joints.front();
            },
            py::arg("target"), py::arg("seed"))
        .def(
            "jacobian",
            [](const NumericalKinematics &self,
               const py::object &joints_input) {
                Eigen::VectorXd joints =
                    CheckedRealEigen<Eigen::VectorXd>(joints_input, "joints");

                Eigen::MatrixXd jacobian;
                if (!self.GetJacobian(joints, jacobian)) {
                    throw py::value_error(
                        "joint vector does not match robot DOF");
                }
                return jacobian;
            },
            py::arg("joints"))
        .def(
            "set_tcp",
            [](NumericalKinematics &self, const py::object &pose_input) {
                Eigen::Matrix4d pose =
                    CheckedRealEigen<Eigen::Matrix4d>(pose_input, "pose");

                RequireRigidTransform(pose, "TCP pose");
                return self.SetTCP(SE3d(pose));
            },
            py::arg("pose"))
        .def(
            "set_user_frame",
            [](NumericalKinematics &self, const py::object &pose_input) {
                Eigen::Matrix4d pose =
                    CheckedRealEigen<Eigen::Matrix4d>(pose_input, "pose");

                RequireRigidTransform(pose, "user frame");
                return self.SetUserFrame(SE3d(pose));
            },
            py::arg("pose"))
        .def_property_readonly("tcp",
                               [](const NumericalKinematics &self) {
                                   return self.GetTCP().GetTransform();
                               })
        .def_property_readonly("user_frame",
                               [](const NumericalKinematics &self) {
                                   return self.GetUserFrame().GetTransform();
                               })
        .def("clear_tcp", &NumericalKinematics::ClearTCP)
        .def("clear_user_frame", &NumericalKinematics::ClearUserFrame);
}

} // namespace holistic_motion::python
