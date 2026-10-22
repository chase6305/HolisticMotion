#include "BindingTransforms.h"
#include "Bindings.h"
#include "RealInputs.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "holistic_motion/kinematics/fep/FEPKinematics.h"
#include "holistic_motion/kinematics/srs/SRSKinematics.h"
#include "holistic_motion/planning/NullSpacePlanner.h"

namespace holistic_motion::python {
using namespace robotics;

void BindSevenAxisKinematics(py::module_ &module) {
    py::enum_<SRSSolveMethod>(module, "SRSSolveMethod")
        .value("SEEDED_NUMERICAL", SRSSolveMethod::SEEDED_NUMERICAL)
        .value("CONFIGURATION", SRSSolveMethod::CONFIGURATION)
        .value("ALL_CONFIGURATIONS", SRSSolveMethod::ALL_CONFIGURATIONS)
        .value("NEAREST_REDUNDANCY", SRSSolveMethod::NEAREST_REDUNDANCY);

    py::enum_<SRSSolveStatus>(module, "SRSSolveStatus")
        .value("SUCCESS", SRSSolveStatus::SUCCESS)
        .value("INVALID_INPUT", SRSSolveStatus::INVALID_INPUT)
        .value("INCOMPATIBLE_MODEL", SRSSolveStatus::INCOMPATIBLE_MODEL)
        .value("NO_SOLUTION", SRSSolveStatus::NO_SOLUTION);

    py::class_<SRSConfiguration>(module, "SRSConfiguration")
        .def(py::init<>())
        .def_property(
            "shoulder", ReadParameter(&SRSConfiguration::shoulder),
            WriteRealParameter(&SRSConfiguration::shoulder, "shoulder"))
        .def_property("elbow", ReadParameter(&SRSConfiguration::elbow),
                      WriteRealParameter(&SRSConfiguration::elbow, "elbow"))
        .def_property("wrist", ReadParameter(&SRSConfiguration::wrist),
                      WriteRealParameter(&SRSConfiguration::wrist, "wrist"))
        .def_property(
            "redundancy", ReadParameter(&SRSConfiguration::redundancy),
            WriteRealParameter(&SRSConfiguration::redundancy, "redundancy"));

    py::class_<SRSGeometryAnalysis>(module, "SRSGeometryAnalysis")
        .def_readonly("structurally_compatible",
                      &SRSGeometryAnalysis::structurally_compatible)
        .def_readonly("closed_form_compatible",
                      &SRSGeometryAnalysis::closed_form_compatible)
        .def_readonly("shoulder_center", &SRSGeometryAnalysis::shoulder_center)
        .def_readonly("wrist_center", &SRSGeometryAnalysis::wrist_center)
        .def_readonly("upper_arm_length",
                      &SRSGeometryAnalysis::upper_arm_length)
        .def_readonly("forearm_length", &SRSGeometryAnalysis::forearm_length)
        .def_readonly("shoulder_axis_residual",
                      &SRSGeometryAnalysis::shoulder_axis_residual)
        .def_readonly("wrist_axis_residual",
                      &SRSGeometryAnalysis::wrist_axis_residual)
        .def_readonly("shoulder_orthogonality_residual",
                      &SRSGeometryAnalysis::shoulder_orthogonality_residual)
        .def_readonly("wrist_orthogonality_residual",
                      &SRSGeometryAnalysis::wrist_orthogonality_residual)
        .def_readonly("elbow_angle_offset",
                      &SRSGeometryAnalysis::elbow_angle_offset)
        .def_readonly("elbow_angle_direction",
                      &SRSGeometryAnalysis::elbow_angle_direction);

    py::class_<SRSSolveReport>(module, "SRSSolveReport")
        .def_readonly("status", &SRSSolveReport::status)
        .def_readonly("method", &SRSSolveReport::method)
        .def_readonly("closed_form_compatible",
                      &SRSSolveReport::closed_form_compatible)
        .def_readonly("solutions", &SRSSolveReport::solutions)
        .def_readonly("configurations", &SRSSolveReport::configurations)
        .def_readonly("minimum_singular_values",
                      &SRSSolveReport::minimum_singular_values)
        .def_readonly("minimum_joint_limit_margins",
                      &SRSSolveReport::minimum_joint_limit_margins)
        .def_readonly("joint_limit_margins",
                      &SRSSolveReport::joint_limit_margins)
        .def_readonly("near_singularities", &SRSSolveReport::near_singularities)
        .def_readonly("joint_limit_hits", &SRSSolveReport::joint_limit_hits);

    py::class_<SRSKinematics, NumericalKinematics,
               std::shared_ptr<SRSKinematics>>(module, "SRSKinematics")
        .def_property_readonly("compatible", &SRSKinematics::IsCompatible)
        .def("analyze_geometry", &SRSKinematics::AnalyzeGeometry,
             py::arg("axis_tolerance") = 1e-4)
        .def(
            "configuration",
            [](const SRSKinematics &self, const py::object &joints) {
                return self.GetConfiguration(
                    CheckedRealEigen<Eigen::VectorXd>(joints, "joints"));
            },
            py::arg("joints"))
        .def(
            "solve",
            [](const SRSKinematics &self, const py::object &target_input,
               const py::object &seed_input, SRSSolveMethod method) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                std::vector<Eigen::VectorXd> solutions;
                bool success;
                {
                    py::gil_scoped_release release;
                    success = self.Solve(SE3d(target), seed, method, solutions);
                }
                if (!success) {
                    throw py::value_error(
                        "SRS inverse kinematics did not converge");
                }
                return solutions;
            },
            py::arg("target"), py::arg("seed"),
            py::arg("method") = SRSSolveMethod::NEAREST_REDUNDANCY)
        .def(
            "solve_detailed",
            [](const SRSKinematics &self, const py::object &target_input,
               const py::object &seed_input, SRSSolveMethod method) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                if (!IsRigidTransform(target)) {
                    SRSSolveReport report;
                    report.status = SRSSolveStatus::INVALID_INPUT;
                    report.method = method;
                    return report;
                }
                py::gil_scoped_release release;
                return self.SolveDetailed(SE3d(target), seed, method);
            },
            py::arg("target"), py::arg("seed"),
            py::arg("method") = SRSSolveMethod::NEAREST_REDUNDANCY)
        .def(
            "solve_user",
            [](const SRSKinematics &self, const py::object &target_input,
               const py::object &seed_input, SRSSolveMethod method) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                std::vector<Eigen::VectorXd> solutions;
                const SE3d base_target = self.RemoveUserFrame(SE3d(target));
                bool success;
                {
                    py::gil_scoped_release release;
                    success = self.Solve(base_target, seed, method, solutions);
                }
                if (!success) {
                    throw py::value_error(
                        "SRS inverse kinematics did not converge");
                }
                return solutions;
            },
            py::arg("target"), py::arg("seed"),
            py::arg("method") = SRSSolveMethod::NEAREST_REDUNDANCY)
        .def(
            "solve_detailed_user",
            [](const SRSKinematics &self, const py::object &target_input,
               const py::object &seed_input, SRSSolveMethod method) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                if (!IsRigidTransform(target)) {
                    SRSSolveReport report;
                    report.status = SRSSolveStatus::INVALID_INPUT;
                    report.method = method;
                    return report;
                }
                const SE3d base_target = self.RemoveUserFrame(SE3d(target));
                py::gil_scoped_release release;
                return self.SolveDetailed(base_target, seed, method);
            },
            py::arg("target"), py::arg("seed"),
            py::arg("method") = SRSSolveMethod::NEAREST_REDUNDANCY)
        .def(
            "solve_configuration",
            [](const SRSKinematics &self, const py::object &target_input,
               const SRSConfiguration &config, const py::object &seed_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                const SRSConfiguration configuration = config;
                Eigen::VectorXd solution;
                bool success;
                {
                    py::gil_scoped_release release;
                    success = self.SolveConfiguration(SE3d(target), configuration, seed,
                                                      solution);
                }
                if (!success) {
                    throw py::value_error("SRS configuration is infeasible");
                }
                return solution;
            },
            py::arg("target"), py::arg("configuration"), py::arg("seed"))
        .def(
            "analytic_elbow_seed",
            [](const SRSKinematics &self, const py::object &target_input,
               const SRSConfiguration &config, const py::object &seed_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                Eigen::VectorXd result;
                if (!self.GetAnalyticElbowSeed(SE3d(target), config, seed,
                                               result)) {
                    throw py::value_error(
                        "target is outside the SRS workspace");
                }
                return result;
            },
            py::arg("target"), py::arg("configuration"), py::arg("seed"))
        .def(
            "analytic_solution",
            [](const SRSKinematics &self, const py::object &target_input,
               const SRSConfiguration &config, const py::object &seed_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                Eigen::VectorXd result;
                if (!self.GetAnalyticSolution(SE3d(target), config, seed,
                                              result)) {
                    throw py::value_error(
                        "no feasible closed-form SRS solution");
                }
                return result;
            },
            py::arg("target"), py::arg("configuration"), py::arg("seed"))
        .def(
            "null_space_velocity",
            [](const SRSKinematics &self, const py::object &joints_input,
               const py::object &direction_input) {
                Eigen::VectorXd joints =
                    CheckedRealEigen<Eigen::VectorXd>(joints_input, "joints");
                Eigen::VectorXd direction = CheckedRealEigen<Eigen::VectorXd>(
                    direction_input, "direction");

                Eigen::VectorXd velocity;
                bool success;
                {
                    py::gil_scoped_release release;
                    success = self.GetNullSpaceVelocity(joints, direction, velocity);
                }
                if (!success) {
                    throw py::value_error("expected a compatible 7R chain");
                }
                return velocity;
            },
            py::arg("joints"), py::arg("preferred_direction"));

    py::enum_<FEPSolveMethod>(module, "FEPSolveMethod")
        .value("SEEDED_NUMERICAL", FEPSolveMethod::SEEDED_NUMERICAL)
        .value("CONFIGURATION", FEPSolveMethod::CONFIGURATION)
        .value("ALL_CONFIGURATIONS", FEPSolveMethod::ALL_CONFIGURATIONS)
        .value("NEAREST_REDUNDANCY", FEPSolveMethod::NEAREST_REDUNDANCY)
        .value("COMPATIBILITY", FEPSolveMethod::COMPATIBILITY);

    py::enum_<FEPBackend>(module, "FEPBackend")
        .value("AUTO", FEPBackend::AUTO)
        .value("CPU", FEPBackend::CPU)
        .value("CUDA", FEPBackend::CUDA);

    py::class_<FEPConfiguration>(module, "FEPConfiguration")
        .def(py::init<>())
        .def_property(
            "shoulder", ReadParameter(&FEPConfiguration::shoulder),
            WriteRealParameter(&FEPConfiguration::shoulder, "shoulder"))
        .def_property("elbow", ReadParameter(&FEPConfiguration::elbow),
                      WriteRealParameter(&FEPConfiguration::elbow, "elbow"))
        .def_property("wrist", ReadParameter(&FEPConfiguration::wrist),
                      WriteRealParameter(&FEPConfiguration::wrist, "wrist"))
        .def_property(
            "redundancy", ReadParameter(&FEPConfiguration::redundancy),
            WriteRealParameter(&FEPConfiguration::redundancy, "redundancy"));

    py::class_<FEPKinematics, NumericalKinematics,
               std::shared_ptr<FEPKinematics>>(module, "FEPKinematics")
        .def_property_readonly("compatible", &FEPKinematics::IsCompatible)
        .def_property_readonly_static(
            "cuda_available",
            [](py::object) { return FEPKinematics::HasCudaBackend(); })
        .def_property_readonly_static(
            "cuda_compiled",
            [](py::object) { return FEPKinematics::IsCudaCompiled(); })
        .def("resolve_backend", &FEPKinematics::ResolveBackend,
             py::arg("backend"), py::arg("batch_size"))
        .def(
            "forward_batch",
            [](const FEPKinematics &self, const py::object &joints_input,
               FEPBackend backend) {
                Eigen::MatrixXd joints =
                    CheckedRealEigen<Eigen::MatrixXd>(joints_input, "joints");

                std::vector<Eigen::Matrix4d> poses;
                bool success;
                {
                    py::gil_scoped_release release;
                    success = self.ForwardBatch(joints, backend, poses);
                }
                if (!success) {
                    throw py::value_error(
                        "batch FK requires finite in-limit (N, 7) joints "
                        "and an available requested backend");
                }
                py::array_t<double> result(py::array::ShapeContainer{
                    static_cast<py::ssize_t>(poses.size()),
                    static_cast<py::ssize_t>(4), static_cast<py::ssize_t>(4)});
                auto values = result.mutable_unchecked<3>();
                for (py::ssize_t pose = 0;
                     pose < static_cast<py::ssize_t>(poses.size()); ++pose)
                    for (py::ssize_t row = 0; row < 4; ++row)
                        for (py::ssize_t col = 0; col < 4; ++col)
                            values(pose, row, col) =
                                poses[static_cast<std::size_t>(pose)](row, col);
                return result;
            },
            py::arg("joints"), py::arg("backend") = FEPBackend::AUTO)
        .def(
            "configuration",
            [](const FEPKinematics &self, const py::object &joints) {
                return self.GetConfiguration(
                    CheckedRealEigen<Eigen::VectorXd>(joints, "joints"));
            },
            py::arg("joints"))
        .def(
            "solve",
            [](const FEPKinematics &self, const py::object &target_input,
               const py::object &seed_input, FEPSolveMethod method) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                std::vector<Eigen::VectorXd> solutions;
                bool success;
                {
                    py::gil_scoped_release release;
                    success = self.Solve(SE3d(target), seed, method, solutions);
                }
                if (!success) {
                    throw py::value_error(
                        "FEP inverse kinematics did not converge");
                }
                return solutions;
            },
            py::arg("target"), py::arg("seed"),
            py::arg("method") = FEPSolveMethod::NEAREST_REDUNDANCY)
        .def(
            "solve_configuration",
            [](const FEPKinematics &self, const py::object &target_input,
               const FEPConfiguration &config, const py::object &seed_input) {
                Eigen::Matrix4d target =
                    CheckedRealEigen<Eigen::Matrix4d>(target_input, "target");
                Eigen::VectorXd seed =
                    CheckedRealEigen<Eigen::VectorXd>(seed_input, "seed");

                RequireRigidTransform(target, "target");
                const FEPConfiguration configuration = config;
                Eigen::VectorXd solution;
                bool success;
                {
                    py::gil_scoped_release release;
                    success = self.SolveConfiguration(SE3d(target), configuration, seed,
                                                      solution);
                }
                if (!success) {
                    throw py::value_error("FEP configuration is infeasible");
                }
                return solution;
            },
            py::arg("target"), py::arg("configuration"), py::arg("seed"));

    py::class_<planning::NullSpacePlanner>(module, "NullSpacePlanner")
        .def(py::init<std::shared_ptr<SRSKinematics>>(), py::arg("kinematics"))
        .def(
            "plan",
            [](const planning::NullSpacePlanner &self,
               const py::object &start_input, const py::object &direction_input,
               const py::object &steps_input, const py::object &step_input) {
                const Eigen::VectorXd start =
                    CheckedRealEigen<Eigen::VectorXd>(start_input, "start");
                const Eigen::VectorXd direction =
                    CheckedRealEigen<Eigen::VectorXd>(direction_input,
                                                      "preferred_direction");
                const int steps = CheckedRealEigen<int>(steps_input, "steps");
                const double step_size =
                    CheckedRealEigen<double>(step_input, "step_size");
                std::vector<Eigen::VectorXd> path;
                bool success;
                {
                    py::gil_scoped_release release;
                    success = self.Plan(start, direction, steps, step_size, path);
                }
                if (!success) {
                    throw py::value_error("null-space planning failed");
                }
                return path;
            },
            py::arg("start"), py::arg("preferred_direction"), py::arg("steps"),
            py::arg("step_size") = 0.05);
}

} // namespace holistic_motion::python
