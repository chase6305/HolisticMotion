#include "Bindings.h"
#include "PlanningCallbacks.h"
#include "RealInputs.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"
namespace holistic_motion::python {

void BindSamplingPlanning(pybind11::module_ &module) {
    namespace py = pybind11;
    using robotics::planning::JointSpace;
    using robotics::planning::PathOptimizationOptions;
    using robotics::planning::PathOptimizationResult;
    using robotics::planning::PathOptimizationStatistics;
    using robotics::planning::PathOptimizationStatus;
    using robotics::planning::PathOptimizer;
    using robotics::planning::PlanningOptions;
    using robotics::planning::PlanningResult;
    using robotics::planning::PlanningStatistics;
    using robotics::planning::PlanningStatus;
    using robotics::planning::SamplingAlgorithm;
    using robotics::planning::SamplingPlanner;

    py::enum_<PathOptimizationStatus>(module, "PathOptimizationStatus")
        .value("OPTIMIZED", PathOptimizationStatus::OPTIMIZED)
        .value("UNCHANGED", PathOptimizationStatus::UNCHANGED)
        .value("TIMEOUT", PathOptimizationStatus::TIMEOUT)
        .value("INVALID_PATH", PathOptimizationStatus::INVALID_PATH);
    py::class_<JointSpace>(module, "JointSpace")
        .def(py::init([](const py::object &lower, const py::object &upper,
                         std::vector<std::string> names) {
                 return JointSpace(
                     CheckedRealEigen<Eigen::VectorXd>(lower, "lower_limits"),
                     CheckedRealEigen<Eigen::VectorXd>(upper, "upper_limits"),
                     std::move(names));
             }),
             py::arg("lower_limits"), py::arg("upper_limits"),
             py::arg("joint_names") = std::vector<std::string>{})
        .def_property_readonly("dof", &JointSpace::GetDOF)
        .def_property_readonly("lower_limits",
                               [](const JointSpace &space) {
                                   return Eigen::VectorXd(space.GetLowerLimits());
                               })
        .def_property_readonly("upper_limits",
                               [](const JointSpace &space) {
                                   return Eigen::VectorXd(space.GetUpperLimits());
                               })
        .def_property_readonly("joint_weights",
                               [](const JointSpace &space) {
                                   return Eigen::VectorXd(space.GetJointWeights());
                               })
        .def_property_readonly(
            "joint_names",
            [](const JointSpace &space) { return space.GetJointNames(); })
        .def_property_readonly("continuous_joint_indices",
                               &JointSpace::GetContinuousJoints)
        .def(
            "with_joint_weights",
            [](const JointSpace &space, const py::object &weights) {
                return space.WithJointWeights(
                    CheckedRealEigen<Eigen::VectorXd>(weights, "weights"));
            },
            py::arg("weights"))
        .def(
            "with_continuous_joints",
            [](const JointSpace &space, const py::object &indices) {
                return space.WithContinuousJoints(
                    CheckedRealEigen<std::vector<std::size_t>>(indices, "indices"));
            },
            py::arg("indices"))
        .def(
            "unwrap_path",
            [](const JointSpace &space, const py::object &path) {
                return space.UnwrapPath(
                    CheckedRealEigen<std::vector<Eigen::VectorXd>>(path, "path"));
            },
            py::arg("path"));

    py::class_<PathOptimizationOptions>(module, "PathOptimizationOptions")
        .def(py::init<>())
        .def_property(
            "max_iterations",
            ReadParameter(&PathOptimizationOptions::max_iterations),
            WriteRealParameter(&PathOptimizationOptions::max_iterations,
                               "max_iterations"))
        .def_property(
            "timeout_seconds",
            ReadParameter(&PathOptimizationOptions::timeout_seconds),
            WriteRealParameter(&PathOptimizationOptions::timeout_seconds,
                               "timeout_seconds"))
        .def_property("step_size",
                      ReadParameter(&PathOptimizationOptions::step_size),
                      WriteRealParameter(&PathOptimizationOptions::step_size,
                                         "step_size"))
        .def_property(
            "line_search_steps",
            ReadParameter(&PathOptimizationOptions::line_search_steps),
            WriteRealParameter(&PathOptimizationOptions::line_search_steps,
                               "line_search_steps"))
        .def_property(
            "line_search_decay",
            ReadParameter(&PathOptimizationOptions::line_search_decay),
            WriteRealParameter(&PathOptimizationOptions::line_search_decay,
                               "line_search_decay"))
        .def_property(
            "edge_resolution",
            ReadParameter(&PathOptimizationOptions::edge_resolution),
            WriteRealParameter(&PathOptimizationOptions::edge_resolution,
                               "edge_resolution"))
        .def_property(
            "length_weight",
            ReadParameter(&PathOptimizationOptions::length_weight),
            WriteRealParameter(&PathOptimizationOptions::length_weight,
                               "length_weight"))
        .def_property(
            "smoothness_weight",
            ReadParameter(&PathOptimizationOptions::smoothness_weight),
            WriteRealParameter(&PathOptimizationOptions::smoothness_weight,
                               "smoothness_weight"))
        .def_property(
            "state_cost_weight",
            ReadParameter(&PathOptimizationOptions::state_cost_weight),
            WriteRealParameter(&PathOptimizationOptions::state_cost_weight,
                               "state_cost_weight"))
        .def_property(
            "finite_difference_step",
            ReadParameter(&PathOptimizationOptions::finite_difference_step),
            WriteRealParameter(&PathOptimizationOptions::finite_difference_step,
                               "finite_difference_step"))
        .def_property(
            "state_cost_step_size",
            ReadParameter(&PathOptimizationOptions::state_cost_step_size),
            WriteRealParameter(&PathOptimizationOptions::state_cost_step_size,
                               "state_cost_step_size"))
        .def_property(
            "minimum_improvement",
            ReadParameter(&PathOptimizationOptions::minimum_improvement),
            WriteRealParameter(&PathOptimizationOptions::minimum_improvement,
                               "minimum_improvement"));
    py::class_<PathOptimizationStatistics>(module, "PathOptimizationStatistics")
        .def_readonly("iterations", &PathOptimizationStatistics::iterations)
        .def_readonly("attempted_updates",
                      &PathOptimizationStatistics::attempted_updates)
        .def_readonly("line_search_evaluations",
                      &PathOptimizationStatistics::line_search_evaluations)
        .def_readonly("accepted_updates",
                      &PathOptimizationStatistics::accepted_updates)
        .def_readonly("collision_checks",
                      &PathOptimizationStatistics::collision_checks)
        .def_readonly("state_cost_evaluations",
                      &PathOptimizationStatistics::state_cost_evaluations)
        .def_readonly(
            "state_cost_gradient_evaluations",
            &PathOptimizationStatistics::state_cost_gradient_evaluations)
        .def_readonly("optimization_time_ms",
                      &PathOptimizationStatistics::optimization_time_ms)
        .def_readonly("initial_objective",
                      &PathOptimizationStatistics::initial_objective)
        .def_readonly("final_objective",
                      &PathOptimizationStatistics::final_objective)
        .def_readonly("initial_path_length",
                      &PathOptimizationStatistics::initial_path_length)
        .def_readonly("final_path_length",
                      &PathOptimizationStatistics::final_path_length);
    py::class_<PathOptimizationResult>(module, "PathOptimizationResult")
        .def_readonly("joint_space", &PathOptimizationResult::joint_space)
        .def_property_readonly("success", &PathOptimizationResult::Success)
        .def_readonly("status", &PathOptimizationResult::status)
        .def_readonly("path", &PathOptimizationResult::path)
        .def_readonly("statistics", &PathOptimizationResult::statistics)
        .def_readonly("feasible", &PathOptimizationResult::feasible)
        .def_readonly("message", &PathOptimizationResult::message);
    py::class_<PathOptimizer>(
        module, "PathOptimizer", py::dynamic_attr(),
        py::custom_type_setup(PlanningGarbageCollection<PathOptimizer>::Setup))
        .def(py::init([](JointSpace space, const py::object &validator) {
                 return PathOptimizer(std::move(space), PlanningValidator(validator));
             }),
             py::arg("joint_space"), py::arg("state_validator") = py::none())
        .def_property_readonly("joint_space", &PathOptimizer::GetJointSpace)
        .def(py::init([](const py::object &lower, const py::object &upper,
                         const py::object &validator) {
                 return PathOptimizer(
                     CheckedRealEigen<Eigen::VectorXd>(lower, "lower_limits"),
                     CheckedRealEigen<Eigen::VectorXd>(upper, "upper_limits"),
                     PlanningValidator(validator));
             }),
             py::arg("lower_limits"), py::arg("upper_limits"),
             py::arg("state_validator") = py::none())
        .def(
            "set_state_validator",
            [](PathOptimizer &self, const py::object &callback) {
                self.SetStateValidator(PlanningValidator(callback));
            },
            py::arg("validator"))
        .def(
            "set_state_cost",
            [](PathOptimizer &self, const py::object &callback) {
                if (callback.is_none()) {
                    self.SetStateCost({});
                    return;
                }
                self.SetStateCost(PlanningPythonCallback<double>{
                    PlanningPythonFunction(callback.cast<py::function>()),
                    "state_cost result"});
            },
            py::arg("state_cost"))
        .def(
            "set_state_cost_gradient",
            [](PathOptimizer &self, const py::object &callback) {
                if (callback.is_none()) {
                    self.SetStateCostGradient({});
                    return;
                }
                self.SetStateCostGradient(PlanningPythonCallback<Eigen::VectorXd>{
                    PlanningPythonFunction(callback.cast<py::function>()),
                    "state_cost_gradient result"});
            },
            py::arg("state_cost_gradient"))
        .def(
            "set_joint_weights",
            [](PathOptimizer &self, const py::object &weights) {
                self.SetJointWeights(
                    CheckedRealEigen<Eigen::VectorXd>(weights, "weights"));
            },
            py::arg("weights"))
        .def(
            "set_continuous_joints",
            [](PathOptimizer &self, const py::object &indices) {
                self.SetContinuousJoints(
                    CheckedRealEigen<std::vector<std::size_t>>(indices, "indices"));
            },
            py::arg("indices"))
        .def(
            "optimize",
            [](const PathOptimizer &self, const py::object &path,
               const PathOptimizationOptions &options, bool release_gil) {
                const auto states =
                    CheckedRealEigen<std::vector<Eigen::VectorXd>>(path, "path");
                if (!release_gil)
                    return self.Optimize(states, options);
                const PathOptimizer request = self;
                const PathOptimizationOptions request_options = options;
                py::gil_scoped_release release;
                return request.Optimize(states, request_options);
            },
            py::arg("path"), py::arg("options") = PathOptimizationOptions{},
            py::kw_only(), py::arg("release_gil") = false);

    py::enum_<SamplingAlgorithm>(module, "SamplingAlgorithm")
        .value("RRT_CONNECT", SamplingAlgorithm::RRT_CONNECT)
        .value("RRT_STAR", SamplingAlgorithm::RRT_STAR)
        .value("INFORMED_RRT_STAR", SamplingAlgorithm::INFORMED_RRT_STAR);
    py::enum_<PlanningStatus>(module, "PlanningStatus")
        .value("EXACT_SOLUTION", PlanningStatus::EXACT_SOLUTION)
        .value("TIMEOUT", PlanningStatus::TIMEOUT)
        .value("INVALID_START", PlanningStatus::INVALID_START)
        .value("INVALID_GOAL", PlanningStatus::INVALID_GOAL)
        .value("INVALID_PROBLEM", PlanningStatus::INVALID_PROBLEM)
        .value("NO_SOLUTION", PlanningStatus::NO_SOLUTION);
    py::class_<PlanningOptions>(module, "PlanningOptions")
        .def(py::init<>())
        .def_readwrite("algorithm", &PlanningOptions::algorithm)
        .def_property("timeout_seconds",
                      ReadParameter(&PlanningOptions::timeout_seconds),
                      WriteRealParameter(&PlanningOptions::timeout_seconds,
                                         "timeout_seconds"))
        .def_property("max_iterations",
                      ReadParameter(&PlanningOptions::max_iterations),
                      WriteRealParameter(&PlanningOptions::max_iterations,
                                         "max_iterations"))
        .def_property("extension_range",
                      ReadParameter(&PlanningOptions::extension_range),
                      WriteRealParameter(&PlanningOptions::extension_range,
                                         "extension_range"))
        .def_property(
            "goal_bias", ReadParameter(&PlanningOptions::goal_bias),
            WriteRealParameter(&PlanningOptions::goal_bias, "goal_bias"))
        .def_property("edge_resolution",
                      ReadParameter(&PlanningOptions::edge_resolution),
                      WriteRealParameter(&PlanningOptions::edge_resolution,
                                         "edge_resolution"))
        .def_property("simplify_path",
                      ReadParameter(&PlanningOptions::simplify_path),
                      WriteRealParameter(&PlanningOptions::simplify_path,
                                         "simplify_path"))
        .def_property("shortcut_attempts",
                      ReadParameter(&PlanningOptions::shortcut_attempts),
                      WriteRealParameter(&PlanningOptions::shortcut_attempts,
                                         "shortcut_attempts"))
        .def_property("interpolate_path",
                      ReadParameter(&PlanningOptions::interpolate_path),
                      WriteRealParameter(&PlanningOptions::interpolate_path,
                                         "interpolate_path"))
        .def_property("interpolation_points",
                      ReadParameter(&PlanningOptions::interpolation_points),
                      WriteRealParameter(&PlanningOptions::interpolation_points,
                                         "interpolation_points"))
        .def_property(
            "random_seed", ReadParameter(&PlanningOptions::random_seed),
            WriteRealParameter(&PlanningOptions::random_seed, "random_seed"));
    py::class_<PlanningStatistics>(module, "PlanningStatistics")
        .def_readonly("iterations", &PlanningStatistics::iterations)
        .def_readonly("sampled_states", &PlanningStatistics::sampled_states)
        .def_readonly("valid_states", &PlanningStatistics::valid_states)
        .def_readonly("collision_checks", &PlanningStatistics::collision_checks)
        .def_readonly("tree_nodes", &PlanningStatistics::tree_nodes)
        .def_readonly("planning_time_ms", &PlanningStatistics::planning_time_ms)
        .def_readonly("initial_path_length",
                      &PlanningStatistics::initial_path_length)
        .def_readonly("final_path_length",
                      &PlanningStatistics::final_path_length);
    py::class_<PlanningResult>(module, "PlanningResult")
        .def_readonly("joint_space", &PlanningResult::joint_space)
        .def_property_readonly("success", &PlanningResult::Success)
        .def_readonly("status", &PlanningResult::status)
        .def_readonly("path", &PlanningResult::path)
        .def_readonly("statistics", &PlanningResult::statistics)
        .def_readonly("message", &PlanningResult::message);
    py::class_<SamplingPlanner>(
        module, "SamplingPlanner", py::dynamic_attr(),
        py::custom_type_setup(PlanningGarbageCollection<SamplingPlanner>::Setup))
        .def(py::init([](JointSpace space, const py::object &validator) {
                 return SamplingPlanner(std::move(space), PlanningValidator(validator));
             }),
             py::arg("joint_space"), py::arg("state_validator") = py::none())
        .def_property_readonly("joint_space", &SamplingPlanner::GetJointSpace)
        .def(py::init([](const py::object &lower, const py::object &upper,
                         const py::object &validator) {
                 return SamplingPlanner(
                     CheckedRealEigen<Eigen::VectorXd>(lower, "lower_limits"),
                     CheckedRealEigen<Eigen::VectorXd>(upper, "upper_limits"),
                     PlanningValidator(validator));
             }),
             py::arg("lower_limits"), py::arg("upper_limits"),
             py::arg("state_validator") = py::none())
        .def(
            "set_state_validator",
            [](SamplingPlanner &self, const py::object &callback) {
                self.SetStateValidator(PlanningValidator(callback));
            },
            py::arg("validator"))
        .def(
            "set_joint_weights",
            [](SamplingPlanner &self, const py::object &weights) {
                self.SetJointWeights(
                    CheckedRealEigen<Eigen::VectorXd>(weights, "weights"));
            },
            py::arg("weights"))
        .def(
            "set_continuous_joints",
            [](SamplingPlanner &self, const py::object &indices) {
                self.SetContinuousJoints(
                    CheckedRealEigen<std::vector<std::size_t>>(indices, "indices"));
            },
            py::arg("indices"))
        .def(
            "plan",
            [](const SamplingPlanner &self, const py::object &start,
               const py::object &goal, const PlanningOptions &options,
               bool release_gil) {
                const auto start_state =
                    CheckedRealEigen<Eigen::VectorXd>(start, "start");
                const auto goal_state = CheckedRealEigen<Eigen::VectorXd>(goal, "goal");
                if (!release_gil)
                    return self.Plan(start_state, goal_state, options);
                const SamplingPlanner request = self;
                const PlanningOptions request_options = options;
                py::gil_scoped_release release;
                return request.Plan(start_state, goal_state, request_options);
            },
            py::arg("start"), py::arg("goal"), py::arg("options") = PlanningOptions{},
            py::kw_only(), py::arg("release_gil") = false);
}

} // namespace holistic_motion::python
