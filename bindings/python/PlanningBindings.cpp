#include "Bindings.h"

#include <cmath>
#include <set>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "holistic_motion/planning/SamplingPlanner.h"
#ifdef HOLISTICMOTION_HAS_COLLISION
#include "holistic_motion/collision/CollisionModel.h"
#include "holistic_motion/collision/SphereCollisionModel.h"
#endif

namespace holistic_motion::python {

void BindSamplingPlanning(pybind11::module_ &module) {
  namespace py = pybind11;
  using robotics::planning::PlanningOptions;
  using robotics::planning::PlanningResult;
  using robotics::planning::PlanningStatistics;
  using robotics::planning::PlanningStatus;
  using robotics::planning::SamplingAlgorithm;
  using robotics::planning::SamplingPlanner;

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
      .def_readwrite("timeout_seconds", &PlanningOptions::timeout_seconds)
      .def_readwrite("max_iterations", &PlanningOptions::max_iterations)
      .def_readwrite("extension_range", &PlanningOptions::extension_range)
      .def_readwrite("goal_bias", &PlanningOptions::goal_bias)
      .def_readwrite("edge_resolution", &PlanningOptions::edge_resolution)
      .def_readwrite("simplify_path", &PlanningOptions::simplify_path)
      .def_readwrite("shortcut_attempts", &PlanningOptions::shortcut_attempts)
      .def_readwrite("interpolate_path", &PlanningOptions::interpolate_path)
      .def_readwrite("interpolation_points",
                     &PlanningOptions::interpolation_points)
      .def_readwrite("random_seed", &PlanningOptions::random_seed);
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
      .def_property_readonly("success", &PlanningResult::Success)
      .def_readonly("status", &PlanningResult::status)
      .def_readonly("path", &PlanningResult::path)
      .def_readonly("statistics", &PlanningResult::statistics)
      .def_readonly("message", &PlanningResult::message);
  auto planner =
      py::class_<SamplingPlanner>(module, "SamplingPlanner")
          .def(py::init<Eigen::VectorXd, Eigen::VectorXd,
                        SamplingPlanner::StateValidator>(),
               py::arg("lower_limits"), py::arg("upper_limits"),
               py::arg("state_validator") = SamplingPlanner::StateValidator{})
          .def("set_state_validator", &SamplingPlanner::SetStateValidator,
               py::arg("validator"))
          .def("set_joint_weights", &SamplingPlanner::SetJointWeights,
               py::arg("weights"))
          .def("set_continuous_joints", &SamplingPlanner::SetContinuousJoints,
               py::arg("indices"))
          .def("plan", &SamplingPlanner::Plan, py::arg("start"),
               py::arg("goal"), py::arg("options") = PlanningOptions{});
#ifdef HOLISTICMOTION_HAS_COLLISION
  planner.def_static(
      "from_sphere_collision_model",
      [](Eigen::VectorXd lower, Eigen::VectorXd upper,
         robotics::collision::SphereCollisionModel &collision_model,
         double security_margin) {
        if (!std::isfinite(security_margin) || security_margin < 0.0) {
          throw std::invalid_argument(
              "security_margin must be finite and non-negative");
        }
        if (lower.size() != collision_model.GetConfigurationSize()) {
          throw std::invalid_argument(
              "full-space limits must match sphere collision model nq");
        }
        return SamplingPlanner(
            std::move(lower), std::move(upper),
            [&collision_model, security_margin](const Eigen::VectorXd &state) {
              return !collision_model.InCollision(state, security_margin, true);
            });
      },
      py::arg("lower_limits"), py::arg("upper_limits"),
      py::arg("collision_model"), py::arg("security_margin") = 0.0,
      py::keep_alive<0, 3>());
  planner.def_static(
      "from_collision_model",
      [](Eigen::VectorXd lower, Eigen::VectorXd upper,
         robotics::collision::CollisionModel &collision_model,
         std::vector<std::size_t> active_indices,
         Eigen::VectorXd context_configuration) {
        if (active_indices.empty()) {
          if (lower.size() != collision_model.GetConfigurationSize()) {
            throw std::invalid_argument(
                "full-space limits must match collision model nq");
          }
          return SamplingPlanner(
              std::move(lower), std::move(upper),
              [&collision_model](const Eigen::VectorXd &state) {
                return !collision_model.InCollision(state, true);
              });
        }
        if (context_configuration.size() !=
            collision_model.GetConfigurationSize()) {
          throw std::invalid_argument(
              "context_configuration must match collision model nq");
        }
        if (!context_configuration.allFinite()) {
          throw std::invalid_argument(
              "context_configuration must contain finite values");
        }
        if (lower.size() != static_cast<Eigen::Index>(active_indices.size())) {
          throw std::invalid_argument(
              "active limits and active_indices must have equal size");
        }
        std::set<std::size_t> unique_indices;
        for (std::size_t index : active_indices) {
          if (index >= static_cast<std::size_t>(context_configuration.size())) {
            throw std::out_of_range("active configuration index");
          }
          if (!unique_indices.insert(index).second) {
            throw std::invalid_argument("active_indices must be unique");
          }
        }
        return SamplingPlanner(
            std::move(lower), std::move(upper),
            [&collision_model, active_indices = std::move(active_indices),
             context = std::move(context_configuration)](
                const Eigen::VectorXd &active) mutable {
              Eigen::VectorXd full = context;
              for (Eigen::Index i = 0; i < active.size(); ++i) {
                full[static_cast<Eigen::Index>(
                    active_indices[static_cast<std::size_t>(i)])] = active[i];
              }
              return !collision_model.InCollision(full, true);
            });
      },
      py::arg("lower_limits"), py::arg("upper_limits"),
      py::arg("collision_model"),
      py::arg("active_indices") = std::vector<std::size_t>{},
      py::arg("context_configuration") = Eigen::VectorXd{},
      py::keep_alive<0, 3>());
  planner.def_static(
      "from_collision_joints",
      [](robotics::collision::CollisionModel &collision_model,
         std::vector<std::string> joint_names,
         Eigen::VectorXd context_configuration, double security_margin) {
        if (!std::isfinite(security_margin) || security_margin < 0.0) {
          throw std::invalid_argument(
              "security_margin must be finite and non-negative");
        }
        if (context_configuration.size() !=
                collision_model.GetConfigurationSize() ||
            !context_configuration.allFinite()) {
          throw std::invalid_argument("context_configuration must be finite "
                                      "and match collision model nq");
        }
        Eigen::VectorXd lower =
            collision_model.GetJointLowerLimits(joint_names);
        Eigen::VectorXd upper =
            collision_model.GetJointUpperLimits(joint_names);
        return SamplingPlanner(
            std::move(lower), std::move(upper),
            [&collision_model, joint_names = std::move(joint_names),
             context = std::move(context_configuration),
             security_margin](const Eigen::VectorXd &active) {
              Eigen::VectorXd full =
                  collision_model.ConfigurationWithJointPositions(
                      context, joint_names, active);
              return security_margin > 0.0
                         ? !collision_model.IsWithinDistance(
                               full, security_margin, true)
                         : !collision_model.InCollision(full, true);
            });
      },
      py::arg("collision_model"), py::arg("joint_names"),
      py::arg("context_configuration"), py::arg("security_margin") = 0.0,
      py::keep_alive<0, 1>());
#endif
}

} // namespace holistic_motion::python
