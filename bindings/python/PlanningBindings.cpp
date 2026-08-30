#include "Bindings.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"
#ifdef HOLISTICMOTION_HAS_COLLISION
#include "holistic_motion/collision/CollisionModel.h"
#include "holistic_motion/collision/SphereCollisionModel.h"
#endif

namespace holistic_motion::python {

#ifdef HOLISTICMOTION_HAS_COLLISION
namespace {

class SphereDistanceCache {
public:
  explicit SphereDistanceCache(
      robotics::collision::SphereCollisionModel &collision_model)
      : collision_model_(collision_model) {}

  const robotics::collision::SphereDistanceResult &
  CostDistance(const Eigen::VectorXd &state) {
    state_ = state;
    distance_ = collision_model_.MinimumDistance(state);
    pair_revision_ = collision_model_.GetCollisionPairRevision();
    reusable_ = true;
    return distance_;
  }

  double ValidationDistance(const Eigen::VectorXd &state) {
    if (reusable_ && Matches(state)) {
      reusable_ = false;
      return distance_.distance;
    }
    reusable_ = false;
    return collision_model_.MinimumDistance(state).distance;
  }

private:
  bool Matches(const Eigen::VectorXd &state) const {
    return state_.size() == state.size() &&
           pair_revision_ == collision_model_.GetCollisionPairRevision() &&
           (state_.array() == state.array()).all();
  }

  robotics::collision::SphereCollisionModel &collision_model_;
  Eigen::VectorXd state_;
  robotics::collision::SphereDistanceResult distance_;
  std::size_t pair_revision_{0};
  bool reusable_{false};
};

} // namespace
#endif

void BindSamplingPlanning(pybind11::module_ &module) {
  namespace py = pybind11;
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
  py::class_<PathOptimizationOptions>(module, "PathOptimizationOptions")
      .def(py::init<>())
      .def_readwrite("max_iterations", &PathOptimizationOptions::max_iterations)
      .def_readwrite("timeout_seconds",
                     &PathOptimizationOptions::timeout_seconds)
      .def_readwrite("step_size", &PathOptimizationOptions::step_size)
      .def_readwrite("line_search_steps",
                     &PathOptimizationOptions::line_search_steps)
      .def_readwrite("line_search_decay",
                     &PathOptimizationOptions::line_search_decay)
      .def_readwrite("edge_resolution",
                     &PathOptimizationOptions::edge_resolution)
      .def_readwrite("length_weight", &PathOptimizationOptions::length_weight)
      .def_readwrite("smoothness_weight",
                     &PathOptimizationOptions::smoothness_weight)
      .def_readwrite("state_cost_weight",
                     &PathOptimizationOptions::state_cost_weight)
      .def_readwrite("finite_difference_step",
                     &PathOptimizationOptions::finite_difference_step)
      .def_readwrite("state_cost_step_size",
                     &PathOptimizationOptions::state_cost_step_size)
      .def_readwrite("minimum_improvement",
                     &PathOptimizationOptions::minimum_improvement);
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
      .def_property_readonly("success", &PathOptimizationResult::Success)
      .def_readonly("status", &PathOptimizationResult::status)
      .def_readonly("path", &PathOptimizationResult::path)
      .def_readonly("statistics", &PathOptimizationResult::statistics)
      .def_readonly("feasible", &PathOptimizationResult::feasible)
      .def_readonly("message", &PathOptimizationResult::message);
  auto optimizer =
      py::class_<PathOptimizer>(module, "PathOptimizer")
          .def(py::init<Eigen::VectorXd, Eigen::VectorXd,
                        PathOptimizer::StateValidator>(),
               py::arg("lower_limits"), py::arg("upper_limits"),
               py::arg("state_validator") = PathOptimizer::StateValidator{})
          .def("set_state_validator", &PathOptimizer::SetStateValidator,
               py::arg("validator"))
          .def("set_state_cost", &PathOptimizer::SetStateCost,
               py::arg("state_cost"))
          .def("set_state_cost_gradient", &PathOptimizer::SetStateCostGradient,
               py::arg("state_cost_gradient"))
          .def("set_joint_weights", &PathOptimizer::SetJointWeights,
               py::arg("weights"))
          .def("set_continuous_joints", &PathOptimizer::SetContinuousJoints,
               py::arg("indices"))
          .def("optimize", &PathOptimizer::Optimize, py::arg("path"),
               py::arg("options") = PathOptimizationOptions{});
#ifdef HOLISTICMOTION_HAS_COLLISION
  optimizer.def_static(
      "from_sphere_collision_model",
      [](Eigen::VectorXd lower, Eigen::VectorXd upper,
         robotics::collision::SphereCollisionModel &collision_model,
         double security_margin, double clearance) {
        if (!std::isfinite(security_margin) || security_margin < 0.0 ||
            !std::isfinite(clearance) || clearance < 0.0 ||
            (clearance > 0.0 && clearance < security_margin))
          throw std::invalid_argument(
              "clearance must be finite and at least security_margin");
        if (lower.size() != collision_model.GetConfigurationSize())
          throw std::invalid_argument(
              "full-space limits must match sphere collision model nq");
        if (clearance > 0.0 && collision_model.GetCollisionPairCount() == 0)
          throw std::invalid_argument(
              "positive clearance requires active sphere collision pairs");
        auto cache =
            clearance > 0.0
                ? std::make_shared<SphereDistanceCache>(collision_model)
                : std::shared_ptr<SphereDistanceCache>{};
        PathOptimizer result(
            std::move(lower), std::move(upper),
            [&collision_model, cache, security_margin,
             clearance](const Eigen::VectorXd &state) {
              if (collision_model.GetCollisionPairCount() == 0)
                return true;
              if (clearance > 0.0)
                return cache->ValidationDistance(state) > security_margin;
              return !collision_model.InCollision(state, security_margin, true);
            });
        if (clearance > 0.0) {
          result.SetStateCost([&collision_model, cache,
                               clearance](const Eigen::VectorXd &state) {
            if (collision_model.GetCollisionPairCount() == 0)
              return 0.0;
            const double deficit =
                std::max(0.0, clearance - cache->CostDistance(state).distance);
            return deficit * deficit;
          });
          if (collision_model.GetConfigurationSize() ==
              collision_model.GetVelocitySize()) {
            result.SetStateCostGradient(
                [&collision_model,
                 clearance](const Eigen::VectorXd &state) -> Eigen::VectorXd {
                  if (collision_model.GetCollisionPairCount() == 0)
                    return Eigen::VectorXd::Zero(state.size());
                  const auto query =
                      collision_model.MinimumDistanceWithGradient(state);
                  const double deficit =
                      std::max(0.0, clearance - query.distance_result.distance);
                  if (deficit == 0.0)
                    return Eigen::VectorXd::Zero(state.size());
                  return (-2.0 * deficit * query.gradient).eval();
                });
          }
        }
        return result;
      },
      py::arg("lower_limits"), py::arg("upper_limits"),
      py::arg("collision_model"), py::arg("security_margin") = 0.0,
      py::arg("clearance") = 0.0, py::keep_alive<0, 3>());
  optimizer.def_static(
      "from_collision_joints",
      [](robotics::collision::CollisionModel &collision_model,
         std::vector<std::string> joint_names,
         Eigen::VectorXd context_configuration, double security_margin,
         double clearance) {
        if (!std::isfinite(security_margin) || security_margin < 0.0 ||
            !std::isfinite(clearance) || clearance < 0.0 ||
            (clearance > 0.0 && clearance < security_margin))
          throw std::invalid_argument(
              "clearance must be finite and at least security_margin");
        if (context_configuration.size() !=
                collision_model.GetConfigurationSize() ||
            !context_configuration.allFinite())
          throw std::invalid_argument("context_configuration must be finite "
                                      "and match collision model nq");
        Eigen::VectorXd lower =
            collision_model.GetJointLowerLimits(joint_names);
        Eigen::VectorXd upper =
            collision_model.GetJointUpperLimits(joint_names);
        PathOptimizer result(
            std::move(lower), std::move(upper),
            [&collision_model, joint_names, context = context_configuration,
             security_margin](const Eigen::VectorXd &active) {
              const Eigen::VectorXd full =
                  collision_model.ConfigurationWithJointPositions(
                      context, joint_names, active);
              return security_margin > 0.0
                         ? !collision_model.IsWithinDistance(
                               full, security_margin, true)
                         : !collision_model.InCollision(full, true);
            });
        if (clearance > 0.0) {
          result.SetStateCost([&collision_model, joint_names,
                               context = context_configuration,
                               clearance](const Eigen::VectorXd &active) {
            const Eigen::VectorXd full =
                collision_model.ConfigurationWithJointPositions(
                    context, joint_names, active);
            const double deficit = std::max(
                0.0,
                clearance - collision_model.MinimumDistance(full).distance);
            return deficit * deficit;
          });
        }
        return result;
      },
      py::arg("collision_model"), py::arg("joint_names"),
      py::arg("context_configuration"), py::arg("security_margin") = 0.0,
      py::arg("clearance") = 0.0, py::keep_alive<0, 1>());
#endif

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
