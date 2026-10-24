#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "holistic_motion/planning/NullSpacePlanner.h"
#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"

using holistic_motion::robotics::planning::NullSpacePlanner;
using holistic_motion::robotics::planning::PathOptimizer;
using holistic_motion::robotics::planning::PathOptimizationOptions;
using holistic_motion::robotics::planning::PlanningOptions;
using holistic_motion::robotics::planning::SamplingAlgorithm;
using holistic_motion::robotics::planning::SamplingPlanner;

namespace {
bool OutsideObstacle(const Eigen::VectorXd &q) {
  // Vertical wall with passages above and below it.
  return !(q[0] > -0.2 && q[0] < 0.2 && q[1] > -0.75 && q[1] < 0.75);
}
} // namespace

int main() {
    for (double weight : {1e-18, 1e-30}) {
        SamplingPlanner blocked(
            Eigen::VectorXd::Constant(1, -1.0), Eigen::VectorXd::Constant(1, 1.0),
            [](const Eigen::VectorXd &q) { return q[0] < -0.25 || q[0] > -0.15; });
        blocked.SetJointWeights(Eigen::VectorXd::Constant(1, weight));
        PlanningOptions scaled;
        scaled.extension_range = 0.3 * std::sqrt(weight);
        scaled.edge_resolution = 0.01;
        scaled.goal_bias = 1.0;
        scaled.max_iterations = 10;
        scaled.simplify_path = false;
        const auto result = blocked.Plan(Eigen::VectorXd::Constant(1, -0.9),
                                         Eigen::VectorXd::Constant(1, 0.9), scaled);
        if (result.Success() || !result.path.empty()) {
            std::cerr << "small metric weights bypassed an impassable wall\n";
            return 8;
        }
    }

  auto incompatible =
      std::make_shared<holistic_motion::robotics::SRSKinematics>(
          std::vector<holistic_motion::robotics::JointNode>(1));
  NullSpacePlanner null_space(incompatible);
  std::vector<Eigen::VectorXd> failed_path{Eigen::VectorXd::Zero(7)};
  if (null_space.Plan(Eigen::VectorXd::Zero(7), Eigen::VectorXd::Ones(7), 1,
                      0.01, failed_path) ||
      !failed_path.empty()) {
    return 6;
  }

  PathOptimizer path_optimizer(Eigen::Vector2d(-1.0, -1.0),
                               Eigen::Vector2d(1.0, 1.0));
  const std::vector<Eigen::VectorXd> rough_path{
      Eigen::Vector2d(-0.8, 0.0), Eigen::Vector2d(-0.4, 0.5),
      Eigen::Vector2d(0.0, -0.5), Eigen::Vector2d(0.4, 0.5),
      Eigen::Vector2d(0.8, 0.0)};
  const auto optimized = path_optimizer.Optimize(rough_path);
  if (!optimized.Success() ||
      optimized.statistics.final_objective >=
          optimized.statistics.initial_objective ||
      (optimized.path.front() - rough_path.front()).norm() > 1e-12 ||
      (optimized.path.back() - rough_path.back()).norm() > 1e-12) {
    return 5;
  }

  {
    PathOptimizer boundary_optimizer(
        Eigen::VectorXd::Zero(1), Eigen::VectorXd::Constant(1, 2.0),
        [](const Eigen::VectorXd &q) { return q[0] > 0.3; });
    holistic_motion::robotics::planning::PathOptimizationOptions boundary_options;
    boundary_options.edge_resolution = 2.0;
    const auto boundary = boundary_optimizer.Optimize(
        {Eigen::VectorXd::Constant(1, 1.0), Eigen::VectorXd::Constant(1, 0.3)},
        boundary_options);
    if (boundary.Success() || !boundary.path.empty() ||
        boundary.status != holistic_motion::robotics::planning::PathOptimizationStatus::INVALID_PATH) {
      std::cerr << "invalid exact endpoint escaped path validation\n";
      return 7;
    }
  }

  {
    PathOptimizer stationary_optimizer(Eigen::VectorXd::Constant(1, -1.0),
                                      Eigen::VectorXd::Constant(1, 1.0));
    std::size_t gradient_calls = 0;
    stationary_optimizer.SetStateCost(
        [](const Eigen::VectorXd &) { return 1.0; });
    stationary_optimizer.SetStateCostGradient(
        [&gradient_calls](const Eigen::VectorXd &q) {
          ++gradient_calls;
          return q;
        });
    PathOptimizationOptions stationary_options;
    stationary_options.length_weight = 0.0;
    stationary_options.smoothness_weight = 0.0;
    stationary_options.state_cost_weight = 1.0;
    stationary_options.state_cost_step_size = 0.0;
    const auto stationary = stationary_optimizer.Optimize(
        {Eigen::VectorXd::Constant(1, -0.5), Eigen::VectorXd::Constant(1, 0.2),
         Eigen::VectorXd::Constant(1, 0.5)},
        stationary_options);
    if (!stationary.Success() || gradient_calls != 0 ||
        stationary.statistics.state_cost_evaluations != 1 ||
        stationary.statistics.state_cost_gradient_evaluations != 0 ||
        stationary.path[1][0] != 0.2) {
      std::cerr << "zero-gain state cost still requested its gradient\n";
      return 10;
    }
  }

  // Unchanged costs elsewhere on the path must not hide local progress or
  // allow a locally worse trial from an inaccurate gradient callback.
  for (double sign : {-1.0, 1.0}) {
      PathOptimizer local_optimizer(Eigen::VectorXd::Constant(1, -1.0),
                                    Eigen::VectorXd::Constant(1, 1.0));
      local_optimizer.SetStateCost(
          [](const Eigen::VectorXd &q) { return q[0] < -0.5 ? 1e100 : q[0] * q[0]; });
      local_optimizer.SetStateCostGradient([sign](const Eigen::VectorXd &q) {
          return Eigen::VectorXd::Constant(1, q[0] < -0.5 ? 0.0 : sign * 2.0 * q[0]);
      });
      holistic_motion::robotics::planning::PathOptimizationOptions local_options;
      local_options.length_weight = 0.0;
      local_options.smoothness_weight = 0.0;
      local_options.state_cost_weight = 1.0;
      local_options.state_cost_step_size = 1.0;
      local_options.step_size = 0.1;
      local_options.minimum_improvement = sign > 0.0 ? 1e-8 : 0.0;
      local_options.max_iterations = 1;
      local_options.line_search_steps = 1;
      const auto local = local_optimizer.Optimize(
          {Eigen::VectorXd::Zero(1), Eigen::VectorXd::Constant(1, -0.8),
           Eigen::VectorXd::Constant(1, 0.7), Eigen::VectorXd::Zero(1)},
          local_options);
      if (!local.Success() ||
          std::abs(local.path[2][0] - (sign > 0.0 ? 0.6 : 0.7)) > 1e-14) {
          std::cerr << "unrelated cost masked a local objective change\n";
          return 9;
      }
  }

  std::size_t direct_checks = 0;
  SamplingPlanner direct_planner(Eigen::Vector2d(-1.0, -1.0),
                                 Eigen::Vector2d(1.0, 1.0),
                                 [&direct_checks](const Eigen::VectorXd &) {
                                   ++direct_checks;
                                   return true;
                                 });
  const auto direct = direct_planner.Plan(Eigen::Vector2d(-0.5, -0.25),
                                          Eigen::Vector2d(0.5, 0.25));
  if (!direct.Success() || direct.path.size() != 2 ||
      direct.statistics.iterations != 0 ||
      direct.statistics.sampled_states != 0 ||
      direct_checks != direct.statistics.collision_checks) {
    return 4;
  }

  SamplingPlanner planner(Eigen::Vector2d(-1.0, -1.0),
                          Eigen::Vector2d(1.0, 1.0), OutsideObstacle);
  PlanningOptions options;
  options.timeout_seconds = 1.0;
  options.extension_range = 0.15;
  options.edge_resolution = 0.02;
  options.random_seed = 7;
  options.shortcut_attempts = 80;
  const Eigen::Vector2d start(-0.8, 0.0);
  const Eigen::Vector2d goal(0.8, 0.0);

  for (SamplingAlgorithm algorithm :
       {SamplingAlgorithm::RRT_CONNECT, SamplingAlgorithm::RRT_STAR,
        SamplingAlgorithm::INFORMED_RRT_STAR}) {
    options.algorithm = algorithm;
    if (algorithm != SamplingAlgorithm::RRT_CONNECT) {
      options.timeout_seconds = 0.08;
      options.max_iterations = 2000;
    }
    const auto result = planner.Plan(start, goal, options);
    if (!result.Success() || result.path.empty() ||
        (result.path.front() - start).norm() > 1e-12 ||
        (result.path.back() - goal).norm() > 1e-12) {
      std::cerr << "sampling planner failed: " << result.message << '\n';
      return 1;
    }
    for (const auto &state : result.path) {
      if (!OutsideObstacle(state))
        return 2;
    }
    if (result.statistics.final_path_length >
        result.statistics.initial_path_length + 1e-9) {
      return 3;
    }
  }
    // A tiny informed subset must not turn one iteration into an unbounded
    // rejection loop. Exercise the weighted mixed-topology regression without
    // using an elapsed-time threshold.
    Eigen::VectorXd narrow_lower = Eigen::VectorXd::Constant(6, -1.0);
    Eigen::VectorXd narrow_upper = Eigen::VectorXd::Constant(6, 1.0);
    for (Eigen::Index axis : {0, 3}) {
        narrow_lower[axis] = -std::acos(-1.0);
        narrow_upper[axis] = std::acos(-1.0);
    }
    SamplingPlanner narrow(
        narrow_lower, narrow_upper, [](const Eigen::VectorXd &q) {
            return !(std::abs(q[0]) < 0.2 && std::abs(q[1]) < 0.22);
        });
    narrow.SetContinuousJoints({0, 3});
    narrow.SetJointWeights(
        (Eigen::VectorXd(6) << 0.25, 0.015625, 0.015625, 0.015625, 4.0, 8.0)
            .finished());
    PlanningOptions narrow_options;
    narrow_options.algorithm = SamplingAlgorithm::INFORMED_RRT_STAR;
    narrow_options.max_iterations = 1317;
    narrow_options.random_seed = 18081385724285266904ULL;
    narrow_options.goal_bias = 0.17;
    narrow_options.extension_range = 0.5;
    narrow_options.edge_resolution = 0.08;
    narrow_options.shortcut_attempts = 0;
    narrow_options.timeout_seconds = 5.0;
    Eigen::VectorXd narrow_start = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd narrow_goal = narrow_start;
    narrow_start[0] = -0.85;
    narrow_goal[0] = 0.85;
    const auto narrow_result =
        narrow.Plan(narrow_start, narrow_goal, narrow_options);
    if (!narrow_result.Success() ||
        narrow_result.statistics.iterations != narrow_options.max_iterations ||
        narrow_result.statistics.sampled_states >
            128 * narrow_options.max_iterations) {
        std::cerr << "informed rejection sampling exhausted its budget\n";
        return 9;
    }
    return 0;
}
