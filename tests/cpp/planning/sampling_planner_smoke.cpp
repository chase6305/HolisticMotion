#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "holistic_motion/planning/NullSpacePlanner.h"
#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"

using holistic_motion::robotics::planning::NullSpacePlanner;
using holistic_motion::robotics::planning::PathOptimizer;
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
  return 0;
}
