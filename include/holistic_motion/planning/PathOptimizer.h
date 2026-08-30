#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics::planning {

enum class PathOptimizationStatus {
  OPTIMIZED,
  UNCHANGED,
  TIMEOUT,
  INVALID_PATH
};

struct PathOptimizationOptions {
  std::size_t max_iterations{100};
  double timeout_seconds{1.0};
  double step_size{0.35};
  std::size_t line_search_steps{4};
  double line_search_decay{0.5};
  double edge_resolution{0.02};
  double length_weight{1.0};
  double smoothness_weight{0.25};
  double state_cost_weight{0.0};
  double finite_difference_step{1e-3};
  double state_cost_step_size{0.1};
  double minimum_improvement{1e-9};
};

struct PathOptimizationStatistics {
  std::size_t iterations{0};
  std::size_t attempted_updates{0};
  std::size_t line_search_evaluations{0};
  std::size_t accepted_updates{0};
  std::size_t collision_checks{0};
  std::size_t state_cost_evaluations{0};
  std::size_t state_cost_gradient_evaluations{0};
  double optimization_time_ms{0.0};
  double initial_objective{0.0};
  double final_objective{0.0};
  double initial_path_length{0.0};
  double final_path_length{0.0};
};

struct PathOptimizationResult {
  PathOptimizationStatus status{PathOptimizationStatus::INVALID_PATH};
  std::vector<Eigen::VectorXd> path;
  PathOptimizationStatistics statistics;
  std::string message;
  bool feasible{false};

  bool Success() const { return feasible; }
};

/// Feasibility-preserving joint-space path smoother.
///
/// Endpoints remain fixed. Every accepted waypoint update decreases the
/// weighted length/smoothness objective and validates both adjacent edges.
class PathOptimizer {
public:
  using StateValidator = std::function<bool(const Eigen::VectorXd &)>;
  using StateCost = std::function<double(const Eigen::VectorXd &)>;
  using StateCostGradient =
      std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;

  PathOptimizer(Eigen::VectorXd lower_limits, Eigen::VectorXd upper_limits,
                StateValidator validator = {});

  void SetStateValidator(StateValidator validator);
  void SetStateCost(StateCost state_cost);
  void SetStateCostGradient(StateCostGradient state_cost_gradient);
  void SetJointWeights(const Eigen::VectorXd &weights);
  void SetContinuousJoints(const std::vector<std::size_t> &indices);

  PathOptimizationResult
  Optimize(const std::vector<Eigen::VectorXd> &path,
           const PathOptimizationOptions &options = {}) const;

private:
  Eigen::VectorXd lower_limits_;
  Eigen::VectorXd upper_limits_;
  Eigen::VectorXd weights_;
  std::vector<bool> continuous_;
  StateValidator validator_;
  StateCost state_cost_;
  StateCostGradient state_cost_gradient_;
};

} // namespace holistic_motion::robotics::planning
