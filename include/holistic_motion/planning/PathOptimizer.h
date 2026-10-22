#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "holistic_motion/planning/JointSpace.h"

namespace holistic_motion::robotics::planning {
namespace detail {
struct PlanningCallbackAccess;
}

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
  // Actual projected offsets are used; circular steps are capped at pi/2.
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
  /// Space used for this request, independent of later optimizer setters.
  std::optional<JointSpace> joint_space;

  bool Success() const { return feasible; }
};

/// Feasibility-preserving joint-space path smoother.
///
/// Endpoints remain fixed. Every accepted waypoint update decreases the
/// weighted length/smoothness objective and validates both adjacent edges.
/// Input validation checks each original waypoint exactly, as well as the
/// configured interior edge samples, to preserve validity at roundoff
/// boundaries. Each call snapshots options, joint space and callback wrappers;
/// reentrant edits affect subsequent requests. Captured callback state retains
/// its usual ownership. Concurrent Optimize calls
/// additionally require thread-safe validator/cost/gradient callbacks and no
/// concurrent setter calls.
class PathOptimizer {
    friend struct detail::PlanningCallbackAccess;
public:
  using StateValidator = std::function<bool(const Eigen::VectorXd &)>;
  using StateCost = std::function<double(const Eigen::VectorXd &)>;
  using StateCostGradient =
      std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;

  PathOptimizer(Eigen::VectorXd lower_limits, Eigen::VectorXd upper_limits,
                StateValidator validator = {});
  explicit PathOptimizer(JointSpace joint_space, StateValidator validator = {});

  /// Independent topology snapshot; callbacks and their state are not copied.
  JointSpace GetJointSpace() const { return joint_space_; }

  void SetStateValidator(StateValidator validator);
  void SetStateCost(StateCost state_cost);
  void SetStateCostGradient(StateCostGradient state_cost_gradient);
  void SetJointWeights(const Eigen::VectorXd &weights);
  void SetContinuousJoints(const std::vector<std::size_t> &indices);

  PathOptimizationResult
  Optimize(const std::vector<Eigen::VectorXd> &path,
           const PathOptimizationOptions &options = {}) const;

private:
  JointSpace joint_space_;
  StateValidator validator_;
  StateCost state_cost_;
  StateCostGradient state_cost_gradient_;
};

} // namespace holistic_motion::robotics::planning
