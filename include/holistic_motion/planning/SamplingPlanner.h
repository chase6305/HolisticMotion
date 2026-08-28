#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics::planning {

enum class SamplingAlgorithm { RRT_CONNECT, RRT_STAR, INFORMED_RRT_STAR };

enum class PlanningStatus {
  EXACT_SOLUTION,
  TIMEOUT,
  INVALID_START,
  INVALID_GOAL,
  INVALID_PROBLEM,
  NO_SOLUTION
};

struct PlanningOptions {
  SamplingAlgorithm algorithm{SamplingAlgorithm::RRT_CONNECT};
  double timeout_seconds{2.0};
  std::size_t max_iterations{100000};
  double extension_range{0.2};
  double goal_bias{0.05};
  double edge_resolution{0.02};
  bool simplify_path{true};
  std::size_t shortcut_attempts{100};
  bool interpolate_path{false};
  std::size_t interpolation_points{100};
  std::uint64_t random_seed{0};
};

struct PlanningStatistics {
  std::size_t iterations{0};
  std::size_t sampled_states{0};
  std::size_t valid_states{0};
  std::size_t collision_checks{0};
  std::size_t tree_nodes{0};
  double planning_time_ms{0.0};
  double initial_path_length{0.0};
  double final_path_length{0.0};
};

struct PlanningResult {
  PlanningStatus status{PlanningStatus::NO_SOLUTION};
  std::vector<Eigen::VectorXd> path;
  PlanningStatistics statistics;
  std::string message;

  bool Success() const { return status == PlanningStatus::EXACT_SOLUTION; }
};

// Lightweight joint-space sampling planner implemented by HolisticMotion.
// The validator receives complete configurations and is deliberately
// independent of any particular collision library.
class SamplingPlanner {
public:
  using StateValidator = std::function<bool(const Eigen::VectorXd &)>;

  SamplingPlanner(Eigen::VectorXd lower_limits, Eigen::VectorXd upper_limits,
                  StateValidator validator = {});

  void SetStateValidator(StateValidator validator);
  void SetJointWeights(const Eigen::VectorXd &weights);
  void SetContinuousJoints(const std::vector<std::size_t> &indices);

  PlanningResult Plan(const Eigen::VectorXd &start, const Eigen::VectorXd &goal,
                      const PlanningOptions &options = {}) const;

private:
  Eigen::VectorXd lower_limits_;
  Eigen::VectorXd upper_limits_;
  Eigen::VectorXd weights_;
  std::vector<bool> continuous_;
  StateValidator validator_;
};

} // namespace holistic_motion::robotics::planning
