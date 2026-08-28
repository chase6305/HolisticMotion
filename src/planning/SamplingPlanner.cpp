#include "holistic_motion/planning/SamplingPlanner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace holistic_motion::robotics::planning {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct Node {
  Eigen::VectorXd state;
  std::size_t parent{0};
  double cost{0.0};
  std::vector<std::size_t> children;
};

struct Tree {
  std::vector<Node> nodes;
  bool rooted_at_start{true};
};

enum class ExtendStatus { TRAPPED, ADVANCED, REACHED };

double WrappedDifference(double difference) {
  return std::remainder(difference, 2.0 * kPi);
}

class PlanningContext {
public:
  PlanningContext(const Eigen::VectorXd &lower, const Eigen::VectorXd &upper,
                  const Eigen::VectorXd &weights,
                  const std::vector<bool> &continuous,
                  const SamplingPlanner::StateValidator &validator,
                  const PlanningOptions &options,
                  PlanningStatistics &statistics,
                  std::chrono::steady_clock::time_point deadline)
      : lower_(lower), upper_(upper), weights_(weights),
        continuous_(continuous), validator_(validator), options_(options),
        statistics_(statistics), deadline_(deadline),
        generator_(options.random_seed) {}

  bool TimedOut() const {
    return std::chrono::steady_clock::now() >= deadline_;
  }

  Eigen::VectorXd Difference(const Eigen::VectorXd &from,
                             const Eigen::VectorXd &to) const {
    Eigen::VectorXd difference = to - from;
    for (Eigen::Index i = 0; i < difference.size(); ++i) {
      if (continuous_[static_cast<std::size_t>(i)]) {
        difference[i] = WrappedDifference(difference[i]);
      }
    }
    return difference;
  }

  Eigen::VectorXd Normalize(Eigen::VectorXd state) const {
    for (Eigen::Index i = 0; i < state.size(); ++i) {
      if (continuous_[static_cast<std::size_t>(i)]) {
        state[i] =
            lower_[i] +
            std::fmod(std::fmod(state[i] - lower_[i], 2.0 * kPi) + 2.0 * kPi,
                      2.0 * kPi);
      } else {
        state[i] = std::clamp(state[i], lower_[i], upper_[i]);
      }
    }
    return state;
  }

  double Distance(const Eigen::VectorXd &first,
                  const Eigen::VectorXd &second) const {
    const Eigen::VectorXd delta = Difference(first, second);
    return std::sqrt((weights_.array() * delta.array().square()).sum());
  }

  Eigen::VectorXd Interpolate(const Eigen::VectorXd &from,
                              const Eigen::VectorXd &to, double ratio) const {
    return Normalize(from + ratio * Difference(from, to));
  }

  Eigen::VectorXd Steer(const Eigen::VectorXd &from,
                        const Eigen::VectorXd &to) const {
    const double distance = Distance(from, to);
    if (distance <= options_.extension_range)
      return Normalize(to);
    return Interpolate(from, to, options_.extension_range / distance);
  }

  Eigen::VectorXd SampleUniform() {
    ++statistics_.sampled_states;
    Eigen::VectorXd sample(lower_.size());
    for (Eigen::Index i = 0; i < sample.size(); ++i) {
      std::uniform_real_distribution<double> distribution(lower_[i], upper_[i]);
      sample[i] = distribution(generator_);
    }
    return sample;
  }

  bool IsStateValid(const Eigen::VectorXd &state) {
    ++statistics_.collision_checks;
    const bool valid = !validator_ || validator_(state);
    if (valid)
      ++statistics_.valid_states;
    return valid;
  }

  bool IsMotionValid(const Eigen::VectorXd &from, const Eigen::VectorXd &to) {
    if (TimedOut())
      return false;
    const Eigen::VectorXd delta = Difference(from, to).cwiseAbs();
    const std::size_t segments = std::max<std::size_t>(
        1, static_cast<std::size_t>(
               std::ceil(delta.maxCoeff() / options_.edge_resolution)));
    // Check the far endpoint first, then interior states. This quickly
    // rejects extensions whose newly sampled endpoint is invalid.
    if (!IsStateValid(to))
      return false;
    for (std::size_t i = 1; i < segments; ++i) {
      if (TimedOut())
        return false;
      if (!IsStateValid(
              Interpolate(from, to, static_cast<double>(i) / segments))) {
        return false;
      }
    }
    return true;
  }

  std::size_t Nearest(const Tree &tree, const Eigen::VectorXd &state) const {
    std::size_t best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
      const double distance = Distance(tree.nodes[i].state, state);
      if (distance < best_distance) {
        best_distance = distance;
        best = i;
      }
    }
    return best;
  }

  std::vector<std::size_t> Near(const Tree &tree, const Eigen::VectorXd &state,
                                double radius) const {
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
      if (Distance(tree.nodes[i].state, state) <= radius)
        result.push_back(i);
    }
    return result;
  }

  ExtendStatus Extend(Tree &tree, const Eigen::VectorXd &target,
                      std::size_t &new_index) {
    const std::size_t nearest = Nearest(tree, target);
    Eigen::VectorXd candidate = Steer(tree.nodes[nearest].state, target);
    if (Distance(tree.nodes[nearest].state, candidate) < 1e-12 ||
        !IsMotionValid(tree.nodes[nearest].state, candidate)) {
      return ExtendStatus::TRAPPED;
    }
    new_index = tree.nodes.size();
    tree.nodes.push_back({candidate,
                          nearest,
                          tree.nodes[nearest].cost +
                              Distance(tree.nodes[nearest].state, candidate),
                          {}});
    tree.nodes[nearest].children.push_back(new_index);
    return Distance(candidate, target) < 1e-9 ? ExtendStatus::REACHED
                                              : ExtendStatus::ADVANCED;
  }

  std::vector<Eigen::VectorXd> Trace(const Tree &tree,
                                     std::size_t index) const {
    std::vector<Eigen::VectorXd> result;
    while (true) {
      result.push_back(tree.nodes[index].state);
      if (index == tree.nodes[index].parent)
        break;
      index = tree.nodes[index].parent;
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  std::mt19937_64 &Generator() { return generator_; }

private:
  const Eigen::VectorXd &lower_;
  const Eigen::VectorXd &upper_;
  const Eigen::VectorXd &weights_;
  const std::vector<bool> &continuous_;
  const SamplingPlanner::StateValidator &validator_;
  const PlanningOptions &options_;
  PlanningStatistics &statistics_;
  std::chrono::steady_clock::time_point deadline_;
  std::mt19937_64 generator_;
};

double PathLength(const std::vector<Eigen::VectorXd> &path,
                  const PlanningContext &context) {
  double length = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    length += context.Distance(path[i - 1], path[i]);
  }
  return length;
}

void Shortcut(std::vector<Eigen::VectorXd> &path, PlanningContext &context,
              std::size_t attempts) {
  if (path.size() < 3)
    return;
  for (std::size_t attempt = 0; attempt < attempts && path.size() >= 3;
       ++attempt) {
    if (context.TimedOut())
      return;
    std::vector<double> cumulative(path.size(), 0.0);
    for (std::size_t i = 1; i < path.size(); ++i) {
      cumulative[i] =
          cumulative[i - 1] + context.Distance(path[i - 1], path[i]);
    }
    if (cumulative.back() <= 1e-12)
      return;

    std::uniform_real_distribution<double> distribution(0.0, cumulative.back());
    double first_distance = distribution(context.Generator());
    double second_distance = distribution(context.Generator());
    if (first_distance > second_distance)
      std::swap(first_distance, second_distance);
    if (second_distance - first_distance <= 1e-9)
      continue;

    const auto sample_at = [&](double distance, std::size_t &segment) {
      const auto upper =
          std::upper_bound(cumulative.begin(), cumulative.end(), distance);
      segment = std::min<std::size_t>(
          std::max<std::size_t>(1, upper - cumulative.begin()) - 1,
          path.size() - 2);
      const double span = cumulative[segment + 1] - cumulative[segment];
      const double ratio =
          span <= 1e-12 ? 0.0 : (distance - cumulative[segment]) / span;
      return context.Interpolate(path[segment], path[segment + 1], ratio);
    };

    std::size_t first_segment = 0;
    std::size_t second_segment = 0;
    const Eigen::VectorXd first = sample_at(first_distance, first_segment);
    const Eigen::VectorXd second = sample_at(second_distance, second_segment);
    const double direct_length = context.Distance(first, second);
    if (direct_length + 1e-9 >= second_distance - first_distance ||
        !context.IsMotionValid(first, second))
      continue;

    std::vector<Eigen::VectorXd> shortened;
    shortened.reserve(path.size() + 2);
    const auto append = [&](const Eigen::VectorXd &state) {
      if (shortened.empty() ||
          context.Distance(shortened.back(), state) > 1e-12) {
        shortened.push_back(state);
      }
    };
    for (std::size_t i = 0; i <= first_segment; ++i)
      append(path[i]);
    append(first);
    append(second);
    for (std::size_t i = second_segment + 1; i < path.size(); ++i)
      append(path[i]);
    path = std::move(shortened);
  }
}

void InterpolatePath(std::vector<Eigen::VectorXd> &path,
                     const PlanningContext &context, std::size_t points) {
  if (path.size() < 2 || points <= path.size())
    return;
  std::vector<double> cumulative(path.size(), 0.0);
  for (std::size_t i = 1; i < path.size(); ++i) {
    cumulative[i] = cumulative[i - 1] + context.Distance(path[i - 1], path[i]);
  }
  if (cumulative.back() <= 1e-12)
    return;
  std::vector<Eigen::VectorXd> output;
  output.reserve(points);
  std::size_t segment = 1;
  for (std::size_t i = 0; i < points; ++i) {
    const double distance = cumulative.back() * static_cast<double>(i) /
                            static_cast<double>(points - 1);
    while (segment + 1 < cumulative.size() && cumulative[segment] < distance) {
      ++segment;
    }
    const double span = cumulative[segment] - cumulative[segment - 1];
    const double ratio =
        span <= 1e-12 ? 0.0 : (distance - cumulative[segment - 1]) / span;
    output.push_back(
        context.Interpolate(path[segment - 1], path[segment], ratio));
  }
  path = std::move(output);
}

std::vector<Eigen::VectorXd>
ConnectPaths(const Tree &first, std::size_t first_index, const Tree &second,
             std::size_t second_index, const PlanningContext &context) {
  auto first_path = context.Trace(first, first_index);
  auto second_path = context.Trace(second, second_index);
  if (!first.rooted_at_start) {
    std::swap(first_path, second_path);
  }
  // The goal-rooted trace runs goal -> connection after reversal.
  std::reverse(second_path.begin(), second_path.end());
  if (!first_path.empty() && !second_path.empty() &&
      context.Distance(first_path.back(), second_path.front()) < 1e-9) {
    second_path.erase(second_path.begin());
  }
  first_path.insert(first_path.end(), second_path.begin(), second_path.end());
  return first_path;
}

std::vector<Eigen::VectorXd>
PlanRRTConnect(const Eigen::VectorXd &start, const Eigen::VectorXd &goal,
               PlanningContext &context, const PlanningOptions &options,
               PlanningStatistics &statistics,
               const std::chrono::steady_clock::time_point &deadline) {
  Tree first{{Node{start, 0, 0.0, {}}}, true};
  Tree second{{Node{goal, 0, 0.0, {}}}, false};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (; statistics.iterations < options.max_iterations &&
         std::chrono::steady_clock::now() < deadline;
       ++statistics.iterations) {
    Eigen::VectorXd sample = unit(context.Generator()) < options.goal_bias
                                 ? second.nodes.front().state
                                 : context.SampleUniform();
    std::size_t first_index = 0;
    if (context.Extend(first, sample, first_index) == ExtendStatus::TRAPPED) {
      std::swap(first, second);
      continue;
    }
    std::size_t second_index = 0;
    ExtendStatus status;
    do {
      status =
          context.Extend(second, first.nodes[first_index].state, second_index);
    } while (status == ExtendStatus::ADVANCED &&
             std::chrono::steady_clock::now() < deadline);
    if (status == ExtendStatus::REACHED) {
      statistics.tree_nodes = first.nodes.size() + second.nodes.size();
      return ConnectPaths(first, first_index, second, second_index, context);
    }
    std::swap(first, second);
  }
  statistics.tree_nodes = first.nodes.size() + second.nodes.size();
  return {};
}

std::vector<Eigen::VectorXd>
PlanRRTStar(const Eigen::VectorXd &start, const Eigen::VectorXd &goal,
            PlanningContext &context, const PlanningOptions &options,
            PlanningStatistics &statistics,
            const std::chrono::steady_clock::time_point &deadline,
            bool informed) {
  Tree tree{{Node{start, 0, 0.0, {}}}, true};
  std::size_t best_goal = std::numeric_limits<std::size_t>::max();
  double best_cost = std::numeric_limits<double>::infinity();
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  auto update_descendant_costs = [&](std::size_t root, double delta) {
    std::vector<std::size_t> pending = tree.nodes[root].children;
    while (!pending.empty()) {
      const std::size_t index = pending.back();
      pending.pop_back();
      tree.nodes[index].cost += delta;
      pending.insert(pending.end(), tree.nodes[index].children.begin(),
                     tree.nodes[index].children.end());
    }
  };
  for (; statistics.iterations < options.max_iterations &&
         std::chrono::steady_clock::now() < deadline;
       ++statistics.iterations) {
    Eigen::VectorXd sample;
    if (unit(context.Generator()) < options.goal_bias) {
      sample = goal;
    } else {
      // Rejection sampling implements an informed subset without a
      // heavyweight ellipsoid dependency and remains valid for weighted
      // and wrapped joint metrics.
      do {
        sample = context.SampleUniform();
      } while (informed && std::isfinite(best_cost) &&
               context.Distance(start, sample) +
                       context.Distance(sample, goal) >
                   best_cost &&
               std::chrono::steady_clock::now() < deadline);
    }
    const std::size_t nearest = context.Nearest(tree, sample);
    Eigen::VectorXd candidate =
        context.Steer(tree.nodes[nearest].state, sample);
    if (!context.IsMotionValid(tree.nodes[nearest].state, candidate))
      continue;
    const double dimension = static_cast<double>(start.size());
    const double radius = std::min(
        options.extension_range * 4.0,
        options.extension_range * 2.0 *
            std::pow(std::log(static_cast<double>(tree.nodes.size() + 1)) /
                         static_cast<double>(tree.nodes.size() + 1),
                     1.0 / dimension));
    auto near = context.Near(tree, candidate,
                             std::max(radius, options.extension_range));
    std::size_t parent = nearest;
    double cost = tree.nodes[nearest].cost +
                  context.Distance(tree.nodes[nearest].state, candidate);
    for (std::size_t index : near) {
      const double candidate_cost =
          tree.nodes[index].cost +
          context.Distance(tree.nodes[index].state, candidate);
      if (candidate_cost < cost &&
          context.IsMotionValid(tree.nodes[index].state, candidate)) {
        parent = index;
        cost = candidate_cost;
      }
    }
    const std::size_t inserted = tree.nodes.size();
    tree.nodes.push_back({candidate, parent, cost, {}});
    tree.nodes[parent].children.push_back(inserted);
    for (std::size_t index : near) {
      const double rewired =
          cost + context.Distance(candidate, tree.nodes[index].state);
      if (rewired < tree.nodes[index].cost &&
          context.IsMotionValid(candidate, tree.nodes[index].state)) {
        const double delta = rewired - tree.nodes[index].cost;
        auto &old_children = tree.nodes[tree.nodes[index].parent].children;
        old_children.erase(
            std::remove(old_children.begin(), old_children.end(), index),
            old_children.end());
        tree.nodes[index].parent = inserted;
        tree.nodes[index].cost = rewired;
        tree.nodes[inserted].children.push_back(index);
        update_descendant_costs(index, delta);
      }
    }
    const double remaining = context.Distance(candidate, goal);
    if (remaining <= options.extension_range && cost + remaining < best_cost &&
        context.IsMotionValid(candidate, goal)) {
      best_goal = tree.nodes.size();
      best_cost = cost + remaining;
      tree.nodes.push_back({goal, inserted, best_cost, {}});
      tree.nodes[inserted].children.push_back(best_goal);
    }
  }
  statistics.tree_nodes = tree.nodes.size();
  return best_goal == std::numeric_limits<std::size_t>::max()
             ? std::vector<Eigen::VectorXd>{}
             : context.Trace(tree, best_goal);
}

} // namespace

SamplingPlanner::SamplingPlanner(Eigen::VectorXd lower_limits,
                                 Eigen::VectorXd upper_limits,
                                 StateValidator validator)
    : lower_limits_(std::move(lower_limits)),
      upper_limits_(std::move(upper_limits)), validator_(std::move(validator)) {
  if (lower_limits_.size() == 0 ||
      lower_limits_.size() != upper_limits_.size() ||
      !lower_limits_.allFinite() || !upper_limits_.allFinite() ||
      (lower_limits_.array() >= upper_limits_.array()).any()) {
    throw std::invalid_argument(
        "joint limits must be finite, ordered, and non-empty");
  }
  weights_ =
      (upper_limits_ - lower_limits_).array().square().inverse().matrix();
  continuous_.assign(static_cast<std::size_t>(lower_limits_.size()), false);
}

void SamplingPlanner::SetStateValidator(StateValidator validator) {
  validator_ = std::move(validator);
}

void SamplingPlanner::SetJointWeights(const Eigen::VectorXd &weights) {
  if (weights.size() != lower_limits_.size() || !weights.allFinite() ||
      (weights.array() <= 0.0).any()) {
    throw std::invalid_argument(
        "joint weights must be finite, positive, and dimensionally compatible");
  }
  weights_ = weights;
}

void SamplingPlanner::SetContinuousJoints(
    const std::vector<std::size_t> &indices) {
  continuous_.assign(static_cast<std::size_t>(lower_limits_.size()), false);
  for (std::size_t index : indices) {
    if (index >= continuous_.size())
      throw std::out_of_range("continuous joint index");
    if (upper_limits_[static_cast<Eigen::Index>(index)] -
            lower_limits_[static_cast<Eigen::Index>(index)] <
        2.0 * kPi - 1e-9) {
      throw std::invalid_argument(
          "continuous joints require a range of at least 2*pi");
    }
    continuous_[index] = true;
  }
}

PlanningResult SamplingPlanner::Plan(const Eigen::VectorXd &start,
                                     const Eigen::VectorXd &goal,
                                     const PlanningOptions &options) const {
  PlanningResult result;
  const auto started = std::chrono::steady_clock::now();
  auto finish = [&](PlanningStatus status, std::string message) {
    result.status = status;
    result.message = std::move(message);
    result.statistics.planning_time_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
    return result;
  };
  const bool finite_options = std::isfinite(options.timeout_seconds) &&
                              std::isfinite(options.extension_range) &&
                              std::isfinite(options.goal_bias) &&
                              std::isfinite(options.edge_resolution);
  const bool supported_algorithm =
      options.algorithm == SamplingAlgorithm::RRT_CONNECT ||
      options.algorithm == SamplingAlgorithm::RRT_STAR ||
      options.algorithm == SamplingAlgorithm::INFORMED_RRT_STAR;
  if (start.size() != lower_limits_.size() ||
      goal.size() != lower_limits_.size() || !start.allFinite() ||
      !goal.allFinite() || !finite_options || !supported_algorithm ||
      options.timeout_seconds <= 0.0 || options.max_iterations == 0 ||
      options.extension_range <= 0.0 || options.edge_resolution <= 0.0 ||
      options.goal_bias < 0.0 || options.goal_bias > 1.0 ||
      (options.interpolate_path && options.interpolation_points < 2)) {
    return finish(PlanningStatus::INVALID_PROBLEM,
                  "invalid dimensions or planning options");
  }
  const auto deadline =
      started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(options.timeout_seconds));
  auto in_bounds = [&](const Eigen::VectorXd &state) {
    return (state.array() >= lower_limits_.array()).all() &&
           (state.array() <= upper_limits_.array()).all();
  };
  if (!in_bounds(start))
    return finish(PlanningStatus::INVALID_START,
                  "start is outside joint limits");
  if (!in_bounds(goal))
    return finish(PlanningStatus::INVALID_GOAL, "goal is outside joint limits");

  PlanningContext context(lower_limits_, upper_limits_, weights_, continuous_,
                          validator_, options, result.statistics, deadline);
  const bool start_valid = context.IsStateValid(start);
  if (context.TimedOut())
    return finish(PlanningStatus::TIMEOUT, "planning time limit reached");
  if (!start_valid) {
    return finish(PlanningStatus::INVALID_START, "start state is invalid");
  }
  const bool goal_valid = context.IsStateValid(goal);
  if (context.TimedOut())
    return finish(PlanningStatus::TIMEOUT, "planning time limit reached");
  if (!goal_valid) {
    return finish(PlanningStatus::INVALID_GOAL, "goal state is invalid");
  }
  if (context.Distance(start, goal) < 1e-12) {
    result.path = {start};
    result.statistics.tree_nodes = 1;
    return finish(PlanningStatus::EXACT_SOLUTION,
                  "start already satisfies the goal");
  }
  // A direct, fully validated edge is already the shortest path in the
  // planner's weighted joint metric. Avoid random sampling in this common
  // case, which also makes unobstructed plans deterministic for every
  // algorithm.
  if (context.IsMotionValid(start, goal)) {
    result.path = {start, goal};
    result.statistics.tree_nodes = 2;
    result.statistics.initial_path_length = context.Distance(start, goal);
    if (options.interpolate_path) {
      InterpolatePath(result.path, context, options.interpolation_points);
    }
    result.statistics.final_path_length = result.statistics.initial_path_length;
    return finish(PlanningStatus::EXACT_SOLUTION, "direct path found");
  }
  if (context.TimedOut())
    return finish(PlanningStatus::TIMEOUT, "planning time limit reached");
  switch (options.algorithm) {
  case SamplingAlgorithm::RRT_CONNECT:
    result.path = PlanRRTConnect(start, goal, context, options,
                                 result.statistics, deadline);
    break;
  case SamplingAlgorithm::RRT_STAR:
    result.path = PlanRRTStar(start, goal, context, options, result.statistics,
                              deadline, false);
    break;
  case SamplingAlgorithm::INFORMED_RRT_STAR:
    result.path = PlanRRTStar(start, goal, context, options, result.statistics,
                              deadline, true);
    break;
  }
  if (result.path.empty()) {
    const bool timed_out = std::chrono::steady_clock::now() >= deadline;
    return finish(timed_out ? PlanningStatus::TIMEOUT
                            : PlanningStatus::NO_SOLUTION,
                  timed_out ? "planning time limit reached" : "no path found");
  }
  result.statistics.initial_path_length = PathLength(result.path, context);
  if (options.simplify_path)
    Shortcut(result.path, context, options.shortcut_attempts);
  if (options.interpolate_path) {
    InterpolatePath(result.path, context, options.interpolation_points);
  }
  result.statistics.final_path_length = PathLength(result.path, context);
  return finish(PlanningStatus::EXACT_SOLUTION, "exact path found");
}

} // namespace holistic_motion::robotics::planning
