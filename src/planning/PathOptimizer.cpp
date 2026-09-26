#include "holistic_motion/planning/PathOptimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "JointSpaceTopology.h"
#include "PathGeometryWorkspace.h"
#include "PlanningDeadline.h"

namespace holistic_motion::robotics::planning {
namespace {

class OptimizationContext {
  public:
    OptimizationContext(
        const Eigen::VectorXd &lower, const Eigen::VectorXd &upper,
        const Eigen::VectorXd &weights, const std::vector<bool> &continuous,
        const PathOptimizer::StateValidator &validator,
        const PathOptimizer::StateCost &state_cost,
        const PathOptimizer::StateCostGradient &state_cost_gradient,
        const PathOptimizationOptions &options,
        PathOptimizationStatistics &statistics,
        std::chrono::steady_clock::time_point deadline)
        : lower_(lower), upper_(upper), weights_(weights),
          inverse_weights_(weights.cwiseInverse()),
          has_continuous_(std::any_of(continuous.begin(), continuous.end(),
                                      [](bool value) { return value; })),
          continuous_(continuous), validator_(validator),
          state_cost_(state_cost), state_cost_gradient_(state_cost_gradient),
          options_(options), statistics_(statistics), deadline_(deadline) {}

    bool TimedOut() const {
        return std::chrono::steady_clock::now() >= deadline_;
    }

    Eigen::VectorXd Difference(const Eigen::VectorXd &from,
                               const Eigen::VectorXd &to) const {
        return detail::Difference(from, to, continuous_);
    }

    void Normalize(Eigen::VectorXd &state) const {
        for (Eigen::Index i = 0; i < state.size(); ++i)
            state[i] = NormalizeCoordinate(state[i], i);
    }

    Eigen::VectorXd NormalizeInterpolation(Eigen::VectorXd state) const {
        if (!has_continuous_)
            return state;
        return detail::Normalize(std::move(state), lower_, upper_, continuous_,
                                 false);
    }

    double SquaredNorm(const Eigen::VectorXd &value) const {
        return (weights_.array() * value.array().square()).sum();
    }

    double Distance(const Eigen::VectorXd &first,
                    const Eigen::VectorXd &second) const {
        return std::sqrt(SquaredNorm(Difference(first, second)));
    }

    double PathLength(const std::vector<Eigen::VectorXd> &path) const {
        double value = 0.0;
        for (std::size_t i = 1; i < path.size(); ++i)
            value += Distance(path[i - 1], path[i]);
        return value;
    }

    double GeometryObjective(const std::vector<Eigen::VectorXd> &path) const {
        if (options_.length_weight == 0.0 && options_.smoothness_weight == 0.0)
            return 0.0;
        Eigen::VectorXd previous = Difference(path[0], path[1]);
        double length =
            options_.length_weight > 0.0 ? std::sqrt(SquaredNorm(previous)) : 0.0;
        double smoothness = 0.0;
        for (std::size_t i = 1; i + 1 < path.size(); ++i) {
            Eigen::VectorXd next = Difference(path[i], path[i + 1]);
            if (options_.length_weight > 0.0)
                length += std::sqrt(SquaredNorm(next));
            if (options_.smoothness_weight > 0.0)
                smoothness += SquaredNorm(next - previous);
            previous = std::move(next);
        }
        return options_.length_weight * length +
               options_.smoothness_weight * smoothness;
    }

    void PreconditionedDescent(const Eigen::VectorXd &gradient,
                               Eigen::VectorXd &direction) const {
        if (!gradient.allFinite())
            throw std::invalid_argument(
                "combined path objective gradient must be finite");
        direction = (-gradient.array() * inverse_weights_.array()).matrix();
        if (!direction.allFinite()) {
            // Only the relative ratios matter after normalization. Form them
            // as mantissa/exponent pairs so even a subnormal positive weight
            // cannot overflow its reciprocal or the preconditioned gradient.
            // This also works where long double has the range of double.
            const auto ratio = [&](Eigen::Index i, int &exponent) {
                int gradient_exponent, weight_exponent, ratio_exponent;
                const double mantissa =
                    std::frexp(std::frexp(std::abs(gradient[i]), &gradient_exponent) /
                                   std::frexp(weights_[i], &weight_exponent),
                               &ratio_exponent);
                exponent = gradient_exponent - weight_exponent + ratio_exponent;
                return mantissa;
            };
            int largest_exponent = std::numeric_limits<int>::min();
            double largest_mantissa = 0.0;
            for (Eigen::Index i = 0; i < gradient.size(); ++i) {
                if (gradient[i] == 0.0)
                    continue;
                int exponent;
                const double mantissa = ratio(i, exponent);
                if (exponent > largest_exponent ||
                    (exponent == largest_exponent && mantissa > largest_mantissa)) {
                    largest_exponent = exponent;
                    largest_mantissa = mantissa;
                }
            }
            for (Eigen::Index i = 0; i < gradient.size(); ++i) {
                if (gradient[i] == 0.0) {
                    direction[i] = 0.0;
                    continue;
                }
                int exponent;
                const double mantissa = ratio(i, exponent);
                direction[i] = -std::copysign(std::scalbn(mantissa / largest_mantissa,
                                                          exponent - largest_exponent),
                                              gradient[i]);
            }
            return;
        }
        const double maximum = direction.cwiseAbs().maxCoeff();
        if (maximum > 1e-15)
            direction /= maximum;
    }

    double StateCostValue(const Eigen::VectorXd &state) {
        if (!state_cost_ || options_.state_cost_weight == 0.0)
            return 0.0;
        ++statistics_.state_cost_evaluations;
        const double value = state_cost_(state);
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument(
                "state cost must return a finite, non-negative value");
        return value;
    }

    Eigen::VectorXd StateCostGradient(const Eigen::VectorXd &state,
                                      double current_cost) {
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(state.size());
        if (!state_cost_ || options_.state_cost_weight == 0.0)
            return gradient;
        if (state_cost_gradient_) {
            ++statistics_.state_cost_gradient_evaluations;
            gradient = state_cost_gradient_(state);
            if (gradient.size() != state.size() || !gradient.allFinite())
                throw std::invalid_argument(
                    "state cost gradient must be finite and match the state "
                    "dimension");
            return gradient;
        }
        Eigen::VectorXd positive = state;
        Eigen::VectorXd negative = state;
        for (Eigen::Index i = 0; i < state.size(); ++i) {
            const bool continuous = continuous_[static_cast<std::size_t>(i)];
            // Keep circular samples on opposite sides of the current point.
            // Steps of pi or more can alias or reverse the two sample
            // directions.
            const double step = continuous
                                    ? std::min(options_.finite_difference_step,
                                               0.5 * detail::kPi)
                                    : options_.finite_difference_step;
            positive[i] = NormalizeCoordinate(state[i] + step, i);
            negative[i] = NormalizeCoordinate(state[i] - step, i);
            const auto difference = [continuous](double from, double to) {
                const double value = to - from;
                return continuous ? detail::WrappedDifference(value) : value;
            };
            double forward_step = difference(state[i], positive[i]);
            double backward_step = difference(negative[i], state[i]);
            // Discard a nearly collapsed side before evaluating its cost, since
            // dividing its roundoff by a tiny offset can dominate the
            // derivative.
            const double relative_tolerance =
                std::sqrt(std::numeric_limits<double>::epsilon());
            if (forward_step < relative_tolerance * backward_step)
                forward_step = 0.0;
            else if (backward_step < relative_tolerance * forward_step)
                backward_step = 0.0;
            const double span = forward_step + backward_step;
            if (span > 0.0) {
                const double positive_cost = forward_step <= 0.0
                                                 ? current_cost
                                                 : StateCostValue(positive);
                if (TimedOut()) {
                    positive[i] = state[i];
                    negative[i] = state[i];
                    break;
                }
                const double negative_cost = backward_step <= 0.0
                                                 ? current_cost
                                                 : StateCostValue(negative);
                if (forward_step > 0.0 && backward_step > 0.0 &&
                    forward_step != backward_step) {
                    // Derivative of the nonuniform three-point interpolant at
                    // state. A secant estimates it at the samples' midpoint
                    // instead and can point uphill after joint-limit
                    // projection.
                    gradient[i] =
                        (backward_step / span) *
                            ((positive_cost - current_cost) / forward_step) +
                        (forward_step / span) *
                            ((current_cost - negative_cost) / backward_step);
                } else {
                    gradient[i] = (positive_cost - negative_cost) / span;
                }
            }
            positive[i] = state[i];
            negative[i] = state[i];
            if (TimedOut())
                break;
        }
        if (!gradient.allFinite())
            throw std::invalid_argument(
                "finite-difference state cost gradient must be finite");
        return gradient;
    }

    bool IsStateValid(const Eigen::VectorXd &state) {
        if (!validator_)
            return true;
        ++statistics_.collision_checks;
        return validator_(state);
    }

    std::size_t SegmentCount(const Eigen::VectorXd &difference) const {
        return detail::SegmentCount(difference, options_.edge_resolution);
    }

    bool IsMotionValid(const Eigen::VectorXd &from, const Eigen::VectorXd &to) {
        if (TimedOut())
            return false;
        if (!validator_)
            return true;
        const Eigen::VectorXd difference = Difference(from, to);
        const std::size_t segments = SegmentCount(difference);
        for (std::size_t i = 1; i < segments; ++i) {
            if (TimedOut() ||
                !IsStateValid(NormalizeInterpolation(
                    from + static_cast<double>(i) / segments * difference)) ||
                TimedOut())
                return false;
        }
        // Reconstructing the endpoint can round it across a validity boundary.
        // Validate the exact waypoint that will be returned to the caller.
        return !TimedOut() && IsStateValid(to) && !TimedOut();
    }

    bool IsMotionInteriorValid(const Eigen::VectorXd &from,
                               const Eigen::VectorXd &to) {
        if (TimedOut())
            return false;
        if (!validator_)
            return true;
        const Eigen::VectorXd difference = Difference(from, to);
        const std::size_t segments = SegmentCount(difference);
        for (std::size_t i = 1; i < segments; ++i) {
            if (TimedOut() ||
                !IsStateValid(NormalizeInterpolation(
                    from + static_cast<double>(i) / segments * difference)) ||
                TimedOut())
                return false;
        }
        return true;
    }

  private:
    double NormalizeCoordinate(double value, Eigen::Index index) const {
        return detail::NormalizeCoordinate(value, index, lower_, upper_,
                                           continuous_);
    }

    const Eigen::VectorXd &lower_;
    const Eigen::VectorXd &upper_;
    const Eigen::VectorXd &weights_;
    Eigen::VectorXd inverse_weights_;
    bool has_continuous_{false};
    const std::vector<bool> &continuous_;
    const PathOptimizer::StateValidator &validator_;
    const PathOptimizer::StateCost &state_cost_;
    const PathOptimizer::StateCostGradient &state_cost_gradient_;
    const PathOptimizationOptions &options_;
    PathOptimizationStatistics &statistics_;
    std::chrono::steady_clock::time_point deadline_;
};

} // namespace

PathOptimizer::PathOptimizer(Eigen::VectorXd lower_limits,
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
    const Eigen::VectorXd ranges = upper_limits_ - lower_limits_;
    weights_ = ranges.array().square().inverse().matrix();
    if (!ranges.allFinite() || !weights_.allFinite() ||
        (weights_.array() <= 0.0).any())
        throw std::invalid_argument(
            "joint limit ranges must produce finite positive default weights");
    continuous_.assign(static_cast<std::size_t>(lower_limits_.size()), false);
}

void PathOptimizer::SetStateValidator(StateValidator validator) {
    validator_ = std::move(validator);
}

void PathOptimizer::SetStateCost(StateCost state_cost) {
    state_cost_ = std::move(state_cost);
}

void PathOptimizer::SetStateCostGradient(
    StateCostGradient state_cost_gradient) {
    state_cost_gradient_ = std::move(state_cost_gradient);
}

void PathOptimizer::SetJointWeights(const Eigen::VectorXd &weights) {
    if (weights.size() != lower_limits_.size() || !weights.allFinite() ||
        (weights.array() <= 0.0).any())
        throw std::invalid_argument(
            "joint weights must be finite and positive");
    weights_ = weights;
}

void PathOptimizer::SetContinuousJoints(
    const std::vector<std::size_t> &indices) {
    std::vector<bool> continuous(static_cast<std::size_t>(lower_limits_.size()),
                                 false);
    for (std::size_t index : indices) {
        if (index >= continuous.size())
            throw std::out_of_range("continuous joint index");
        if (upper_limits_[static_cast<Eigen::Index>(index)] -
                lower_limits_[static_cast<Eigen::Index>(index)] <
            2.0 * detail::kPi - 1e-9)
            throw std::invalid_argument(
                "continuous joints require a range of at least 2*pi");
        continuous[index] = true;
    }
    continuous_ = std::move(continuous);
}

PathOptimizationResult
PathOptimizer::Optimize(const std::vector<Eigen::VectorXd> &path,
                        const PathOptimizationOptions &options) const {
    PathOptimizationResult result;
    const auto started = std::chrono::steady_clock::now();
    auto finish = [&](PathOptimizationStatus status, std::string message) {
        result.status = status;
        result.message = std::move(message);
        result.statistics.optimization_time_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started)
                .count();
        return result;
    };
    if (path.size() < 2 || options.max_iterations == 0 ||
        !std::isfinite(options.timeout_seconds) ||
        !std::isfinite(options.step_size) ||
        !std::isfinite(options.line_search_decay) ||
        !std::isfinite(options.edge_resolution) ||
        !std::isfinite(options.length_weight) ||
        !std::isfinite(options.smoothness_weight) ||
        !std::isfinite(options.state_cost_weight) ||
        !std::isfinite(options.finite_difference_step) ||
        !std::isfinite(options.state_cost_step_size) ||
        !std::isfinite(options.minimum_improvement) ||
        options.timeout_seconds <= 0.0 || options.step_size <= 0.0 ||
        options.step_size > 1.0 || options.line_search_steps == 0 ||
        options.line_search_decay <= 0.0 || options.line_search_decay >= 1.0 ||
        options.edge_resolution <= 0.0 || options.length_weight < 0.0 ||
        options.smoothness_weight < 0.0 || options.state_cost_weight < 0.0 ||
        options.finite_difference_step <= 0.0 ||
        options.state_cost_step_size < 0.0 ||
        options.length_weight + options.smoothness_weight +
                options.state_cost_weight <=
            0.0 ||
        options.minimum_improvement < 0.0) {
        return finish(PathOptimizationStatus::INVALID_PATH,
                      "invalid path or optimization options");
    }
    const auto deadline =
        detail::DeadlineAfter(started, options.timeout_seconds);
    OptimizationContext context(lower_limits_, upper_limits_, weights_,
                                continuous_, validator_, state_cost_,
                                state_cost_gradient_, options,
                                result.statistics, deadline);
    result.path = path;
    for (const auto &state : result.path) {
        if (state.size() != lower_limits_.size() || !state.allFinite() ||
            (state.array() < lower_limits_.array()).any() ||
            (state.array() > upper_limits_.array()).any()) {
            result.path.clear();
            return finish(PathOptimizationStatus::INVALID_PATH,
                          "path contains an invalid state");
        }
    }
    const bool start_valid = context.IsStateValid(result.path.front());
    if (context.TimedOut()) {
        result.path.clear();
        return finish(PathOptimizationStatus::TIMEOUT,
                      "time limit reached while validating the input path");
    }
    if (!start_valid) {
        result.path.clear();
        return finish(PathOptimizationStatus::INVALID_PATH,
                      "path start state is invalid");
    }
    for (std::size_t i = 1; i < result.path.size(); ++i) {
        if (!context.IsMotionValid(result.path[i - 1], result.path[i])) {
            if (context.TimedOut()) {
                result.path.clear();
                return finish(
                    PathOptimizationStatus::TIMEOUT,
                    "time limit reached while validating the input path");
            }
            result.path.clear();
            return finish(PathOptimizationStatus::INVALID_PATH,
                          "input path contains an invalid edge");
        }
    }
    result.feasible = true;
    result.statistics.initial_path_length = context.PathLength(result.path);
    if (!std::isfinite(result.statistics.initial_path_length)) {
        result.path.clear();
        result.feasible = false;
        return finish(PathOptimizationStatus::INVALID_PATH,
                      "path objective is not finite");
    }
    if (result.path.size() == 2) {
        result.statistics.initial_objective =
            options.length_weight * result.statistics.initial_path_length;
        if (!std::isfinite(result.statistics.initial_objective)) {
            result.path.clear();
            result.feasible = false;
            return finish(PathOptimizationStatus::INVALID_PATH,
                          "path objective is not finite");
        }
        result.statistics.final_objective = result.statistics.initial_objective;
        result.statistics.final_path_length =
            result.statistics.initial_path_length;
        return finish(PathOptimizationStatus::UNCHANGED,
                      "path has no interior waypoints to optimize");
    }
    std::vector<double> state_costs;
    double total_state_cost = 0.0;
    if (options.state_cost_weight > 0.0) {
        if (!state_cost_) {
            result.path.clear();
            result.feasible = false;
            return finish(PathOptimizationStatus::INVALID_PATH,
                          "state_cost_weight requires a state cost callback");
        }
        state_costs.assign(result.path.size(), 0.0);
        for (std::size_t i = 1; i + 1 < result.path.size(); ++i) {
            state_costs[i] = context.StateCostValue(result.path[i]);
            total_state_cost += state_costs[i];
            if (context.TimedOut()) {
                result.statistics.initial_objective =
                    std::numeric_limits<double>::quiet_NaN();
                result.statistics.final_objective =
                    std::numeric_limits<double>::quiet_NaN();
                result.statistics.final_path_length =
                    result.statistics.initial_path_length;
                return finish(
                    PathOptimizationStatus::TIMEOUT,
                    "time limit reached while evaluating the input path; "
                    "returning the validated feasible path");
            }
        }
    }
    result.statistics.initial_objective =
        context.GeometryObjective(result.path) +
        options.state_cost_weight * total_state_cost;
    if (!std::isfinite(result.statistics.initial_objective)) {
        result.path.clear();
        result.feasible = false;
        return finish(PathOptimizationStatus::INVALID_PATH,
                      "path objective is not finite");
    }
    detail::PathGeometryWorkspace geometry(weights_, continuous_,
                                           options.length_weight,
                                           options.smoothness_weight);
    Eigen::VectorXd gradient(lower_limits_.size());
    Eigen::VectorXd direction(lower_limits_.size());
    Eigen::VectorXd previous(lower_limits_.size());
    Eigen::VectorXd candidate(lower_limits_.size());
    bool any_update = false;
    for (std::size_t sweep = 0; sweep < options.max_iterations; ++sweep) {
        ++result.statistics.iterations;
        bool iteration_update = false;
        const std::size_t interior_count = result.path.size() - 2;
        const bool reverse_sweep = sweep % 2 == 1;
        for (std::size_t offset = 0; offset < interior_count; ++offset) {
            const std::size_t i =
                reverse_sweep ? interior_count - offset : offset + 1;
            if (context.TimedOut())
                break;
            ++result.statistics.attempted_updates;
            const double current_state_cost =
                options.state_cost_weight > 0.0 ? state_costs[i] : 0.0;
            geometry.SetWaypoint(result.path, i);
            geometry.Gradient(gradient);
            if (options.state_cost_weight > 0.0) {
                gradient += options.state_cost_weight *
                            options.state_cost_step_size *
                            context.StateCostGradient(result.path[i],
                                                      current_state_cost);
            }
            if (context.TimedOut())
                break;
            context.PreconditionedDescent(gradient, direction);
            if (direction.cwiseAbs().maxCoeff() <= 1e-15)
                continue;
            previous = result.path[i];
            const double previous_local_objective =
                geometry.LocalObjective() +
                options.state_cost_weight * current_state_cost;
            double step = options.step_size;
            bool accepted = false;
            for (std::size_t search = 0; search < options.line_search_steps;
                 ++search, step *= options.line_search_decay) {
                ++result.statistics.line_search_evaluations;
                candidate = previous + step * direction;
                context.Normalize(candidate);
                if (context.Difference(previous, candidate)
                        .cwiseAbs()
                        .maxCoeff() <= 1e-15)
                    break;
                result.path[i] = candidate;
                const double candidate_geometry = geometry.CandidateObjective(
                    result.path[i - 1], candidate, result.path[i + 1]);
                if (!std::isfinite(candidate_geometry)) {
                    result.path[i] = previous;
                    continue;
                }
                // Other waypoints are unchanged. Comparing only the affected
                // non-negative terms prevents unrelated large costs from
                // hiding an improvement or accepting a locally worse trial.
                // The candidate state cost is non-negative, so geometry alone
                // is a lower bound before invoking that callback.
                if (previous_local_objective - candidate_geometry <
                    options.minimum_improvement) {
                    result.path[i] = previous;
                    continue;
                }
                const double candidate_state_cost =
                    context.StateCostValue(candidate);
                if (context.TimedOut()) {
                    result.path[i] = previous;
                    break;
                }
                const double candidate_local_objective =
                    candidate_geometry +
                    options.state_cost_weight * candidate_state_cost;
                if (previous_local_objective - candidate_local_objective >=
                        options.minimum_improvement &&
                    context.IsStateValid(candidate) &&
                    context.IsMotionInteriorValid(result.path[i - 1], candidate) &&
                    context.IsMotionInteriorValid(candidate, result.path[i + 1])) {
                    if (options.state_cost_weight > 0.0)
                        state_costs[i] = candidate_state_cost;
                    accepted = true;
                    break;
                }
                result.path[i] = previous;
                if (context.TimedOut())
                    break;
            }
            if (!accepted)
                continue;
            ++result.statistics.accepted_updates;
            iteration_update = true;
            any_update = true;
        }
        if (context.TimedOut() || !iteration_update)
            break;
    }
    result.statistics.final_objective =
        context.GeometryObjective(result.path) +
        options.state_cost_weight *
            std::accumulate(state_costs.begin(), state_costs.end(), 0.0);
    result.statistics.final_path_length = context.PathLength(result.path);
    if (context.TimedOut())
        return finish(PathOptimizationStatus::TIMEOUT,
                      "time limit reached; returning the best feasible path");
    return finish(any_update ? PathOptimizationStatus::OPTIMIZED
                             : PathOptimizationStatus::UNCHANGED,
                  any_update ? "path optimized"
                             : "path already locally optimal");
}

} // namespace holistic_motion::robotics::planning
