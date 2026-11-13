#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"

using namespace holistic_motion::robotics::planning;

int main() {
    PlanningOptions options;
    options.max_iterations = 64;
    options.edge_resolution = 0.01;
    options.simplify_path = false;
    const Eigen::VectorXd lower = Eigen::VectorXd::Constant(1, -1.0);
    const Eigen::VectorXd upper = -lower;
    const Eigen::VectorXd start = Eigen::VectorXd::Constant(1, -0.8);
    const Eigen::VectorXd goal = -start;

    auto lifetime = std::make_shared<int>(42);
    std::weak_ptr<int> retained = lifetime;
    bool alive_during_callback = true;
    std::unique_ptr<SamplingPlanner> planner;
    planner = std::make_unique<SamplingPlanner>(
        lower, upper, [&, lifetime](const Eigen::VectorXd &q) {
            options.edge_resolution = std::numeric_limits<double>::infinity();
            planner->SetStateValidator({});
            alive_during_callback = alive_during_callback && !retained.expired();
            return std::abs(q[0]) >= 0.1;
        });
    lifetime.reset();
    const auto result = planner->Plan(start, goal, options);
    if (result.status != PlanningStatus::NO_SOLUTION || !alive_during_callback ||
        !retained.expired())
        return 1;
    if (planner->Plan(start, goal, options).status != PlanningStatus::INVALID_PROBLEM)
        return 2;
    options.edge_resolution = 0.01;
    if (!planner->Plan(start, goal, options).Success())
        return 3;

    PathOptimizationOptions optimization_options;
    optimization_options.edge_resolution = 0.01;
    lifetime = std::make_shared<int>(43);
    retained = lifetime;
    std::unique_ptr<PathOptimizer> optimizer;
    optimizer = std::make_unique<PathOptimizer>(
        lower, upper, [&, lifetime](const Eigen::VectorXd &q) {
            optimization_options.edge_resolution =
                std::numeric_limits<double>::infinity();
            optimizer->SetStateValidator({});
            alive_during_callback = alive_during_callback && !retained.expired();
            return std::abs(q[0]) >= 0.1;
        });
    lifetime.reset();
    const std::vector<Eigen::VectorXd> path{start, goal};
    const auto optimized = optimizer->Optimize(path, optimization_options);
    if (optimized.Success() || !alive_during_callback || !retained.expired() ||
        optimized.status != PathOptimizationStatus::INVALID_PATH)
        return 4;
    optimization_options.edge_resolution = 0.01;
    if (!optimizer->Optimize(path, optimization_options).Success())
        return 5;

    // Replacing either objective callback inside its own invocation must keep
    // that callable alive and keep one cost/gradient pair for the full request.
    PathOptimizer objective(lower, upper);
    optimization_options.state_cost_weight = 1.0;
    optimization_options.length_weight = 0.0;
    optimization_options.smoothness_weight = 0.0;
    optimization_options.max_iterations = 8;
    lifetime = std::make_shared<int>(44);
    retained = lifetime;
    objective.SetStateCost([&, lifetime](const Eigen::VectorXd &q) {
        objective.SetStateCost({});
        alive_during_callback = alive_during_callback && !retained.expired();
        return q[0] * q[0];
    });
    objective.SetStateCostGradient([&objective](const Eigen::VectorXd &q) {
        objective.SetStateCostGradient({});
        return (2.0 * q).eval();
    });
    lifetime.reset();
    const auto cost_result =
        objective.Optimize({start, goal, start}, optimization_options);
    if (!cost_result.Success() || !alive_during_callback || !retained.expired() ||
        cost_result.statistics.state_cost_gradient_evaluations == 0 ||
        cost_result.statistics.accepted_updates == 0 ||
        !(cost_result.statistics.final_objective <
          cost_result.statistics.initial_objective))
        return 6;

    for (auto algorithm : {SamplingAlgorithm::RRT_CONNECT, SamplingAlgorithm::RRT_STAR,
                           SamplingAlgorithm::INFORMED_RRT_STAR}) {
        for (bool edit_start : {false, true}) {
            Eigen::VectorXd requested_start = start, requested_goal = goal;
            bool edited = false;
            SamplingPlanner changing_endpoints(
                lower, upper, [&](const Eigen::VectorXd &) {
                    if (!edited) {
                        edited = true;
                        (edit_start ? requested_start : requested_goal).resize(0);
                    }
                    return true;
                });
            PlanningOptions isolated_options;
            isolated_options.algorithm = algorithm;
            const auto planned = changing_endpoints.Plan(
                requested_start, requested_goal, isolated_options);
            if (!edited || !planned.Success() || planned.path.size() != 2 ||
                !planned.path.front().isApprox(start, 0.0) ||
                !planned.path.back().isApprox(goal, 0.0))
                return 7;
            if (changing_endpoints
                    .Plan(requested_start, requested_goal, isolated_options)
                    .status != PlanningStatus::INVALID_PROBLEM)
                return 8;
        }
    }

    std::vector<Eigen::VectorXd> requested_path{start, goal};
    PathOptimizer changing_input(lower, upper, [&](const Eigen::VectorXd &) {
        requested_path.clear();
        return true;
    });
    const auto independent_path = changing_input.Optimize(requested_path);
    if (!requested_path.empty() || !independent_path.Success() ||
        independent_path.path.size() != 2 ||
        !independent_path.path.front().isApprox(start, 0.0) ||
        !independent_path.path.back().isApprox(goal, 0.0))
        return 9;

    // Reject malformed endpoints before invoking any user validator.
    std::size_t validation_calls = 0;
    PathOptimizer validating_input(lower, upper, [&](const Eigen::VectorXd &) {
        ++validation_calls;
        return true;
    });
    const std::vector<Eigen::VectorXd> invalid_states{
        Eigen::VectorXd::Zero(2),
        Eigen::VectorXd::Constant(1, std::numeric_limits<double>::quiet_NaN()),
        Eigen::VectorXd::Constant(1, std::numeric_limits<double>::infinity()),
        Eigen::VectorXd::Constant(1, upper[0] + 1.0)};
    for (const auto &invalid_state : invalid_states) {
        for (bool invalid_start : {false, true}) {
            std::vector<Eigen::VectorXd> invalid_path{start, goal};
            invalid_path[invalid_start ? 0 : 1] = invalid_state;
            const auto invalid_result = validating_input.Optimize(invalid_path);
            if (validation_calls != 0 || invalid_result.statistics.collision_checks != 0 ||
                invalid_result.status != PathOptimizationStatus::INVALID_PATH ||
                invalid_result.feasible || !invalid_result.path.empty() ||
                invalid_result.message != "path contains an invalid state" ||
                !invalid_result.joint_space || invalid_result.joint_space->GetDOF() != 1)
                return 12;
        }
    }

    // A nested request from an interior sample must not overwrite the outer
    // request's interpolation difference or the sample passed to its callback.
    bool nested = false, nested_success = false, sample_preserved = true;
    std::vector<double> outer_samples;
    std::unique_ptr<SamplingPlanner> reentrant;
    reentrant = std::make_unique<SamplingPlanner>(
        lower, upper, [&](const Eigen::VectorXd &state) {
            if (nested)
                return true;
            const double position = state[0];
            if (!nested_success && std::abs(position + 0.7) < 1e-12) {
                nested = true;
                PlanningOptions inner_options;
                inner_options.edge_resolution = 0.2;
                const auto inner_result = reentrant->Plan(
                    Eigen::VectorXd::Constant(1, -0.3),
                    Eigen::VectorXd::Constant(1, 0.4), inner_options);
                nested_success = inner_result.Success();
                nested = false;
                sample_preserved = state[0] == position;
            }
            outer_samples.push_back(state[0]);
            return true;
        });
    PlanningOptions outer_options;
    outer_options.edge_resolution = 0.1;
    const auto reentrant_result = reentrant->Plan(start, goal, outer_options);
    if (!reentrant_result.Success() || !nested_success || !sample_preserved ||
        outer_samples.size() != 18 ||
        reentrant_result.statistics.collision_checks != outer_samples.size())
        return 10;
    for (std::size_t index = 3; index < outer_samples.size(); ++index)
        if (std::abs(outer_samples[index] - (-0.8 + 0.1 * (index - 2))) > 1e-12)
            return 11;
    return 0;
}
