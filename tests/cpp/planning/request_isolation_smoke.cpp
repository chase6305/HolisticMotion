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
    return 0;
}
