#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "holistic_motion/planning/SamplingPlanner.h"

using namespace holistic_motion::robotics::planning;

// Synthetic joint-space wall: no external robot assets or collision dependency.
// Fix the iteration budget and seed so faster builds perform the same search.
int main(int argc, char **argv) {
    double goal_bias = 0.05;
    try {
        if (argc > 2)
            throw std::invalid_argument("too many arguments");
        if (argc == 2) {
            const std::string argument(argv[1]);
            std::size_t consumed;
            goal_bias = std::stod(argument, &consumed);
            if (consumed != argument.size() || !std::isfinite(goal_bias) ||
                goal_bias < 0.0 || goal_bias > 1.0)
                throw std::invalid_argument("goal bias must be between zero and one");
        }
    } catch (const std::exception &) {
        std::cerr << "usage: sampling_planner_benchmark [goal_bias in [0, 1]]\n";
        return 2;
    }
    std::cout << "dof,continuous,goal_bias,median_ms,success,iterations,nodes,checks,"
                 "path_length\n";
    for (const Eigen::Index dof : {2, 7, 14}) {
        for (const bool continuous : {false, true}) {
            Eigen::VectorXd lower = Eigen::VectorXd::Constant(dof, -1.0);
            Eigen::VectorXd upper = Eigen::VectorXd::Constant(dof, 1.0);
            if (continuous) {
                lower[0] = -std::acos(-1.0);
                upper[0] = std::acos(-1.0);
            }
            SamplingPlanner planner(lower, upper, [](const Eigen::VectorXd &q) {
                return !(std::abs(q[0]) < 0.2 && std::abs(q[1]) < 0.75);
            });
            Eigen::VectorXd weights(dof);
            for (Eigen::Index index = 0; index < dof; ++index)
                weights[index] = 0.25 * (1 + index % 3);
            planner.SetJointWeights(weights);
            if (continuous)
                planner.SetContinuousJoints({0});

            Eigen::VectorXd start = Eigen::VectorXd::Zero(dof);
            Eigen::VectorXd goal = start;
            start[0] = -0.8;
            goal[0] = 0.8;
            PlanningOptions options;
            options.algorithm = SamplingAlgorithm::RRT_STAR;
            options.timeout_seconds = 60.0;
            options.max_iterations = 4000;
            options.extension_range = 0.25 * std::sqrt(dof / 2.0);
            options.edge_resolution = 0.02;
            options.shortcut_attempts = 0;
            options.random_seed = 7;
            options.goal_bias = goal_bias;

            std::vector<double> elapsed;
            PlanningStatistics statistics;
            bool success = false;
            for (int repeat = 0; repeat < 6; ++repeat) {
                const auto begin = std::chrono::steady_clock::now();
                const auto result = planner.Plan(start, goal, options);
                const double milliseconds =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin)
                        .count();
                if ((!result.Success() &&
                     result.status != PlanningStatus::NO_SOLUTION) ||
                    result.statistics.iterations != options.max_iterations) {
                    std::cerr << "benchmark did not complete its search: "
                              << result.message << '\n';
                    return 1;
                }
                if (repeat > 0)
                    elapsed.push_back(milliseconds);
                statistics = result.statistics;
                success = result.Success();
            }
            std::sort(elapsed.begin(), elapsed.end());
            std::cout << dof << ',' << continuous << ',' << goal_bias << ','
                      << std::fixed << std::setprecision(3)
                      << elapsed[elapsed.size() / 2] << ',' << success << ','
                      << statistics.iterations << ',' << statistics.tree_nodes << ','
                      << statistics.collision_checks << ',' << std::setprecision(12)
                      << statistics.final_path_length << '\n';
        }
    }
}
