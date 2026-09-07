#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "holistic_motion/planning/PathOptimizer.h"

using namespace holistic_motion::robotics::planning;

namespace {
double Checksum(const std::vector<Eigen::VectorXd>& path) {
    double value = 0.0;
    for (std::size_t i = 0; i < path.size(); ++i)
        for (Eigen::Index joint = 0; joint < path[i].size(); ++joint)
            value += (i + 1) * (joint + 1) * path[i][joint];
    return value;
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 1 && argc != 5) {
        std::cerr << "usage: path_optimizer_benchmark [dof waypoints "
                     "continuous validator]\n";
        return 2;
    }
    bool matched = false;
    constexpr double kPi = 3.14159265358979323846;
    std::cout
        << "dof,waypoints,continuous,validator,time_us,iterations,attempted,"
           "line_search,accepted,checks,initial_objective,final_objective,"
           "final_length,checksum\n"
        << std::setprecision(17);
    for (int dof : {2, 7, 14}) {
        for (int count : {32, 256}) {
            for (bool continuous : {false, true}) {
                for (bool validate : {false, true}) {
                    if (argc == 5 && (std::to_string(dof) != argv[1] ||
                                      std::to_string(count) != argv[2] ||
                                      std::to_string(continuous) != argv[3] ||
                                      std::to_string(validate) != argv[4]))
                        continue;
                    matched = true;
                    PathOptimizer::StateValidator validator;
                    if (validate)
                        validator = [](const Eigen::VectorXd& q) {
                            return q.allFinite() && q.squaredNorm() < 200.0;
                        };
                    PathOptimizer optimizer(
                        Eigen::VectorXd::Constant(dof, -kPi),
                        Eigen::VectorXd::Constant(dof, kPi), validator);
                    Eigen::VectorXd weights(dof);
                    for (int joint = 0; joint < dof; ++joint)
                        weights[joint] = std::ldexp(1.0, joint % 3 - 1);
                    optimizer.SetJointWeights(weights);
                    if (continuous) optimizer.SetContinuousJoints({0});
                    std::vector<Eigen::VectorXd> path;
                    for (int i = 0; i < count; ++i) {
                        const double t = static_cast<double>(i) / (count - 1);
                        Eigen::VectorXd q(dof);
                        for (int joint = 0; joint < dof; ++joint)
                            q[joint] = -0.8 + 1.6 * t +
                                       0.3 * std::sin(0.8 * i + joint);
                        if (continuous)
                            q[0] = std::remainder(kPi + q[0], 2.0 * kPi);
                        path.push_back(q);
                    }
                    PathOptimizationOptions options;
                    options.max_iterations = 12;
                    options.timeout_seconds = 30.0;
                    options.edge_resolution = 0.03;
                    options.step_size = 0.2;
                    options.line_search_steps = 6;
                    options.smoothness_weight = 0.5;
                    PathOptimizationResult reference =
                        optimizer.Optimize(path, options);
                    if (!reference.Success() ||
                        reference.status == PathOptimizationStatus::TIMEOUT ||
                        reference.statistics.iterations !=
                            options.max_iterations ||
                        reference.statistics.accepted_updates == 0)
                        return 1;
                    const double checksum = Checksum(reference.path);
                    std::vector<double> samples;
                    const int calls = count == 32 ? 32 : 4;
                    for (int group = -1; group < 7; ++group) {
                        const auto started = std::chrono::steady_clock::now();
                        for (int call = 0; call < calls; ++call) {
                            const auto result =
                                optimizer.Optimize(path, options);
                            if (!result.Success() ||
                                result.status ==
                                    PathOptimizationStatus::TIMEOUT ||
                                result.statistics.accepted_updates !=
                                    reference.statistics.accepted_updates ||
                                result.statistics.final_objective !=
                                    reference.statistics.final_objective)
                                throw std::runtime_error(
                                    "unstable optimization result");
                        }
                        if (group >= 0)
                            samples.push_back(
                                std::chrono::duration<double, std::micro>(
                                    std::chrono::steady_clock::now() - started)
                                    .count() /
                                calls);
                    }
                    std::sort(samples.begin(), samples.end());
                    const auto& stats = reference.statistics;
                    std::cout
                        << dof << ',' << count << ',' << continuous << ','
                        << validate << ',' << samples[3] << ','
                        << stats.iterations << ',' << stats.attempted_updates
                        << ',' << stats.line_search_evaluations << ','
                        << stats.accepted_updates << ','
                        << stats.collision_checks << ','
                        << stats.initial_objective << ','
                        << stats.final_objective << ','
                        << stats.final_path_length << ',' << checksum << '\n';
                }
            }
        }
    }
    return matched ? 0 : 2;
}
