#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "AllocationCounter.h"
#include "holistic_motion/kinematics/srs/SRSKinematics.h"

using namespace holistic_motion::robotics;

namespace {
std::vector<JointNode> MakeModel(double wrist_offset) {
    const double pi = std::acos(-1.0), half_pi = pi / 2.0;
    const std::array<Eigen::Vector3d, 7> translations{{{0, 0, 0},
                                                       {0, 0, 0.1025},
                                                       {0, 0.260, 0},
                                                       {0, 0, 0},
                                                       {0, 0.166, 0},
                                                       {0, wrist_offset, 0.098},
                                                       {0, 0, 0}}};
    const std::array<Eigen::Vector3d, 7> rotations{{{0, 0, 0},
                                                    {half_pi, 0, pi},
                                                    {half_pi, -half_pi, pi},
                                                    {half_pi, 0, 0},
                                                    {-half_pi, pi, 0},
                                                    {half_pi, 0, 0},
                                                    {-half_pi, pi, half_pi}}};
    std::vector<JointNode> nodes;
    for (std::size_t i = 0; i < translations.size(); ++i)
        nodes.emplace_back(SE3d(translations[i], SO3d(rotations[i])),
                           (i == 5 ? -1.0 : 1.0) * Eigen::Vector3d::UnitZ(),
                           JointType::REVOLUTE, -3.0, 3.0);
    nodes.emplace_back(SE3d(Eigen::Vector3d(-0.066, 0, 0), SO3d(0, -half_pi, 0)),
                       Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0);
    return nodes;
}

template <typename Query>
void Measure(const char *name, double offset, int query_count, Query &&query) {
    double checksum = 0.0;
    for (int sample = 0; sample < 64; ++sample)
        checksum += query(sample);
    hm_benchmark::StartCounting();
    for (int sample = 0; sample < 64; ++sample)
        checksum += query(sample);
    const double allocations = static_cast<double>(hm_benchmark::StopCounting()) / 64.0;
    std::vector<double> times;
    for (int repeat = 0; repeat < 8; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        double sum = 0.0;
        for (int sample = 0; sample < query_count; ++sample)
            sum += query(sample);
        const double elapsed = std::chrono::duration<double, std::micro>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count() /
                               query_count;
        if (repeat > 0)
            times.push_back(elapsed);
        if (repeat > 0 && sum != checksum)
            throw std::runtime_error("repeated SRS queries changed their outputs");
        checksum = sum;
    }
    std::sort(times.begin(), times.end());
    std::cout << name << ',' << offset << ',' << times[times.size() / 2] << ','
              << allocations << ',' << checksum << '\n';
}
} // namespace

int main() {
    std::cout << "query,wrist_offset,median_us,allocations,checksum\n"
              << std::setprecision(17);
    for (double offset : {0.0, 3e-5}) {
        const SRSKinematics solver(MakeModel(offset));
        if (!solver.AnalyzeGeometry().closed_form_compatible)
            return 1;
        std::array<Eigen::VectorXd, 32> seeds;
        std::array<SE3d, 32> targets;
        std::array<SRSConfiguration, 32> configurations;
        for (std::size_t sample = 0; sample < seeds.size(); ++sample) {
            auto &seed = seeds[sample];
            seed.resize(7);
            seed << 0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2;
            for (int joint = 0; joint < 7; ++joint)
                seed[joint] += 0.12 * std::sin(0.3 * (sample + 1) * (joint + 1));
            if (!solver.GetFK(seed, targets[sample]))
                return 2;
            configurations[sample] = solver.GetConfiguration(seed);
        }
        Measure("configuration", offset, 50000, [&](int query) {
            const auto result = solver.GetConfiguration(seeds[query % seeds.size()]);
            return result.redundancy + result.shoulder + result.elbow + result.wrist;
        });
        Eigen::VectorXd solution;
        for (std::size_t sample = 0; sample < seeds.size(); ++sample) {
            SE3d actual;
            if (!solver.GetAnalyticSolution(targets[sample], configurations[sample],
                                            seeds[sample], solution) ||
                !solver.GetFK(solution, actual) ||
                (actual.GetTranslation() - targets[sample].GetTranslation()).norm() >
                    1e-5 ||
                actual.GetQuat().angularDistance(targets[sample].GetQuat()) > 1e-5)
                return 3;
        }
        Measure("analytic", offset, 10000, [&](int query) {
            const std::size_t sample = query % seeds.size();
            return solver.GetAnalyticSolution(targets[sample], configurations[sample],
                                              seeds[sample], solution)
                       ? 10.0 + solution.sum()
                       : 0.0;
        });
        Measure("detailed", offset, 2000, [&](int query) {
            const std::size_t sample = query % seeds.size();
            const auto report = solver.SolveDetailed(
                targets[sample], seeds[sample], SRSSolveMethod::ALL_CONFIGURATIONS);
            if (report.status != SRSSolveStatus::SUCCESS)
                throw std::runtime_error("SRS benchmark query did not converge");
            double sum = 0.0;
            for (std::size_t index = 0; index < report.solutions.size(); ++index)
                sum += 10.0 + report.solutions[index].sum() +
                       report.minimum_singular_values[index] +
                       report.minimum_joint_limit_margins[index];
            return sum;
        });
    }
}
