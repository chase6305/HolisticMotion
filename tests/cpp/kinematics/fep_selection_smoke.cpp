#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "FEPSelection.h"

using namespace holistic_motion::robotics::fep_internal;

int main() {
    const Eigen::VectorXd seed = Eigen::VectorXd::LinSpaced(7, -0.3, 0.3);
    std::mt19937 generator(82);
    std::uniform_real_distribution<double> angle(-25.0, 25.0);
    CoordinatePeriods scaled_periods = kUnitCoordinatePeriods;
    for (std::size_t joint = 0; joint < scaled_periods.size(); ++joint)
        scaled_periods[joint] /= 0.5 + 0.25 * joint;
#ifdef HOLISTICMOTION_HAS_CUDA
    const bool available = CudaBackendAvailable();
    std::cout << "FEP scoring CUDA compiled/runtime: 1/" << available << "\n";
#else
    std::cout << "FEP scoring CUDA compiled/runtime: 0/0\n";
#endif
    for (const auto &periods : {kUnitCoordinatePeriods, scaled_periods}) {
        for (int count : {1, 31, 32, 129}) {
            std::vector<Eigen::VectorXd> candidates;
            for (int row = 0; row < count; ++row) {
                Eigen::VectorXd candidate(7);
                for (int joint = 0; joint < 7; ++joint)
                    candidate[joint] = angle(generator);
                candidates.push_back(std::move(candidate));
            }
            candidates.front() = seed;
            const auto cpu = ScoreCandidatesCPU(candidates, seed, periods);
            if (cpu.size() != candidates.size() || cpu.front() != 0.0)
                return 1;
            for (std::size_t row = 0; row < candidates.size(); ++row) {
                double expected = 0.0;
                for (int joint = 0; joint < 7; ++joint) {
                    const double axis_scale =
                        kUnitCoordinatePeriods[joint] / periods[joint];
                    const double delta =
                        (candidates[row][joint] - seed[joint]) * axis_scale;
                    const double phase =
                        std::atan2(std::sin(delta), std::cos(delta)) / axis_scale;
                    expected += phase * phase;
                }
                if (std::abs(cpu[row] - expected) > 1e-10)
                    return 1;
            }
#ifdef HOLISTICMOTION_HAS_CUDA
            std::vector<double> gpu{42.0};
            const bool scored = ScoreCandidatesCUDA(candidates, seed, gpu, periods);
            if (available) {
                if (!scored || gpu.size() != cpu.size())
                    return 1;
                for (std::size_t row = 0; row < gpu.size(); ++row)
                    if (!std::isfinite(gpu[row]) ||
                        std::abs(gpu[row] - cpu[row]) > 1e-11)
                        return 1;
            } else if (scored || !gpu.empty()) {
                std::cerr << "unavailable CUDA scoring retained stale output\n";
                return 1;
            }
#endif
        }
    }
    // With a half-scale axis, 2*pi of coordinate motion is a physical half
    // turn, not an equivalent seed. Ranking must use that axis's 4*pi period.
    auto half_turn = seed.eval();
    half_turn[0] += 2.0 * std::acos(-1.0);
    const auto half_turn_score = ScoreCandidatesCPU({half_turn}, seed, scaled_periods);
    if (half_turn_score.size() != 1 ||
        std::abs(half_turn_score[0] - 4.0 * std::pow(std::acos(-1.0), 2)) > 1e-12)
        return 1;
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        auto periods = scaled_periods;
        periods[0] = invalid;
        if (!ScoreCandidatesCPU({seed}, seed, periods).empty())
            return 1;
#ifdef HOLISTICMOTION_HAS_CUDA
        std::vector<double> scores{42.0};
        if (ScoreCandidatesCUDA({seed}, seed, scores, periods) || !scores.empty())
            return 1;
#endif
    }
    auto nonfinite = seed.eval();
    nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
    const auto huge =
        Eigen::VectorXd::Constant(7, std::numeric_limits<double>::max()).eval();
    if (!ScoreCandidatesCPU({Eigen::VectorXd::Zero(6)}, seed).empty() ||
        !ScoreCandidatesCPU({seed}, Eigen::VectorXd::Zero(6)).empty() ||
        !ScoreCandidatesCPU({nonfinite}, seed).empty() ||
        !ScoreCandidatesCPU({seed}, nonfinite).empty() ||
        !ScoreCandidatesCPU({huge}, -huge).empty())
        return 1;
#ifdef HOLISTICMOTION_HAS_CUDA
    std::vector<double> scores{42.0};
    const std::vector<Eigen::VectorXd> empty;
    if (ScoreCandidatesCUDA(empty, seed, scores) || !scores.empty())
        return 1;
    scores = {42.0};
    if (ScoreCandidatesCUDA({Eigen::VectorXd::Zero(6)}, seed, scores) ||
        !scores.empty())
        return 1;
    Eigen::VectorXd invalid = seed;
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    scores = {42.0};
    if (ScoreCandidatesCUDA({seed}, invalid, scores) || !scores.empty())
        return 1;
    scores = {42.0};
    if (ScoreCandidatesCUDA({invalid}, seed, scores) || !scores.empty())
        return 1;
    // Finite coordinates can still overflow their difference. A failed kernel
    // result must not leave scores that the caller could mistake for success.
    scores = {42.0};
    const auto large = Eigen::VectorXd::Constant(7, std::numeric_limits<double>::max());
    if (ScoreCandidatesCUDA({large}, -large, scores) || !scores.empty())
        return 1;
#endif
    return 0;
}
