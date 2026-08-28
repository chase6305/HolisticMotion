#include "holistic_motion/kinematics/fep/FEPKinematics.h"
#include "FEPSelection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace holistic_motion::robotics {
namespace {

int Direction(double value) { return value >= 0.0 ? 1 : -1; }

double WrappedDistance(const Eigen::VectorXd &lhs, const Eigen::VectorXd &rhs) {
    double total = 0.0;
    for (Eigen::Index i = 0; i < lhs.size(); ++i) {
        total += std::pow(std::remainder(lhs[i] - rhs[i], 2.0 * M_PI), 2);
    }
    return total;
}

bool SameSolution(const Eigen::VectorXd &lhs, const Eigen::VectorXd &rhs) {
    return WrappedDistance(lhs, rhs) < 1e-12;
}

bool PoseWithinTolerance(const NumericalKinematics &solver, const SE3d &target,
                         const Eigen::VectorXd &joints, double translation_tolerance,
                         double angle_tolerance) {
    SE3d actual;
    if (!solver.GetFK(joints, actual))
        return false;
    const Eigen::Matrix4d delta =
            actual.Inverse().GetTransform() * target.GetTransform();
    const double translation_error = delta.block<3, 1>(0, 3).norm();
    const double cosine =
            clamp((delta.block<3, 3>(0, 0).trace() - 1.0) * 0.5, -1.0, 1.0);
    return std::isfinite(translation_error) &&
           translation_error <= translation_tolerance &&
           std::acos(cosine) <= angle_tolerance;
}

} // namespace

bool FEPKinematics::IsCompatible() const noexcept {
    if (joint_nodes_.size() != 8 || GetDOF() != 7)
        return false;
    return std::all_of(joint_nodes_.begin(), joint_nodes_.begin() + 7,
                       [](const JointNode &node) {
                           return node.joint_type == JointType::REVOLUTE ||
                                  node.joint_type == JointType::CONTINUOUS;
                       });
}

bool FEPKinematics::HasCudaBackend() noexcept {
#ifdef HOLISTICMOTION_HAS_CUDA
    return fep_internal::CudaBackendAvailable();
#else
    return false;
#endif
}

bool FEPKinematics::IsCudaCompiled() noexcept {
#ifdef HOLISTICMOTION_HAS_CUDA
    return true;
#else
    return false;
#endif
}

FEPConfiguration FEPKinematics::GetConfiguration(const Eigen::VectorXd &joints) const {
    if (joints.size() != 7 || !joints.allFinite())
        return {};
    return {Direction(joints[1]), Direction(joints[3]), Direction(joints[5]),
            joints[6]};
}

bool FEPKinematics::SolveSeed(const SE3d &target, Eigen::VectorXd seed,
                              Eigen::VectorXd &solution) const {
    if (!IsCompatible() || seed.size() != 7 || !seed.allFinite() ||
        !target.GetTransform().allFinite())
        return false;
    for (Eigen::Index i = 0; i < seed.size(); ++i) {
        seed[i] = clamp(seed[i], joint_nodes_[i].lower_limit,
                        joint_nodes_[i].upper_limit);
    }
    IkRtn result;
    std::vector<double> distances;
    if (!NumericalKinematics::GetIK(target, result, seed, distances) ||
        result.ik_joints.empty()) {
        return false;
    }
    solution = result.ik_joints.front();
    if (solution.size() != 7 || !solution.allFinite())
        return false;

    // The shared interactive solver intentionally accepts sub-millimetre
    // residuals. FEP branch enumeration needs tighter FK round trips so a
    // second local solve uses stricter tolerances only when required.
    if (!PoseWithinTolerance(*this, target, solution, 1e-5, 1e-5)) {
        NumericalKinematics refinement(joint_nodes_);
        refinement.SetMaxIterNum(200);
        refinement.SetTransErrTh(1e-7);
        refinement.SetAngleErrTh(1e-7);
        refinement.SetDamp(0.01);
        IkRtn refined;
        std::vector<double> refined_distances;
        Eigen::VectorXd refined_seed = solution;
        if (refinement.GetIK(target, refined, refined_seed, refined_distances) &&
            !refined.ik_joints.empty()) {
            solution = refined.ik_joints.front();
        }
    }
    return solution.size() == 7 && solution.allFinite() &&
           PoseWithinTolerance(*this, target, solution, 1e-5, 1e-5);
}

bool FEPKinematics::SolveConfiguration(const SE3d &target,
                                       const FEPConfiguration &configuration,
                                       const Eigen::VectorXd &seed,
                                       Eigen::VectorXd &solution) const {
    if (!IsCompatible() || !std::isfinite(configuration.redundancy) ||
        !target.GetTransform().allFinite())
        return false;
    Eigen::VectorXd branch_seed = seed;
    if (branch_seed.size() != 7 || !branch_seed.allFinite())
        return false;
    const std::array<int, 3> indices{1, 3, 5};
    const std::array<int, 3> directions{Direction(configuration.shoulder),
                                        Direction(configuration.elbow),
                                        Direction(configuration.wrist)};
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const int index = indices[i];
        const double magnitude = std::max(std::abs(branch_seed[index]), 0.35);
        branch_seed[index] = directions[i] * magnitude;
    }
    branch_seed[6] = configuration.redundancy;
    if (!SolveSeed(target, branch_seed, solution))
        return false;
    const auto solved = GetConfiguration(solution);
    return solved.shoulder == directions[0] && solved.elbow == directions[1] &&
           solved.wrist == directions[2];
}

bool FEPKinematics::Solve(const SE3d &target, const Eigen::VectorXd &seed,
                          FEPSolveMethod method,
                          std::vector<Eigen::VectorXd> &solutions) const {
    solutions.clear();
    if (!IsCompatible() || seed.size() != 7 || !seed.allFinite())
        return false;
    auto append = [&](const Eigen::VectorXd &candidate) {
        if (std::none_of(solutions.begin(), solutions.end(),
                         [&](const Eigen::VectorXd &existing) {
                             return SameSolution(existing, candidate);
                         })) {
            solutions.push_back(candidate);
        }
    };
    if (method == FEPSolveMethod::SEEDED_NUMERICAL ||
        method == FEPSolveMethod::COMPATIBILITY) {
        Eigen::VectorXd candidate;
        if (SolveSeed(target, seed, candidate))
            append(candidate);
    } else if (method == FEPSolveMethod::CONFIGURATION) {
        Eigen::VectorXd candidate;
        if (SolveConfiguration(target, GetConfiguration(seed), seed, candidate))
            append(candidate);
    } else if (method == FEPSolveMethod::ALL_CONFIGURATIONS) {
        for (int shoulder : {-1, 1})
            for (int elbow : {-1, 1})
                for (int wrist : {-1, 1}) {
                    Eigen::VectorXd candidate;
                    const FEPConfiguration configuration{shoulder, elbow, wrist,
                                                         seed[6]};
                    if (SolveConfiguration(target, configuration, seed, candidate))
                        append(candidate);
                }
    } else {
        Eigen::VectorXd best;
        double best_cost = std::numeric_limits<double>::infinity();
        const double lower = joint_nodes_[6].lower_limit;
        const double upper = joint_nodes_[6].upper_limit;
        double previous_low = std::numeric_limits<double>::quiet_NaN();
        double previous_high = std::numeric_limits<double>::quiet_NaN();
        for (double delta = 0.0; delta <= M_PI + 1e-12; delta += M_PI / 36.0) {
            const std::array<double, 2> signs{-1.0, 1.0};
            const int sign_count = delta == 0.0 ? 1 : 2;
            for (int sign_index = 0; sign_index < sign_count; ++sign_index) {
                const double sign = delta == 0.0 ? 0.0 : signs[sign_index];
                Eigen::VectorXd trial = seed;
                trial[6] = clamp(seed[6] + sign * delta, lower, upper);
                double &previous = sign <= 0.0 ? previous_low : previous_high;
                if (trial[6] == previous)
                    continue;
                previous = trial[6];
                Eigen::VectorXd candidate;
                if (!SolveSeed(target, trial, candidate))
                    continue;
                const double cost = WrappedDistance(candidate, seed);
                if (cost < best_cost) {
                    best_cost = cost;
                    best = candidate;
                }
            }
            if (best.size() == 7)
                break;
        }
        if (best.size() == 7)
            append(best);
    }
    std::vector<double> scores;
#ifdef HOLISTICMOTION_HAS_CUDA
    // Transfers dominate small interactive calls; reserve CUDA for batches.
    if (solutions.size() >= 32)
        fep_internal::ScoreCandidatesCUDA(solutions, seed, scores);
#endif
    if (scores.size() != solutions.size())
        scores = fep_internal::ScoreCandidatesCPU(solutions, seed);
    std::vector<std::size_t> order(solutions.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return scores[lhs] < scores[rhs];
    });
    std::vector<Eigen::VectorXd> sorted;
    sorted.reserve(solutions.size());
    for (const auto index : order)
        sorted.push_back(std::move(solutions[index]));
    solutions = std::move(sorted);
    return !solutions.empty();
}

} // namespace holistic_motion::robotics
