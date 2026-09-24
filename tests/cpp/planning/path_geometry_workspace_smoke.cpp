#ifdef NDEBUG
#undef NDEBUG
#endif
#define EIGEN_RUNTIME_NO_MALLOC

#include <cmath>
#include <iostream>
#include <utility>

#include "PathGeometryWorkspace.h"

using holistic_motion::robotics::planning::detail::PathGeometryWorkspace;

namespace {
constexpr double kPi = 3.14159265358979323846;

double Objective(const std::vector<Eigen::VectorXd> &path,
                 const Eigen::VectorXd &weights, bool continuous,
                 double length_weight, double smoothness_weight) {
    std::vector<Eigen::VectorXd> differences;
    for (std::size_t i = 1; i < path.size(); ++i) {
        Eigen::VectorXd difference = path[i] - path[i - 1];
        if (continuous)
            difference[0] = std::remainder(difference[0], 2.0 * kPi);
        differences.push_back(difference);
    }
    double length = 0.0;
    double smoothness = 0.0;
    for (std::size_t i = 0; i < differences.size(); ++i) {
        length += std::sqrt(differences[i].dot(
            (weights.array() * differences[i].array()).matrix()));
        if (i > 0) {
            const Eigen::VectorXd acceleration =
                differences[i] - differences[i - 1];
            smoothness += acceleration.dot(
                (weights.array() * acceleration.array()).matrix());
        }
    }
    return length_weight * length + smoothness_weight * smoothness;
}
} // namespace

int main() {
    // Large but finite weights must not turn zero or small smoothness
    // derivatives into NaN through a prematurely overflowing coefficient.
    const Eigen::VectorXd large = Eigen::VectorXd::Constant(1, 1e308);
    const Eigen::VectorXd unit = Eigen::VectorXd::Ones(1);
    const std::vector<bool> bounded{false};
    std::vector<Eigen::VectorXd> small_path;
    for (double q : {-0.02, 0.04, -0.01, 0.015, -0.025})
        small_path.push_back(Eigen::VectorXd::Constant(1, q));
    PathGeometryWorkspace large_workspace(large, bounded, 0.0, 1.0);
    Eigen::VectorXd large_gradient(1);
    for (std::size_t i = 1; i + 1 < small_path.size(); ++i) {
        Eigen::internal::set_is_malloc_allowed(false);
        large_workspace.SetWaypoint(small_path, i);
        large_workspace.Gradient(large_gradient);
        Eigen::internal::set_is_malloc_allowed(true);
        const double original = small_path[i][0];
        constexpr double step = 1e-7;
        small_path[i][0] = original + step;
        const double positive = Objective(small_path, unit, false, 0.0, 1.0);
        small_path[i][0] = original - step;
        const double negative = Objective(small_path, unit, false, 0.0, 1.0);
        small_path[i][0] = original;
        if (!large_gradient.allFinite() ||
            std::abs(large_gradient[0] / large[0] -
                     (positive - negative) / (2.0 * step)) > 1e-10) {
            std::cerr
                << "large-weight geometry gradient is not representable\n";
            return 1;
        }
    }
    // A disabled term must not participate in the objective, even if its
    // unweighted value would overflow. Exercise local and line-search costs.
    const std::vector<Eigen::VectorXd> zigzag{
        Eigen::VectorXd::Constant(1, -0.5), Eigen::VectorXd::Constant(1, 0.5),
        Eigen::VectorXd::Constant(1, -0.5)};
    for (double length_weight : {0.0, 1.0}) {
        PathGeometryWorkspace disabled(large, bounded, length_weight, 0.0);
        disabled.SetWaypoint(zigzag, 1);
        disabled.Gradient(large_gradient);
        const double initial = disabled.LocalObjective();
        const Eigen::VectorXd trial = Eigen::VectorXd::Constant(1, 0.4);
        const double candidate =
            disabled.CandidateObjective(zigzag.front(), trial, zigzag.back());
        if (!large_gradient.allFinite() ||
            std::abs(initial / 1e154 - length_weight * 2.0) > 1e-14 ||
            std::abs(candidate / 1e154 - length_weight * 1.8) > 1e-14 ||
            !std::isfinite(initial) || !std::isfinite(candidate)) {
            std::cerr << "disabled geometry term contaminated the objective\n";
            return 1;
        }
    }
    for (int dof : {2, 7, 14}) {
        for (bool continuous : {false, true}) {
            const Eigen::VectorXd weights =
                Eigen::VectorXd::LinSpaced(dof, 0.5, 2.0);
            std::vector<bool> topology(dof, false);
            topology[0] = continuous;
            for (const auto &terms : {std::pair<double, double>{1.0, 0.0},
                                      {0.0, 0.7},
                                      {0.8, 1.3},
                                      {0.0, 0.0}}) {
                std::vector<Eigen::VectorXd> path;
                for (int i = 0; i < 7; ++i) {
                    Eigen::VectorXd state(dof);
                    for (int joint = 0; joint < dof; ++joint)
                        state[joint] =
                            i * 0.13 +
                            0.2 * std::sin((i + 1) * (joint + 1) * 0.7);
                    if (continuous)
                        state[0] = std::remainder(3.0 + state[0], 2.0 * kPi);
                    path.push_back(state);
                }
                PathGeometryWorkspace workspace(weights, topology, terms.first,
                                                terms.second);
                Eigen::VectorXd gradient(dof);
                Eigen::VectorXd candidate(dof);
                // Switch sweep direction and revisit previously changed
                // neighbors.
                for (std::size_t index : {1, 2, 3, 4, 5, 5, 4, 3, 2, 1}) {
                    const double before = Objective(path, weights, continuous,
                                                    terms.first, terms.second);
                    Eigen::internal::set_is_malloc_allowed(false);
                    workspace.SetWaypoint(path, index);
                    workspace.Gradient(gradient);
                    const double local = workspace.LocalObjective();
                    Eigen::internal::set_is_malloc_allowed(true);
                    constexpr double epsilon = 1e-6;
                    for (int joint = 0; joint < dof; ++joint) {
                        const double original = path[index][joint];
                        path[index][joint] = original + epsilon;
                        const double positive =
                            Objective(path, weights, continuous, terms.first,
                                      terms.second);
                        path[index][joint] = original - epsilon;
                        const double negative =
                            Objective(path, weights, continuous, terms.first,
                                      terms.second);
                        path[index][joint] = original;
                        const double reference =
                            (positive - negative) / (2 * epsilon);
                        if (std::abs(reference - gradient[joint]) > 2e-7) {
                            std::cerr << "geometry gradient differs from "
                                         "finite differences\n";
                            return 1;
                        }
                    }
                    // Multiple trials must reuse only the unchanged outer
                    // edges.
                    for (double step : {0.02, -0.01}) {
                        candidate = path[index].array() + step;
                        Eigen::internal::set_is_malloc_allowed(false);
                        const double trial = workspace.CandidateObjective(
                            path[index - 1], candidate, path[index + 1]);
                        Eigen::internal::set_is_malloc_allowed(true);
                        const Eigen::VectorXd original = path[index];
                        path[index] = candidate;
                        const double after =
                            Objective(path, weights, continuous, terms.first,
                                      terms.second);
                        if (std::abs((after - before) - (trial - local)) >
                            1e-11) {
                            std::cerr << "local objective delta differs from "
                                         "full objective\n";
                            return 1;
                        }
                        path[index] = original;
                    }
                    path[index] = candidate;
                }
                for (auto &state : path)
                    state.setZero();
                Eigen::internal::set_is_malloc_allowed(false);
                workspace.SetWaypoint(path, 3);
                workspace.Gradient(gradient);
                const double zero_objective = workspace.LocalObjective();
                Eigen::internal::set_is_malloc_allowed(true);
                if (zero_objective != 0.0 || !gradient.isZero())
                    return 1;
            }
        }
    }
}
