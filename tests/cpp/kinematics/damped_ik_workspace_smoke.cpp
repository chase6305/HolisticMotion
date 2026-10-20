#ifdef NDEBUG
#undef NDEBUG
#endif
#define EIGEN_RUNTIME_NO_MALLOC

#include <Eigen/Cholesky>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "DampedIKWorkspace.h"

using holistic_motion::robotics::detail::DampedIKWorkspace;

int main() {
    std::mt19937 generator(17);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (int dof : {1, 3, 6, 7, 14}) {
        DampedIKWorkspace workspace(dof);
        Eigen::MatrixXd jacobian(6, dof);
        Eigen::VectorXd update(dof);
        Eigen::Matrix<double, 6, 1> error;
        for (int sample = 0; sample < 32; ++sample) {
            for (Eigen::Index row = 0; row < 6; ++row) {
                error[row] = uniform(generator);
                for (Eigen::Index col = 0; col < dof; ++col)
                    jacobian(row, col) = uniform(generator);
            }
            if (sample % 2 == 0) jacobian.bottomRows(3).setZero();
            constexpr double damping = 0.2;
            const Eigen::Matrix<double, 6, 6> normal =
                jacobian * jacobian.transpose() +
                damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
            const Eigen::VectorXd reference =
                jacobian.transpose() * normal.ldlt().solve(error);
            Eigen::internal::set_is_malloc_allowed(false);
            const bool success =
                workspace.Solve(jacobian, error, damping, 1e-4, update);
            Eigen::internal::set_is_malloc_allowed(true);
            if (!success || (update - reference).norm() > 1e-10) {
                std::cerr << "damped SVD differs from the regularized normal "
                             "equations\n";
                return 1;
            }
        }
    }

    // Regularized inversion must preserve representable updates even when
    // squaring the singular value or damping overflows or underflows.
    for (double scale : {1e-320, 1e-200, 1e-160, 1e160, 1e200, 1e308}) {
        DampedIKWorkspace scaled_workspace(1);
        Eigen::MatrixXd scaled_jacobian = Eigen::MatrixXd::Zero(6, 1);
        scaled_jacobian(0, 0) = scale;
        Eigen::Matrix<double, 6, 1> scaled_error =
            Eigen::Matrix<double, 6, 1>::Zero();
        scaled_error[0] = scale;
        Eigen::VectorXd scaled_update(1);
        Eigen::internal::set_is_malloc_allowed(false);
        const bool scaled_success = scaled_workspace.Solve(
            scaled_jacobian, scaled_error, scale, 0.0, scaled_update);
        Eigen::internal::set_is_malloc_allowed(true);
        if (!scaled_success || std::abs(scaled_update[0] - 0.5) > 1e-12) {
            std::cerr << "damped inversion lost a finite scaled update at "
                      << scale << '\n';
            return 1;
        }
    }

    // The gain alone may underflow although multiplying by a large error
    // would produce a representable update. Preserve subnormal gains too.
    for (double sigma : {1e-300, 1e-120}) {
        DampedIKWorkspace scaled_workspace(1);
        Eigen::MatrixXd scaled_jacobian = Eigen::MatrixXd::Zero(6, 1);
        scaled_jacobian(0, 0) = sigma;
        Eigen::Matrix<double, 6, 1> scaled_error =
            Eigen::Matrix<double, 6, 1>::Zero();
        scaled_error[0] = -1e300;
        Eigen::VectorXd scaled_update(1);
        const long double reference =
            static_cast<long double>(scaled_error[0]) * sigma /
            (static_cast<long double>(sigma) * sigma + 1e200L);
        Eigen::internal::set_is_malloc_allowed(false);
        const bool scaled_success = scaled_workspace.Solve(
            scaled_jacobian, scaled_error, 1e100, 0.0, scaled_update);
        Eigen::internal::set_is_malloc_allowed(true);
        if (!scaled_success ||
            std::abs((scaled_update[0] - reference) / reference) > 1e-12L) {
            std::cerr << "damped inversion lost an underflowing gain\n";
            return 1;
        }
    }

    // A rectangular Jacobian can have an unrepresentable singular value or
    // projected error even when every input and the regularized update fit.
    for (double scale : {1e-320, 1e308}) {
        DampedIKWorkspace scaled_workspace(1);
        const Eigen::MatrixXd scaled_jacobian =
            Eigen::MatrixXd::Constant(6, 1, scale);
        const Eigen::Matrix<double, 6, 1> scaled_error =
            Eigen::Matrix<double, 6, 1>::Constant(scale);
        Eigen::VectorXd scaled_update(1);
        Eigen::internal::set_is_malloc_allowed(false);
        const bool scaled_success = scaled_workspace.Solve(
            scaled_jacobian, scaled_error, scale, 0.0, scaled_update);
        Eigen::internal::set_is_malloc_allowed(true);
        if (!scaled_success || std::abs(scaled_update[0] - 6.0 / 7.0) > 1e-12) {
            std::cerr << "rectangular damped inversion lost its scaled update\n";
            return 1;
        }
    }

    for (int dof : {1, 3, 6, 7, 14}) {
        const Eigen::MatrixXd base_jacobian = Eigen::MatrixXd::Random(6, dof);
        const Eigen::Matrix<double, 6, 1> base_error =
            Eigen::Matrix<double, 6, 1>::Random();
        DampedIKWorkspace scaled_workspace(dof);
        Eigen::VectorXd scaled_update(dof);
        for (double scale : {1e-320, 1e308, 1.0}) {
            const Eigen::MatrixXd scaled_jacobian = base_jacobian * scale;
            const Eigen::Matrix<double, 6, 1> scaled_error = base_error * scale;
            const double scaled_damping = 0.2 * scale;
            // Normalize the represented inputs, including subnormal rounding,
            // before forming independent regularized normal equations.
            const Eigen::MatrixXd normalized = scaled_jacobian / scale;
            const double damping = scaled_damping / scale;
            const Eigen::Matrix<double, 6, 6> normal =
                normalized * normalized.transpose() +
                damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
            const Eigen::VectorXd reference =
                normalized.transpose() *
                normal.ldlt().solve((scaled_error / scale).eval());
            Eigen::internal::set_is_malloc_allowed(false);
            const bool scaled_success =
                scaled_workspace.Solve(scaled_jacobian, scaled_error,
                                       scaled_damping, 0.0, scaled_update);
            Eigen::internal::set_is_malloc_allowed(true);
            if (!scaled_success || (scaled_update - reference).norm() > 1e-10) {
                std::cerr << "scaled workspace differs from normalized normal "
                             "equations or retained a previous input scale\n";
                return 1;
            }
        }
    }

    DampedIKWorkspace workspace(7);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, 7);
    jacobian(0, 0) = 5e-5;
    jacobian(1, 1) = 2e-4;
    const Eigen::Matrix<double, 6, 1> error =
        Eigen::Matrix<double, 6, 1>::Ones();
    Eigen::VectorXd update(7);
    Eigen::internal::set_is_malloc_allowed(false);
    const bool success = workspace.Solve(jacobian, error, 0.05, 1e-4, update);
    Eigen::internal::set_is_malloc_allowed(true);
    if (!success || update[0] != 0.0 || update.tail(5).norm() != 0.0 ||
        std::abs(update[1] - 2e-4 / (4e-8 + 0.0025)) > 1e-12) {
        std::cerr << "singular-value cutoff changed\n";
        return 1;
    }
    jacobian(0, 0) = std::numeric_limits<double>::quiet_NaN();
    if (workspace.Solve(jacobian, error, 0.05, 1e-4, update)) {
        std::cerr << "invalid decomposition was accepted\n";
        return 1;
    }
    return 0;
}
