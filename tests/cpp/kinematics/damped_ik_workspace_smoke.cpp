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
