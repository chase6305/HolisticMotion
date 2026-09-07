#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <algorithm>

namespace holistic_motion::robotics::detail {

// Owned by one IK request. All iteration storage is sized once; no mutable
// solver-wide cache is needed for concurrent calls or model changes.
class DampedIKWorkspace {
public:
    explicit DampedIKWorkspace(Eigen::Index dof)
        : svd_(6, dof, Eigen::ComputeThinU | Eigen::ComputeThinV),
          projected_(std::min<Eigen::Index>(6, dof)) {}

    bool Solve(const Eigen::MatrixXd& jacobian,
               const Eigen::Matrix<double, 6, 1>& error, double damping,
               double threshold, Eigen::VectorXd& update) {
        svd_.compute(jacobian);
        if (svd_.info() != Eigen::Success) return false;
        projected_.noalias() = svd_.matrixU().transpose() * error;
        const auto& singular_values = svd_.singularValues();
        for (Eigen::Index i = 0; i < projected_.size(); ++i) {
            const double sigma = singular_values[i];
            const double inverse =
                sigma > threshold ? sigma / (sigma * sigma + damping * damping)
                                  : 0.0;
            projected_[i] *= inverse;
        }
        update.noalias() = svd_.matrixV() * projected_;
        return update.allFinite();
    }

private:
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_;
    Eigen::VectorXd projected_;
};

}  // namespace holistic_motion::robotics::detail
