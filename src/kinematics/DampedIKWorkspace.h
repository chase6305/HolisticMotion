#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace holistic_motion::robotics::detail {

// Owned by one IK request. All iteration storage is sized once; no mutable
// solver-wide cache is needed for concurrent calls or model changes.
class DampedIKWorkspace {
public:
    explicit DampedIKWorkspace(Eigen::Index dof)
        : svd_(6, dof, Eigen::ComputeThinU | Eigen::ComputeThinV),
          projected_(std::min<Eigen::Index>(6, dof)),
          scaled_jacobian_(6, dof) {}

    bool Solve(const Eigen::MatrixXd &jacobian,
               const Eigen::Matrix<double, 6, 1> &error, double damping,
               double threshold, Eigen::VectorXd &update) {
        if (!std::isfinite(damping) || !std::isfinite(threshold) ||
            threshold < 0.0 || !error.allFinite())
            return false;
        svd_.compute(jacobian);
        if (svd_.info() != Eigen::Success) return false;
        const auto &singular_values = svd_.singularValues();
        double jacobian_scale = 1.0;
        if (!singular_values.allFinite() ||
            (singular_values[0] > 0.0 && !std::isnormal(singular_values[0]))) {
            jacobian_scale = jacobian.cwiseAbs().maxCoeff();
            if (!(jacobian_scale > 0.0) || !std::isfinite(jacobian_scale))
                return false;
            scaled_jacobian_ = jacobian / jacobian_scale;
            svd_.compute(scaled_jacobian_);
            if (svd_.info() != Eigen::Success || !singular_values.allFinite())
                return false;
        }
        projected_.noalias() = svd_.matrixU().transpose() * error;
        double error_scale = 1.0;
        const double maximum_error = error.cwiseAbs().maxCoeff();
        if (!projected_.allFinite() ||
            (maximum_error > 0.0 && !std::isnormal(maximum_error))) {
            if (!(maximum_error > 0.0)) return false;
            error_scale = maximum_error;
            const Eigen::Matrix<double, 6, 1> scaled_error =
                error / error_scale;
            projected_.noalias() = svd_.matrixU().transpose() * scaled_error;
        }
        const double scaled_threshold = threshold / jacobian_scale;
        for (Eigen::Index i = 0; i < projected_.size(); ++i) {
            const double sigma = singular_values[i];
            if (sigma <= scaled_threshold) {
                projected_[i] = 0.0;
                continue;
            }
            const double denominator = sigma * sigma + damping * damping;
            const double gain =
                std::isnormal(denominator) ? sigma / denominator : 0.0;
            if (jacobian_scale == 1.0 && error_scale == 1.0 &&
                std::isnormal(denominator) && std::isnormal(gain)) {
                projected_[i] *= gain;
            } else {
                if (!std::isfinite(projected_[i])) return false;
                projected_[i] = ScaledProjection(
                    projected_[i], error_scale, sigma, jacobian_scale, damping);
            }
        }
        update.noalias() = svd_.matrixV() * projected_;
        return update.allFinite();
    }

private:
    // Evaluate error * sigma / (sigma^2 + damping^2), including the input
    // scales, without materializing an unrepresentable singular value, error,
    // square, or inverse. The ordinary path keeps its original rounding.
    static double ScaledProjection(double error, double error_scale,
                                   double sigma, double jacobian_scale,
                                   double damping) {
        int error_exponent, error_scale_exponent, sigma_exponent,
            jacobian_exponent, product_exponent, damping_exponent;
        const double error_mantissa = std::frexp(error, &error_exponent);
        const double error_scale_mantissa =
            std::frexp(error_scale, &error_scale_exponent);
        const double sigma_mantissa =
            std::frexp(std::frexp(sigma, &sigma_exponent) *
                           std::frexp(jacobian_scale, &jacobian_exponent),
                       &product_exponent);
        sigma_exponent += jacobian_exponent + product_exponent;
        const double damping_mantissa =
            std::frexp(std::abs(damping), &damping_exponent);
        const bool sigma_dominates = damping == 0.0 ||
                                     sigma_exponent > damping_exponent ||
                                     (sigma_exponent == damping_exponent &&
                                      sigma_mantissa >= damping_mantissa);
        const double dominant_mantissa =
            sigma_dominates ? sigma_mantissa : damping_mantissa;
        const int dominant_exponent =
            sigma_dominates ? sigma_exponent : damping_exponent;
        const double ratio =
            sigma_dominates ? std::scalbn(damping_mantissa / dominant_mantissa,
                                          damping_exponent - dominant_exponent)
                            : std::scalbn(sigma_mantissa / dominant_mantissa,
                                          sigma_exponent - dominant_exponent);
        const double mantissa =
            error_mantissa * error_scale_mantissa * sigma_mantissa /
            (dominant_mantissa * dominant_mantissa * (1.0 + ratio * ratio));
        return std::scalbn(mantissa, error_exponent + error_scale_exponent +
                                         sigma_exponent -
                                         2 * dominant_exponent);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_;
    Eigen::VectorXd projected_;
    Eigen::MatrixXd scaled_jacobian_;
};

}  // namespace holistic_motion::robotics::detail
