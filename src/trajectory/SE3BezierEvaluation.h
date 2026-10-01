#pragma once

#include "holistic_motion/trajectory/Types.h"

#include <array>
#include <cmath>

namespace holistic_motion::robotics::detail {

// Right-trivialized derivatives of Q0 * product Exp(beta_i(s) * xi_i).
// Acceleration and jerk differentiate the body-twist coordinate vector.
template <int Degree> class SE3BezierEvaluation {
    using Vector = Eigen::Matrix<double, 6, 1>;

public:
    SE3BezierEvaluation(const std::vector<SE3d> &points, double parameter,
                        double length)
        : origin_(points[0]), s_(parameter), length_(length) {
        for (int i = 0; i < Degree; ++i) {
            increments_[i] = points[i + 1] - points[i];
            translation_only_ = translation_only_ &&
                                increments_[i].Coeffs().template tail<3>().isZero(0.0);
        }
    }

    void SetParameter(double parameter) { s_ = parameter; }

    SE3d Position() const {
        SE3d position = origin_;
        Evaluate<0, true>(&position);
        return position;
    }

    SE3Tangentd FirstDerivative() const { return SE3Tangentd(Evaluate<1>()[0]); }
    SE3Tangentd SecondDerivative() const { return SE3Tangentd(Evaluate<2>()[1]); }
    SE3Tangentd ThirdDerivative() const { return SE3Tangentd(Evaluate<3>()[2]); }

    void ComputeDerivatives(SE3Tangentd &tangent, SE3Tangentd &curvature,
                            SE3Tangentd &torsion) const {
        const auto jet = Evaluate<3>();
        tangent = SE3Tangentd(jet[0]);
        curvature = SE3Tangentd(jet[1]);
        torsion = SE3Tangentd(jet[2]);
    }

    void ComputeJet(SE3d &position, SE3Tangentd &tangent, SE3Tangentd &curvature,
                    SE3Tangentd &torsion) const {
        position = origin_;
        const auto jet = Evaluate<3, true>(&position);
        tangent = SE3Tangentd(jet[0]);
        curvature = SE3Tangentd(jet[1]);
        torsion = SE3Tangentd(jet[2]);
    }

private:
    static Vector Bracket(const Vector &a, const Vector &b) {
        Vector result;
        result.head<3>() =
            a.tail<3>().cross(b.head<3>()) + a.head<3>().cross(b.tail<3>());
        result.tail<3>() = a.tail<3>().cross(b.tail<3>());
        return result;
    }

    static void DerivativeTransform(const SE3Tangentd &scaled,
                                    Eigen::Matrix3d &rotation,
                                    Eigen::Vector3d &translation) {
        const Eigen::Vector3d angular = scaled.Coeffs().tail<3>();
        const Eigen::Vector3d linear = scaled.Coeffs().head<3>();
        rotation = SO3Tangentd(angular).Exp().GetRotation().transpose();
        // Apply the SO3 left Jacobian directly to the linear component. Its
        // skew products are cross products; no 3x3 Jacobian is needed here.
        // Keep the exponential's small-angle branch and rotation unchanged.
        const double theta_squared = angular.squaredNorm();
        const Eigen::Vector3d cross = angular.cross(linear);
        if (theta_squared <= Constants<double>::eps) {
            translation = linear + 0.5 * cross;
        } else {
            const double theta = std::sqrt(theta_squared);
            translation = linear + ((1.0 - std::cos(theta)) / theta_squared) * cross +
                          ((theta - std::sin(theta)) / (theta_squared * theta)) *
                              angular.cross(cross);
        }
        // Cross products may overflow before their bounded coefficients apply.
        // Preserve the exponential's finite result at those extreme scales.
        if (!translation.allFinite())
            translation = SO3Tangentd(angular).Ljac() * linear;
    }

    template <int Order, bool WithPosition = false>
    std::array<Vector, Order> Evaluate(SE3d *position = nullptr) const {
        std::array<double, Degree> beta, first, second, third;
        const double s = s_, u = 1 - s;
        if constexpr (Degree == 2) {
            beta = {s * (2 - s), s * s};
            first = {2 * u, 2 * s};
            second = {-2, 2};
            third = {0, 0};
        } else {
            const double s2 = s * s, s3 = s * s2, s4 = s2 * s2, s5 = s2 * s3;
            const double u2 = u * u;
            beta = {(5 - 10 * s + 10 * s2 - 5 * s3 + s4) * s,
                    (10 - 20 * s + 15 * s2 - 4 * s3) * s2, (10 - 15 * s + 6 * s2) * s3,
                    (5 - 4 * s) * s4, s5};
            first = {5 * u2 * u2, 20 * s * u2 * u, 30 * s2 * u2, 20 * s3 * u, 5 * s4};
            second = {-20 * u2 * u, 20 * u2 * u - 60 * s * u2,
                      60 * s * u2 - 60 * s2 * u, 60 * s2 * u - 20 * s3, 20 * s3};
            third = {60 * u2, 120 * s * u - 120 * u2, 60 * s2 - 240 * s * u + 60 * u2,
                     120 * s * u - 120 * s2, 60 * s2};
        }
        std::array<Vector, Order> jet;
        for (auto &derivative : jet)
            derivative.setZero();
        if constexpr (Order > 0) {
            if (translation_only_) {
                if constexpr (WithPosition)
                    *position = Position();
                for (int i = 0; i < Degree; ++i)
                    jet[0] += first[i] * increments_[i].Coeffs();
                // Difference before multiplying the basis. Exactly uniform
                // translations must retain zero curvature and torsion even
                // after division by a short segment's squared/cubed length.
                if constexpr (Order > 1) {
                    if constexpr (Degree == 2) {
                        jet[1] = 2 * (increments_[1] - increments_[0]).Coeffs();
                    } else {
                        const Vector d0 = (increments_[1] - increments_[0]).Coeffs();
                        const Vector d1 = (increments_[2] - increments_[1]).Coeffs();
                        const Vector d2 = (increments_[3] - increments_[2]).Coeffs();
                        const Vector d3 = (increments_[4] - increments_[3]).Coeffs();
                        jet[1] = 20 * u * u * u * d0 + 60 * s * u * u * d1 +
                                 60 * s * s * u * d2 + 20 * s * s * s * d3;
                        if constexpr (Order > 2)
                            jet[2] = 60 * u * u * (d1 - d0) + 120 * s * u * (d2 - d1) +
                                     60 * s * s * (d3 - d2);
                    }
                }
                return Normalize(jet);
            }
        }
        for (int i = 0; i < Degree; ++i) {
            if constexpr (Order > 0) {
                // The first factor has no preceding twist to transport.
                if (i == 0) {
                    if constexpr (WithPosition)
                        *position = position->Compose((beta[0] * increments_[0]).Exp());
                    jet[0] = first[0] * increments_[0].Coeffs();
                    if constexpr (Order > 1)
                        jet[1] = second[0] * increments_[0].Coeffs();
                    if constexpr (Order > 2)
                        jet[2] = third[0] * increments_[0].Coeffs();
                    continue;
                }
            }
            Eigen::Matrix3d rotation;
            Eigen::Vector3d translation;
            if constexpr (WithPosition) {
                const auto factor = (beta[i] * increments_[i]).Exp();
                *position = position->Compose(factor);
                if constexpr (Order > 0) {
                    rotation = factor.GetRotation().transpose();
                    translation = factor.GetTranslation();
                }
            } else {
                DerivativeTransform(beta[i] * increments_[i], rotation, translation);
            }
            if constexpr (Order > 0) {
                const auto transport = [&](const Vector &value) -> Vector {
                    Vector result;
                    result.head<3>() = rotation * (value.head<3>() -
                                                   translation.cross(value.tail<3>()));
                    result.tail<3>() = rotation * value.tail<3>();
                    return result;
                };
                const Vector velocity = transport(jet[0]);
                const Vector &increment = increments_[i].Coeffs();
                if constexpr (Order > 1) {
                    const Vector acceleration = transport(jet[1]);
                    // Every derivative of this factor is a scalar multiple of
                    // the same twist. Reuse its bracket with the transported
                    // velocity instead of evaluating that bracket twice.
                    const Vector commutator = Bracket(increment, velocity);
                    if constexpr (Order > 2) {
                        jet[2] = transport(jet[2]) -
                                 2 * first[i] * Bracket(increment, acceleration) +
                                 first[i] * first[i] * Bracket(increment, commutator) -
                                 second[i] * commutator + third[i] * increment;
                    }
                    jet[1] =
                        acceleration - first[i] * commutator + second[i] * increment;
                }
                jet[0] = velocity + first[i] * increment;
            }
        }
        return Normalize(jet);
    }

    template <std::size_t Order>
    std::array<Vector, Order> Normalize(std::array<Vector, Order> jet) const {
        // Divide successively so finite derivatives survive overflowing L^2/L^3.
        for (std::size_t order = 0; order < Order; ++order)
            for (std::size_t n = 0; n <= order; ++n)
                jet[order] /= length_;
        return jet;
    }

    const SE3d &origin_;
    std::array<SE3Tangentd, Degree> increments_;
    double s_, length_;
    bool translation_only_{true};
};
} // namespace holistic_motion::robotics::detail
