#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "holistic_motion/manif/LieGroup.h"

using namespace holistic_motion::robotics;

template <typename Scalar> void CheckHalfTurns() {
    using Vector = Eigen::Matrix<Scalar, 3, 1>;
    using Matrix = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar pi = std::acos(Scalar(-1));
    const Scalar tolerance = 32 * std::numeric_limits<Scalar>::epsilon();
    const std::array<Vector, 3> axes{Vector::UnitZ(), Vector(1, -2, 3).normalized(),
                                     Vector(-4, 1, 2).normalized()};
    for (const auto &axis : axes) {
        for (Scalar offset : {Scalar(0), Scalar(1e-9), Scalar(1e-7), Scalar(1e-4)}) {
            for (Scalar sign : {Scalar(-1), Scalar(1)}) {
                const SO3Tangent<Scalar> tangent(axis * (pi + sign * offset));
                const Matrix left = tangent.Ljac() * tangent.Ljacinv();
                const Matrix right = tangent.Rjac() * tangent.Rjacinv();
                if (!left.allFinite() || !right.allFinite() ||
                    (left - Matrix::Identity()).norm() > tolerance ||
                    (right - Matrix::Identity()).norm() > tolerance)
                    throw std::runtime_error("half-turn Jacobian is not invertible");

                const auto rotation = tangent.Exp();
                Matrix log_jacobian;
                const auto logarithm = rotation.Log(log_jacobian);
                if (!log_jacobian.allFinite() ||
                    (log_jacobian * logarithm.Rjac() - Matrix::Identity()).norm() >
                        tolerance)
                    throw std::runtime_error("rotation Log Jacobian is inaccurate");

                for (Scalar scale : {Scalar(0.01), Scalar(1), Scalar(10000)}) {
                    const SE3<Scalar> pose(scale * Vector(2, -3, 4), rotation);
                    const auto recovered = pose.Log().Exp();
                    if (!recovered.Coeffs().allFinite() ||
                        (recovered.GetTranslation() - pose.GetTranslation()).norm() >
                            tolerance * pose.GetTranslation().norm() ||
                        (recovered.AsSO3() - rotation).Coeffs().norm() > tolerance)
                        throw std::runtime_error(
                            "half-turn SE3 round trip lost accuracy");
                }
            }
        }
    }
}

template <typename Scalar> void CheckAntipodalSmallRotations() {
    using Vector = Eigen::Matrix<Scalar, 3, 1>;
    using Matrix = Eigen::Matrix<Scalar, 3, 3>;
    const Vector axis = Vector(1, -2, 3).normalized();
    const Scalar boundary = Scalar(2) * std::sqrt(Constants<Scalar>::eps);
    const Scalar tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon();
    for (Scalar factor : {Scalar(0), Scalar(0.001), Scalar(0.5), Scalar(0.99),
                          Scalar(1.01), Scalar(2)}) {
        const SO3Tangent<Scalar> tangent(axis * (boundary * factor));
        const auto rotation = tangent.Exp();
        auto antipodal = rotation;
        antipodal.Coeffs() *= Scalar(-1);
        Matrix positive_jacobian, negative_jacobian;
        const auto positive = rotation.Log(positive_jacobian);
        const auto negative = antipodal.Log(negative_jacobian);
        if (!negative.Coeffs().allFinite() || !negative_jacobian.allFinite() ||
            (positive.Coeffs() - negative.Coeffs()).norm() >
                tolerance * std::max(Scalar(1), tangent.Coeffs().norm()) ||
            (positive_jacobian - negative_jacobian).norm() > tolerance)
            throw std::runtime_error("quaternion sign changed the rotation logarithm");
        const SE3<Scalar> pose(Vector(2, -3, 4), antipodal);
        const SE3<Scalar> equivalent(Vector(2, -3, 4), rotation);
        const auto pose_log = pose.Log();
        const auto equivalent_log = equivalent.Log();
        if (!pose_log.Coeffs().allFinite() ||
            (pose_log.Coeffs() - equivalent_log.Coeffs()).norm() >
                tolerance * std::max(Scalar(1), equivalent_log.Coeffs().norm()))
            throw std::runtime_error("quaternion sign changed the SE3 logarithm");
    }
}

template <typename Scalar> void CheckRotationVectorScales() {
    const int exponent = std::numeric_limits<Scalar>::max_exponent10 - 8;
    const Scalar large = std::pow(Scalar(10), Scalar(exponent));
    const Scalar small = Scalar(1) / large;
    const Scalar subnormal_square =
        std::sqrt(std::numeric_limits<Scalar>::min()) / Scalar(10);
    const Scalar tolerance =
        Scalar(32) * std::numeric_limits<Scalar>::epsilon();
    for (Scalar angle : {Scalar(0), Scalar(0.2), -Scalar(0.7), small, -small,
                         subnormal_square, -subnormal_square, large, -large}) {
        const SO3<Scalar> rotation(Scalar(0), Scalar(0), angle,
                                   EulerAnglesType::rota_vec);
        const Scalar expected_z = std::sin(angle / Scalar(2));
        const Scalar expected_w = std::cos(angle / Scalar(2));
        if (!rotation.Coeffs().allFinite() ||
            rotation.Coeffs()[0] != Scalar(0) ||
            rotation.Coeffs()[1] != Scalar(0) ||
            std::abs(rotation.Coeffs()[3] - expected_w) > tolerance ||
            (expected_z == Scalar(0)
                 ? rotation.Coeffs()[2] != Scalar(0)
                 : std::abs(rotation.Coeffs()[2] / expected_z - Scalar(1)) >
                       tolerance))
            throw std::runtime_error(
                "rotation vector norm lost a finite rotation");
    }
    const SO3<Scalar> tiny_rotation(small, Scalar(2) * small,
                                    -Scalar(3) * small,
                                    EulerAnglesType::rota_vec);
    const Eigen::Matrix<Scalar, 3, 1> expected(small / Scalar(2), small,
                                               -Scalar(1.5) * small);
    for (int axis = 0; axis < 3; ++axis)
        if (std::abs(tiny_rotation.Coeffs()[axis] / expected[axis] -
                     Scalar(1)) > tolerance)
            throw std::runtime_error("tiny rotation vector lost its axis");
    const Scalar subnormal_angle =
        Scalar(16) * std::numeric_limits<Scalar>::denorm_min();
    const SO3<Scalar> denormal(Scalar(0), Scalar(0), subnormal_angle,
                               EulerAnglesType::rota_vec);
    if (!denormal.Coeffs().allFinite() ||
        denormal.Coeffs()[2] != subnormal_angle / Scalar(2))
        throw std::runtime_error("subnormal rotation vector lost its phase");
}

template <typename Scalar> void CheckLargeTangents() {
    using Vector = Eigen::Matrix<Scalar, 3, 1>;
    using Matrix = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar tolerance =
        Scalar(64) * std::numeric_limits<Scalar>::epsilon();
    const std::array<Scalar, 4> angles{
        Scalar(1) / std::numeric_limits<Scalar>::epsilon(),
        std::pow(Scalar(10),
                 Scalar(std::numeric_limits<Scalar>::max_exponent10 / 3 + 4)),
        std::pow(Scalar(10),
                 Scalar(std::numeric_limits<Scalar>::max_exponent10 / 2 + 4)),
        std::pow(Scalar(10),
                 Scalar(std::numeric_limits<Scalar>::max_exponent10 - 8))};
    const auto check = [&](const Matrix &actual, const Matrix &expected) {
        if (!actual.allFinite())
            throw std::runtime_error("large SO3 Jacobian is not finite");
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                if (std::abs(actual(row, column) - expected(row, column)) >
                    tolerance * std::abs(expected(row, column)) +
                        Scalar(8) * std::numeric_limits<Scalar>::denorm_min())
                    throw std::runtime_error(
                        "large SO3 Jacobian lost an analytic entry");
    };
    for (int axis = 0; axis < 3; ++axis) {
        const Vector direction = Vector::Unit(axis);
        const int first = (axis + 1) % 3, second = (axis + 2) % 3;
        for (Scalar magnitude : angles) {
            for (Scalar sign : {Scalar(-1), Scalar(1)}) {
                const Scalar angle = sign * magnitude;
                const SO3Tangent<Scalar> tangent(direction * angle);
                Matrix exp_jacobian;
                const auto rotation = tangent.Exp(exp_jacobian);
                const Scalar sinc = std::sin(angle) / angle;
                Matrix expected = Matrix::Zero();
                expected(axis, axis) = Scalar(1);
                expected(first, first) = expected(second, second) = sinc;
                expected(first, second) =
                    -(Scalar(1) - std::cos(angle)) / angle;
                expected(second, first) = -expected(first, second);
                check(tangent.Ljac(), expected);
                check(tangent.Rjac(), expected.transpose());
                check(exp_jacobian, expected.transpose());
                const Scalar half = angle / Scalar(2);
                Matrix inverse = Matrix::Zero();
                inverse(axis, axis) = Scalar(1);
                inverse(first, first) = inverse(second, second) =
                    half * (std::cos(half) / std::sin(half));
                inverse(first, second) = half;
                inverse(second, first) = -half;
                check(tangent.Ljacinv(), inverse);
                check(tangent.Rjacinv(), inverse.transpose());
                if (!rotation.Coeffs().allFinite() ||
                    (rotation.Coeffs().template head<3>() -
                     direction * std::sin(half))
                            .norm() > tolerance ||
                    std::abs(rotation.Coeffs()[3] - std::cos(half)) > tolerance)
                    throw std::runtime_error(
                        "large SO3 exponential lost its rotation");
            }
        }
    }
}

int main() {
    try {
        CheckHalfTurns<double>();
        CheckHalfTurns<float>();
        CheckAntipodalSmallRotations<double>();
        CheckAntipodalSmallRotations<float>();
        CheckRotationVectorScales<double>();
        CheckRotationVectorScales<float>();
        CheckLargeTangents<double>();
        CheckLargeTangents<float>();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
