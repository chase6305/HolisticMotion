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

int main() {
    try {
        CheckHalfTurns<double>();
        CheckHalfTurns<float>();
        CheckAntipodalSmallRotations<double>();
        CheckAntipodalSmallRotations<float>();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
