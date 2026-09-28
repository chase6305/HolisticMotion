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

int main() {
    try {
        CheckHalfTurns<double>();
        CheckHalfTurns<float>();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
