#include "holistic_motion/manif/LieGroup.h"
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <unsupported/Eigen/AutoDiff>
using AD = Eigen::AutoDiffScalar<Eigen::Vector3d>;
namespace holistic_motion::robotics {
template <> struct Constants<AD> {
    inline static const AD eps = AD(1e-12);
};
} // namespace holistic_motion::robotics
using namespace holistic_motion::robotics;
int main() {
    for (const auto &value :
         {Eigen::Vector3d(.2, -.1, .4), Eigen::Vector3d(2., -1., .5),
          Eigen::Vector3d(1e-8, -2e-8, 3e-8)}) {
        Eigen::Matrix<AD, 3, 1> variables;
        for (int axis = 0; axis < 3; ++axis) {
            variables[axis] = AD(value[axis]);
            variables[axis].derivatives() = Eigen::Vector3d::Unit(axis);
        }
        const SO3Tangent<AD> tangent(variables);
        Eigen::Matrix<AD, 3, 3> right;
        const auto rotation = tangent.Exp(right);
        const auto left = tangent.Ljac();
        const auto inverse = tangent.Ljacinv();
        for (int axis = 0; axis < 3; ++axis) {
            auto high = value, low = value;
            const double step = value.norm() < 1e-6 ? 1e-10 : 1e-6;
            high[axis] += step;
            low[axis] -= step;
            SO3Tangentd positive(high), negative(low);
            auto high_rotation = positive.Exp();
            auto low_rotation = negative.Exp();
            for (int i = 0; i < 4; ++i) {
                double expected =
                    (high_rotation.Coeffs()[i] - low_rotation.Coeffs()[i]) /
                    (2 * step);
                if (!std::isfinite(rotation.Coeffs()[i].value()) ||
                    !rotation.Coeffs()[i].derivatives().allFinite() ||
                    std::abs(rotation.Coeffs()[i].derivatives()[axis] -
                             expected) > 1e-7)
                    return 1;
            }
            Eigen::Matrix3d left_expected =
                                (positive.Ljac() - negative.Ljac()) /
                                (2 * step),
                            inverse_expected =
                                (positive.Ljacinv() - negative.Ljacinv()) /
                                (2 * step);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (!std::isfinite(left(i, j).value()) ||
                        !std::isfinite(inverse(i, j).value()) ||
                        !left(i, j).derivatives().allFinite() ||
                        !inverse(i, j).derivatives().allFinite() ||
                        !right(i, j).derivatives().allFinite() ||
                        std::abs(right(i, j).value() - left(j, i).value()) >
                            1e-12 ||
                        (right(i, j).derivatives() - left(j, i).derivatives())
                                .norm() > 1e-12 ||
                        std::abs(left(i, j).derivatives()[axis] -
                                 left_expected(i, j)) > 1e-7 ||
                        std::abs(inverse(i, j).derivatives()[axis] -
                                 inverse_expected(i, j)) > 1e-7)
                        return 2;
        }
    }
    std::cout << "Eigen autodiff SO3 exponential/Jacobian derivatives agree "
                 "with finite differences\n";
}
