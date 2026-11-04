#include "holistic_motion/manif/LieGroup.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

using namespace holistic_motion::robotics;

static_assert(internal::compute_indices<3, 4, 3>()[2] == 7);
static_assert(internal::compute_indices<>().size() == 0);
static_assert(internal::compute_indices<4>()[0] == 0);
static_assert(std::is_same_v<internal::make_intseq_t<4>, internal::intseq<0, 1, 2, 3>>);

template <class Group> void CheckGroup() {
    using Tangent = typename Group::Tangent;
    using Jacobian = typename Group::Jacobian;
    Tangent first_tangent, second_tangent;
    for (int axis = 0; axis < Group::DoF; ++axis) {
        first_tangent.Coeffs()[axis] = .2 * std::sin(axis + .3);
        second_tangent.Coeffs()[axis] = .1 * std::cos(axis + .7);
    }
    Jacobian exponential_jacobian, logarithm_jacobian;
    const Group first = first_tangent.Exp(exponential_jacobian);
    const Group second = second_tangent.Exp();
    if (!Group::IdentityHelper().GetTransform().isIdentity(1e-14) ||
        !Group::RandomHelper().Coeffs().allFinite() ||
        !Tangent::RandomHelper().Coeffs().allFinite())
        throw std::runtime_error("composed group identity/random helper is invalid");
    const Tangent recovered = first.Log(logarithm_jacobian);
    if (!recovered.IsApprox(first_tangent, 1e-12) ||
        !(logarithm_jacobian * exponential_jacobian).isApprox(Jacobian::Identity(), 1e-12)) {
        std::cerr << "dof=" << Group::DoF
                  << " roundtrip=" << (recovered - first_tangent).Coeffs().norm() << " Jacobians="
                  << (logarithm_jacobian * exponential_jacobian - Jacobian::Identity()).norm()
                  << '\n';
        throw std::runtime_error("composed group exponential/logarithm is inconsistent");
    }
    // SO3's homogeneous transform embeds a 3x3 algebra in a 4x4 matrix.
    // Compare matrix exponentials only when both representations coincide.
    if constexpr (static_cast<int>(Tangent::LieAlg::RowsAtCompileTime) ==
                  static_cast<int>(Group::Transformation::RowsAtCompileTime)) {
        const typename Group::Transformation matrix_exponential = first_tangent.Hat().exp();
        if (!first.GetTransform().isApprox(matrix_exponential, 1e-12))
            throw std::runtime_error("group exponential differs from the matrix exponential");
    }

    const auto product = first.Compose(second);
    const Eigen::Map<const Group> mapped(first.Coeffs().data());
    if (!product.GetTransform().isApprox(first.GetTransform() * second.GetTransform(), 1e-12) ||
        !mapped.Compose(second).IsApprox(product, 1e-12) ||
        !first.Inverse().GetTransform().isApprox(first.GetTransform().inverse(), 1e-12))
        throw std::runtime_error("composed group product/inverse is inconsistent");
    const auto restored = first.template cast<float>().template cast<double>();
    if (!restored.GetTransform().isApprox(first.GetTransform(), 1e-7) ||
        std::abs(restored.GetTransform().determinant() - 1.0) > 1e-12)
        throw std::runtime_error("composed group scalar cast changed the state");

    const TangentBase<Tangent> &generic = first_tangent;
    if (!(generic.Rjacinv() * exponential_jacobian).isApprox(Jacobian::Identity(), 1e-12))
        throw std::runtime_error("composed group generic inverse Jacobian is inconsistent");
    constexpr double step = 1e-6;
    for (int axis = 0; axis < Group::DoF; ++axis) {
        Tangent basis = Tangent::ZeroHelper();
        basis.Coeffs()[axis] = 1.0;
        if (!Tangent::GeneratorHelper(axis).isApprox(basis.Hat(), 0.0))
            throw std::runtime_error("composed group generator differs from its basis");
        Tangent high = first_tangent, low = first_tangent;
        high.Coeffs()[axis] += step;
        low.Coeffs()[axis] -= step;
        const auto positive = high.Exp().Rminus(first);
        const auto negative = low.Exp().Rminus(first);
        const auto expected = ((positive - negative).Coeffs() / (2 * step)).eval();
        if (!expected.isApprox(exponential_jacobian.col(axis), 1e-7))
            throw std::runtime_error("composed group exponential Jacobian differs from samples");
    }
    Tangent bracket;
    bracket.Coeffs() = first_tangent.SmallAdj() * second_tangent.Coeffs();
    if (!bracket.Hat().isApprox(first_tangent.Hat() * second_tangent.Hat() -
                                    second_tangent.Hat() * first_tangent.Hat(),
                                1e-12))
        throw std::runtime_error("composed group small adjoint differs from its Lie bracket");

    typename Group::Vector point = Group::Vector::Ones();
    Eigen::Matrix<double, Group::Dim, Group::DoF> state_jacobian;
    Eigen::Matrix<double, Group::Dim, Group::Dim> point_jacobian;
    const auto acted = first.Act(point, state_jacobian, point_jacobian);
    if (!acted.allFinite() || !state_jacobian.allFinite() || !point_jacobian.allFinite())
        throw std::runtime_error("composed group action/Jacobian is invalid");
    for (int axis = 0; axis < Group::DoF; ++axis) {
        Tangent offset = Tangent::ZeroHelper();
        offset.Coeffs()[axis] = step;
        const auto expected =
            ((first.Rplus(offset).Act(point) - first.Rplus(-offset).Act(point)) / (2 * step))
                .eval();
        if (!expected.isApprox(state_jacobian.col(axis), 1e-7))
            throw std::runtime_error("composed group action Jacobian differs from samples");
    }
}

template <typename Scalar, int Dimension> void CheckTranslationMetric() {
    using Tangent = RnTangent<Scalar, Dimension>;
    using Matrix = typename Tangent::InnerWeightsMatrix;
    Tangent first, second;
    for (int axis = 0; axis < Dimension; ++axis) {
        first.Coeffs()[axis] = Scalar(.25 * (axis + 1));
        second.Coeffs()[axis] = Scalar(-.125 * axis);
    }
    const Scalar expected_inner = first.Coeffs().dot(second.Coeffs());
    const Scalar expected_squared = first.Coeffs().squaredNorm();
    const Scalar expected_norm = first.Coeffs().norm();
    const Scalar tolerance = 32 * std::numeric_limits<Scalar>::epsilon();
    const auto check = [&](const auto &tangent) {
        if ((tangent.InnerWeights().array() != Matrix::Identity().array()).any() ||
            std::abs(tangent.Inner(second) - expected_inner) >
                tolerance * (1 + std::abs(expected_inner)) ||
            std::abs(tangent.SquaredWeightedNorm() - expected_squared) >
                tolerance * (1 + expected_squared) ||
            std::abs(tangent.WeightedNorm() - expected_norm) >
                tolerance * (1 + expected_norm))
            throw std::runtime_error(
                "translation metric differs from Euclidean coordinates");
    };
    check(first);
    check(Eigen::Map<Tangent>(first.Coeffs().data()));
    check(Eigen::Map<const Tangent>(first.Coeffs().data()));
}

template <int Dimension> void CheckTranslation() {
    using Group = Rn<double, Dimension>;
    static_assert(Group::Transformation::RowsAtCompileTime == Dimension + 1);
    Group first, second;
    for (int axis = 0; axis < Dimension; ++axis) {
        first.Coeffs()[axis] = .2 * (axis + 1);
        second.Coeffs()[axis] = -.1 * axis;
    }
    typename Group::Transformation expected = Group::Transformation::Identity();
    expected.template topRightCorner<Dimension, 1>() = first.Coeffs();
    if (!first.GetTransform().isApprox(expected, 0.0) ||
        !first.Compose(second).GetTransform().isApprox(first.GetTransform() * second.GetTransform(),
                                                       1e-14))
        throw std::runtime_error("translation matrix lost its homogeneous coordinate");
    CheckTranslationMetric<double, Dimension>();
    CheckTranslationMetric<float, Dimension>();
}

int main() {
    try {
        CheckTranslation<1>();
        CheckTranslation<2>();
        CheckTranslation<3>();
        CheckTranslation<7>();
        CheckTranslation<32>();
        CheckGroup<SO2d>();
        CheckGroup<SO3d>();
        CheckGroup<SE2d>();
        CheckGroup<SE3d>();
        CheckGroup<Bundle<double, SO3, R3>>();
        CheckGroup<Bundle<double, SE3, R3>>();
        CheckGroup<Bundle<double, SO2, R2>>();
        CheckGroup<SE_2_3d>();
        const Eigen::Vector3d position(1, 2, 3), velocity(.1, .2, .3);
        const Eigen::Quaterniond orientation(Eigen::AngleAxisd(.4, Eigen::Vector3d::UnitY()));
        const SE_2_3d state(position, orientation, velocity);
        Eigen::Matrix<double, 5, 5> expected = Eigen::Matrix<double, 5, 5>::Identity();
        expected.topLeftCorner<3, 3>() = orientation.toRotationMatrix();
        expected.block<3, 1>(0, 3) = position;
        expected.block<3, 1>(0, 4) = velocity;
        if (!state.GetTransform().isApprox(expected, 1e-14) ||
            !state.isometry().isApprox(expected, 1e-14))
            throw std::runtime_error(
                "extended pose constructor lost its position/rotation/velocity");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
