#include "holistic_motion/manif/LieGroup.h"
#include "holistic_motion/manif/ceres/constraint.h"
#include "holistic_motion/manif/ceres/local_parametrization.h"
#include "holistic_motion/manif/ceres/objective.h"

#include <cmath>
#include <iostream>
#include <unsupported/Eigen/AutoDiff>

using AD = Eigen::AutoDiffScalar<Eigen::Matrix<double, 6, 1>>;
namespace holistic_motion::robotics {
template <> struct Constants<AD> {
    inline static const AD eps = AD(1e-12);
};
} // namespace holistic_motion::robotics

using namespace holistic_motion::robotics;

template <class Group> bool CheckFunctors() {
    using Tangent = typename Group::Tangent;
    using Covariance = typename CeresConstraintFunctor<Group>::Covariance;
    Tangent first, second, measurement;
    for (int index = 0; index < Group::DoF; ++index) {
        first.Coeffs()[index] = .1 * std::sin(index + .2);
        second.Coeffs()[index] = .2 * std::cos(index + .3);
        measurement.Coeffs()[index] = .03 * (index + 1);
    }
    Group past = first.Exp(), future = second.Exp();
    Covariance covariance = Covariance::Identity();
    covariance += measurement.Coeffs() * measurement.Coeffs().transpose();
    CeresConstraintFunctor<Group> constraint(measurement, covariance);
    auto copied = constraint;
    typename Tangent::DataType residual;
    if (!copied(past.Coeffs().data(), future.Coeffs().data(), residual.data()))
        return false;
    const auto error = (measurement - (future - past)).Coeffs().eval();
    const double expected_cost = error.dot(covariance.ldlt().solve(error));
    if (std::abs(residual.squaredNorm() - expected_cost) > 1e-12)
        return false;

    CeresObjectiveFunctor<Group> objective(past, 2.0);
    double objective_residual;
    if (!objective(future.Coeffs().data(), &objective_residual) ||
        std::abs(objective_residual - 2.0 * (past - future).Coeffs().norm()) > 1e-12)
        return false;
    auto objective_copy = objective;
    objective.SetTargetState(future);
    objective.Weight(3.0);
    objective(future.Coeffs().data(), &objective_residual);
    if (std::abs(objective_residual) > 1e-14 || objective.Weight() != 3.0 ||
        objective_copy.Weight() != 2.0 || !objective_copy.GetTargetState().IsApprox(past))
        return false;

    // The setter treats the upper triangle as the symmetric covariance.
    Covariance asymmetric = 2.0 * covariance;
    asymmetric.template triangularView<Eigen::StrictlyLower>().setConstant(-99.0);
    constraint.SetMeasurementCovariance(asymmetric);
    const Covariance expected_covariance = 2.0 * covariance;
    if (!constraint.GetMeasurementCovariance().isApprox(expected_covariance, 1e-14))
        return false;
    constraint.SetMeasurement(measurement);
    if (!constraint.GetMeasurement().IsApprox(measurement, 0.0))
        return false;

    CeresLocalParameterizationFunctor<Group> local;
    const auto ad_past = past.template cast<AD>();
    typename Tangent::template TangentTemplate<AD> delta;
    for (int axis = 0; axis < Group::DoF; ++axis) {
        delta.Coeffs()[axis] = AD(0.0);
        delta.Coeffs()[axis].derivatives()[axis] = 1.0;
    }
    typename Group::template LieGroupTemplate<AD> displaced;
    local(ad_past.Coeffs().data(), delta.Coeffs().data(), displaced.Coeffs().data());
    Eigen::Matrix<AD, Group::DoF, 1> ad_residual;
    copied(ad_past.Coeffs().data(), displaced.Coeffs().data(), ad_residual.data());
    AD zero_objective, nonzero_objective;
    objective_copy(displaced.Coeffs().data(), &zero_objective);
    objective(displaced.Coeffs().data(), &nonzero_objective);
    if (!zero_objective.derivatives().allFinite() || zero_objective.derivatives().norm() > 1e-12 ||
        !nonzero_objective.derivatives().allFinite()) {
        std::cerr << "objective derivatives at target: dof=" << Group::DoF
                  << " representation=" << Group::RepSize << " value=" << zero_objective.value()
                  << " gradient=" << zero_objective.derivatives().transpose() << '\n';
        return false;
    }
    if constexpr (std::is_same_v<Group, SO3d> || std::is_same_v<Group, SE3d>) {
        auto antipodal = displaced;
        antipodal.Coeffs().template tail<4>() *= AD(-1.0);
        AD equivalent_objective;
        objective_copy(antipodal.Coeffs().data(), &equivalent_objective);
        if (std::abs(equivalent_objective.value()) > 1e-14 ||
            !equivalent_objective.derivatives().allFinite() ||
            equivalent_objective.derivatives().norm() > 1e-12)
            return false;
    }
    for (int axis = 0; axis < Group::DoF; ++axis) {
        Tangent high = Tangent::ZeroHelper(), low = Tangent::ZeroHelper();
        constexpr double step = 1e-6;
        high.Coeffs()[axis] = step;
        low.Coeffs()[axis] = -step;
        Group positive, negative;
        local(past.Coeffs().data(), high.Coeffs().data(), positive.Coeffs().data());
        local(past.Coeffs().data(), low.Coeffs().data(), negative.Coeffs().data());
        typename Tangent::DataType positive_residual, negative_residual;
        copied(past.Coeffs().data(), positive.Coeffs().data(), positive_residual.data());
        copied(past.Coeffs().data(), negative.Coeffs().data(), negative_residual.data());
        double positive_objective, negative_objective;
        objective(positive.Coeffs().data(), &positive_objective);
        objective(negative.Coeffs().data(), &negative_objective);
        if (std::abs(nonzero_objective.derivatives()[axis] -
                     (positive_objective - negative_objective) / (2 * step)) > 1e-8)
            return false;
        for (int row = 0; row < Group::DoF; ++row) {
            const double expected = (positive_residual[row] - negative_residual[row]) / (2 * step);
            if (!std::isfinite(ad_residual[row].value()) ||
                !ad_residual[row].derivatives().allFinite() ||
                std::abs(ad_residual[row].derivatives()[axis] - expected) > 1e-8)
                return false;
        }
    }
    return true;
}

int main() {
    if (!CheckFunctors<R3d>() || !CheckFunctors<SO2d>() || !CheckFunctors<SO3d>() ||
        !CheckFunctors<SE2d>() || !CheckFunctors<SE3d>()) {
        std::cerr << "manifold residual functor regression\n";
        return 1;
    }
    // Forward raw coordinate arguments while keeping weighted/covariance
    // overloads available for both const and non-const group objects.
    CeresObjectiveFunctor<SO3d> coordinates(.1, .2, .3);
    CeresConstraintFunctor<SO3d> measurement(Eigen::Vector3d(.1, .2, .3));
    return !coordinates.GetTargetState().IsApprox(SO3d(.1, .2, .3), 0.0) ||
           measurement.GetMeasurement().Coeffs() != Eigen::Vector3d(.1, .2, .3);
}
