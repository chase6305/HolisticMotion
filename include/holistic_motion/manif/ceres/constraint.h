#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CONSTRAINT_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CONSTRAINT_H_

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace holistic_motion {
namespace robotics {
template <typename _LieGroup>
class CeresConstraintFunctor {
    using LieGroup = _LieGroup;
    using Tangent = typename _LieGroup::Tangent;

    template <typename _Scalar>
    using LieGroupTemplate =
            typename LieGroup::template LieGroupTemplate<_Scalar>;

    template <typename _Scalar>
    using TangentTemplate = typename Tangent::template TangentTemplate<_Scalar>;

public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND_TYPE(Tangent)

    using Covariance = Eigen::Matrix<double, LieGroup::DoF, LieGroup::DoF>;
    using InformationMatrix = Covariance;

    template <typename... Args>
    CeresConstraintFunctor(Args&&... args)
        : measurement_(std::forward<Args>(args)...),
          measurement_covariance_(Covariance::Identity()) {
        ComputeInformationMatrix();
    }

    template <typename... Args>
    CeresConstraintFunctor(
            const Tangent& measurement,
            const Covariance& measurement_covariance = Covariance::Identity())
        : measurement_(measurement),
          measurement_covariance_(measurement_covariance) {
        ComputeInformationMatrix();
    }

    virtual ~CeresConstraintFunctor() = default;

    template <typename T>
    bool operator()(const T* const past_raw,
                    const T* const futur_raw,
                    T* residuals_raw) const {
        const Eigen::Map<const LieGroupTemplate<T>> state_past(past_raw);
        const Eigen::Map<const LieGroupTemplate<T>> state_future(futur_raw);

        Eigen::Map<TangentTemplate<T>> residuals(residuals_raw);

        /// r = m - ( future (-) past )
        residuals =
                measurement_.template cast<T>() - (state_future - state_past);

        /// r = exp( log(m)^-1 . ( past^-1 . future ) )

        //    residuals =
        //      measurement_.Exp().template cast<T>()
        //        .Between(state_past.Between(state_future)).log();

        residuals.Coeffs() = measurement_sqrt_info_upper_.template cast<T>() *
                             residuals.Coeffs();

        return true;
    }

    Tangent GetMeasurement() const;
    void SetMeasurement(const Tangent& measurement);

    Covariance GetMeasurementCovariance() const;
    void SetMeasurementCovariance(const Covariance covariance);

protected:
    void ComputeInformationMatrix();

protected:
    Tangent measurement_;

    Covariance measurement_covariance_;
    InformationMatrix measurement_sqrt_info_upper_;
};

template <typename _LieGroup>
typename CeresConstraintFunctor<_LieGroup>::Tangent
CeresConstraintFunctor<_LieGroup>::GetMeasurement() const {
    return measurement_;
}

template <typename _LieGroup>
void CeresConstraintFunctor<_LieGroup>::SetMeasurement(
        const Tangent& measurement) {
    measurement_ = measurement;
}

template <typename _LieGroup>
typename CeresConstraintFunctor<_LieGroup>::Covariance
CeresConstraintFunctor<_LieGroup>::GetMeasurementCovariance() const {
    return measurement_covariance_;
}

template <typename _LieGroup>
void CeresConstraintFunctor<_LieGroup>::SetMeasurementCovariance(
        const Covariance covariance) {
    // Ensuring symmetry
    measurement_covariance_ =
            covariance.template selfadjointView<Eigen::Upper>();
    ComputeInformationMatrix();
}

template <typename _LieGroup>
void CeresConstraintFunctor<_LieGroup>::ComputeInformationMatrix() {
    // compute square root information upper matrix

    // ensuring symmetry
    const InformationMatrix info =
            measurement_covariance_.inverse()
                    .template selfadjointView<Eigen::Upper>();

    // Normal Cholesky factorization
    Eigen::LLT<InformationMatrix> llt_of_info(info);
    InformationMatrix R = llt_of_info.matrixU();

    // Factorization not good enough
    if (!info.IsApprox(R.transpose() * R, 1e-6)) {
        Eigen::SelfAdjointEigenSolver<InformationMatrix> es(info);
        Eigen::VectorXd eval = es.eigenvalues().real().cwiseMax(1e-6);

        R = eval.cwiseSqrt().asDiagonal() *
            es.eigenvectors().real().transpose();
    }

    measurement_sqrt_info_upper_ = R;
}

}  // namespace robotics
}  // namespace holistic_motion
#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CONSTRAINT_H_ */
