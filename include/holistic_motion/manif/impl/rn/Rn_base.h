#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_BASE_H_

#include "holistic_motion/manif/impl/lie_group_base.h"
#include "holistic_motion/manif/impl/rn/Rn_properties.h"

namespace holistic_motion {
namespace robotics {
//
// LieGroup
//

/**
 * @brief The base class of the Rn group.
 * @note See Appendix E.
 */
template <typename _Derived>
struct RnBase : LieGroupBase<_Derived> {
private:
    using Base = LieGroupBase<_Derived>;
    using Type = RnBase<_Derived>;

public:
    HOLISTIC_MOTION_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_AUTO_API
    HOLISTIC_MOTION_INHERIT_GROUP_OPERATOR

    using Base::Coeffs;

    using Transformation = typename internal::traits<_Derived>::Transformation;

    // LieGroup common API
protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(RnBase)

public:
    HOLISTIC_MOTION_GROUP_ML_ASSIGN_OP(RnBase)

    /**
     * @brief Get the inverse of this.
     * @param[out] -optional- J_minv_m Jacobian of the inverse wrt this.
     * @note r^-1 = -r
     * @note See Appendix E and Eq. (189).
     */
    LieGroup Inverse(OptJacobianRef J_minv_m = {}) const;

    /**
     * @brief Get the Rn corresponding Lie algebra element in vector form.
     * @param[out] -optional- J_t_m Jacobian of the tangent wrt to this.
     * @return The Rn tangent of this.
     * @note This is the log() map in vector form.
     * @note See Appendix E.
     * @see RnTangent.
     */
    Tangent Log(OptJacobianRef J_t_m = {}) const;

    /**
     * @brief This function is deprecated.
     * Please considere using
     * @ref log instead.
     */
    HOLISTIC_MOTION_DEPRECATED
    Tangent Lift(OptJacobianRef J_t_m = {}) const;

    /**
     * @brief Composition of this and another Rn element.
     * @param[in] m Another Rn element.
     * @param[out] -optional- J_mc_ma Jacobian of the composition wrt this.
     * @param[out] -optional- J_mc_mb Jacobian of the composition wrt m.
     * @return The composition of 'this . m'.
     * @note See Eq. (190).
     */
    template <typename _DerivedOther>
    LieGroup Compose(const LieGroupBase<_DerivedOther>& m,
                     OptJacobianRef J_mc_ma = {},
                     OptJacobianRef J_mc_mb = {}) const;

    /**
     * @brief Translation action on a 2-vector.
     * @param v A 2-vector.
     * @param[out] -optional- J_vout_m The Jacobian of the new object wrt this.
     * @param[out] -optional- J_vout_v The Jacobian of the new object wrt input
     * object.
     * @return The translated 2-vector.
     */
    template <typename _EigenDerived>
    auto
    Act(const Eigen::MatrixBase<_EigenDerived>& v,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> J_vout_m = {},
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> J_vout_v = {})
            const -> Eigen::Matrix<Scalar, Dim, 1>;

    /**
     * @brief Get the ajoint matrix of Rn at this.
     * @note See Eqs. (188).
     */
    Jacobian Adj() const;

    // Rn specific functions

    /**
     * @brief Get the transformation matrix (2D isometry).
     * @note T = | 0 t |
     *           | 0 1 |
     */
    Transformation GetTransform() const;
};

template <typename _Derived>
typename RnBase<_Derived>::Transformation RnBase<_Derived>::GetTransform()
        const {
    Transformation T(Transformation::Identity());
    T.template topRightCorner<Dim, 1>() = Coeffs();
    return T;
}

template <typename _Derived>
typename RnBase<_Derived>::LieGroup RnBase<_Derived>::Inverse(
        OptJacobianRef J_minv_m) const {
    if (J_minv_m) J_minv_m->setIdentity() *= Scalar(-1);

    return LieGroup(-Coeffs());
}

template <typename _Derived>
typename RnBase<_Derived>::Tangent RnBase<_Derived>::Log(
        OptJacobianRef J_t_m) const {
    if (J_t_m) J_t_m->setIdentity();

    return Tangent(Coeffs());
}

template <typename _Derived>
typename RnBase<_Derived>::Tangent RnBase<_Derived>::Lift(
        OptJacobianRef J_t_m) const {
    return Log(J_t_m);
}

template <typename _Derived>
template <typename _DerivedOther>
typename RnBase<_Derived>::LieGroup RnBase<_Derived>::Compose(
        const LieGroupBase<_DerivedOther>& m,
        OptJacobianRef J_mc_ma,
        OptJacobianRef J_mc_mb) const {
    static_assert(std::is_base_of<RnBase<_DerivedOther>, _DerivedOther>::value,
                  "Argument does not inherit from RnBase !");

    static_assert(RnBase<_DerivedOther>::Dim == _DerivedOther::Dim,
                  "Dimension mismatch !");

    if (J_mc_ma) J_mc_ma->setIdentity();

    if (J_mc_mb) J_mc_mb->setIdentity();

    return LieGroup(Coeffs() + m.Coeffs());
}

template <typename _Derived>
template <typename _EigenDerived>
// Eigen::Matrix<typename RnBase<_Derived>::Scalar, RnBase<_Derived>::Dim, 1>
auto RnBase<_Derived>::Act(
        const Eigen::MatrixBase<_EigenDerived>& v,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> J_vout_m,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> J_vout_v)
        const -> Eigen::Matrix<Scalar, Dim, 1> {
    assert_vector_dim(v, Dim);

    if (J_vout_m) {
        J_vout_m->setIdentity();
    }

    if (J_vout_v) {
        J_vout_v->setIdentity();
    }

    return Coeffs() + v;
}

template <typename _Derived>
typename RnBase<_Derived>::Jacobian RnBase<_Derived>::Adj() const {
    static const Jacobian adj = Jacobian::Identity();
    return adj;
}

namespace internal {

//! @brief Random specialization for RnBase objects.
template <typename Derived>
struct RandomEvaluatorImpl<RnBase<Derived>> {
    template <typename T>
    static void run(T& m) {
        using Tangent = typename Derived::Tangent;
        m = Tangent::RandomHelper().Exp();
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_BASE_H_
