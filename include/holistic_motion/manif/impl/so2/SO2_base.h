#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_BASE_H_

#include "holistic_motion/manif/impl/lie_group_base.h"
#include "holistic_motion/manif/impl/so2/SO2_properties.h"

namespace holistic_motion {
namespace robotics {
//
// LieGroup
//

/**
 * @brief The base class of the SO2 group.
 * @note See Appendix A of the paper.
 */
template <typename _Derived>
struct SO2Base : LieGroupBase<_Derived> {
private:
    using Base = LieGroupBase<_Derived>;
    using Type = SO2Base<_Derived>;

public:
    HOLISTIC_MOTION_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_AUTO_API
    HOLISTIC_MOTION_INHERIT_GROUP_OPERATOR

    using Base::Coeffs;

    using Rotation = typename internal::traits<_Derived>::Rotation;
    using Transformation = typename internal::traits<_Derived>::Transformation;

    // LieGroup common API

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SO2Base)

public:
    HOLISTIC_MOTION_GROUP_ML_ASSIGN_OP(SO2Base)

    /**
     * @brief Get the inverse of this.
     * @param[out] -optional- J_minv_m Jacobian of the inverse wrt this.
     * @note z^-1 = z*
     * @note See Eqs. (118,124).
     */
    LieGroup Inverse(OptJacobianRef J_minv_m = {}) const;

    /**
     * @brief Get the SO2 corresponding Lie algebra element in vector form.
     * @param[out] -optional- J_t_m Jacobian of the tangent wrt to this.
     * @return The SO2 tangent of this.
     * @note This is the log() map in vector form.
     * @note See Eq. (115) & Eqs. (79,126).
     * @see SO2Tangent.
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
     * @brief Composition of this and another SO2 element.
     * @param[in] m Another SO2 element.
     * @param[out] -optional- J_mc_ma Jacobian of the composition wrt this.
     * @param[out] -optional- J_mc_mb Jacobian of the composition wrt m.
     * @return The composition of 'this . m'.
     * @note z_c = z_a z_b.
     * @note See Eq. (125).
     */
    template <typename _DerivedOther>
    LieGroup Compose(const LieGroupBase<_DerivedOther>& m,
                     OptJacobianRef J_mc_ma = {},
                     OptJacobianRef J_mc_mb = {}) const;

    /**
     * @brief Rotation action on a 2-vector.
     * @param  v A 2-vector.
     * @param[out] -optional- J_vout_m The Jacobian of the new object wrt this.
     * @param[out] -optional- J_vout_v The Jacobian of the new object wrt input
     * object.
     * @return The rotated 2-vector.
     * @note See Eqs. (129, 130).
     */
    template <typename _EigenDerived>
    Eigen::Matrix<Scalar, 2, 1> Act(
            const Eigen::MatrixBase<_EigenDerived>& v,
            tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 2, 1>>> J_vout_m = {},
            tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 2, 2>>> J_vout_v = {})
            const;

    /**
     * @brief Get the ajoint matrix of SO2 at this.
     * @note See Eqs. (123).
     */
    Jacobian Adj() const;

    // SO2 specific functions

    /**
     * @brief Get the transformation matrix (2D isometry).
     * @note T = | R 0 |
     *           | 0 1 |
     */
    Transformation GetTransform() const;

    /**
     * @brief Get the rotation matrix R.
     */
    Rotation GetRotation() const;

    /**
     * @brief Get the real part of the underlying complex number.
     */
    Scalar Real() const;

    /**
     * @brief Get the imaginary part of the underlying complex number.
     */
    Scalar Imag() const;

    /**
     * @brief Get the angle (rad.).
     */
    Scalar Angle() const;

    /**
     * @brief Normalize the underlying complex number.
     */
    void Normalize();

    // protected:

    /// @todo given a Eigen::Map<const SO2>
    /// Coeffs()->x() return a reference to
    /// temporary ...

    //  Scalar& real();
    //  Scalar& imag();
};

template <typename _Derived>
typename SO2Base<_Derived>::Transformation SO2Base<_Derived>::GetTransform()
        const {
    Transformation T(Transformation::Identity());
    T.template topLeftCorner<2, 2>() = GetRotation();
    return T;
}

template <typename _Derived>
typename SO2Base<_Derived>::Rotation SO2Base<_Derived>::GetRotation() const {
    using std::cos;
    using std::sin;
    const Scalar theta = Angle();
    return (GetRotation() << cos(theta), -sin(theta), sin(theta), cos(theta))
            .finished();
}

template <typename _Derived>
typename SO2Base<_Derived>::LieGroup SO2Base<_Derived>::Inverse(
        OptJacobianRef J_minv_m) const {
    if (J_minv_m) J_minv_m->setConstant(Scalar(-1));

    return LieGroup(Real(), -Imag());
}

template <typename _Derived>
typename SO2Base<_Derived>::Tangent SO2Base<_Derived>::Log(
        OptJacobianRef J_t_m) const {
    if (J_t_m) J_t_m->setConstant(Scalar(1));

    return Tangent(Angle());
}

template <typename _Derived>
typename SO2Base<_Derived>::Tangent SO2Base<_Derived>::Lift(
        OptJacobianRef J_t_m) const {
    return Log(J_t_m);
}

template <typename _Derived>
template <typename _DerivedOther>
typename SO2Base<_Derived>::LieGroup SO2Base<_Derived>::Compose(
        const LieGroupBase<_DerivedOther>& m,
        OptJacobianRef J_mc_ma,
        OptJacobianRef J_mc_mb) const {
    using std::abs;

    static_assert(std::is_base_of<SO2Base<_DerivedOther>, _DerivedOther>::value,
                  "Argument does not inherit from SE2Base !");

    if (J_mc_ma) J_mc_ma->setConstant(Scalar(1));

    if (J_mc_mb) J_mc_mb->setConstant(Scalar(1));

    const auto& m_so2 = static_cast<const SO2Base<_DerivedOther>&>(m);

    Scalar ret_real = Real() * m_so2.Real() - Imag() * m_so2.Imag();
    Scalar ret_imag = Real() * m_so2.Imag() + Imag() * m_so2.Real();

    const Scalar ret_sqnorm = ret_real * ret_real + ret_imag * ret_imag;

    if (abs(ret_sqnorm - Scalar(1)) > Constants<Scalar>::eps) {
        const Scalar scale = ApproxSqrtInv(ret_sqnorm);
        ret_real *= scale;
        ret_imag *= scale;
    }

    return LieGroup(ret_real, ret_imag);
}

template <typename _Derived>
template <typename _EigenDerived>
Eigen::Matrix<typename SO2Base<_Derived>::Scalar, 2, 1> SO2Base<_Derived>::Act(
        const Eigen::MatrixBase<_EigenDerived>& v,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 2, 1>>> J_vout_m,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 2, 2>>> J_vout_v) const {
    assert_vector_dim(v, 2);
    const Rotation R(GetRotation());

    if (J_vout_m) {
        J_vout_m->noalias() = R * skew(Scalar(1)) * v;
    }

    if (J_vout_v) {
        (*J_vout_v) = R;
    }

    return R * v;
}

template <typename _Derived>
typename SO2Base<_Derived>::Jacobian SO2Base<_Derived>::Adj() const {
    static const Jacobian adj = Jacobian::Constant(Scalar(1));
    return adj;
}

// SO2 specific function

template <typename _Derived>
/*const*/ typename SO2Base<_Derived>::Scalar /*&*/
SO2Base<_Derived>::Real() const {
    return Coeffs().x();
}

template <typename _Derived>
/*const*/ typename SO2Base<_Derived>::Scalar /*&*/
SO2Base<_Derived>::Imag() const {
    return Coeffs().y();
}

template <typename _Derived>
typename SO2Base<_Derived>::Scalar SO2Base<_Derived>::Angle() const {
    using std::atan2;
    return atan2(Imag(), Real());
}

// template <typename _Derived>
// typename SO2Base<_Derived>::Scalar&
// SO2Base<_Derived>::real()
//{
//  return Coeffs.x();
//}

// template <typename _Derived>
// typename SO2Base<_Derived>::Scalar&
// SO2Base<_Derived>::imag()
//{
//  return Coeffs.y();
//}

template <typename _Derived>
void SO2Base<_Derived>::Normalize() {
    Coeffs().normalize();
}

namespace internal {

//! @brief Random specialization for SO2Base objects.
template <typename Derived>
struct RandomEvaluatorImpl<SO2Base<Derived>> {
    template <typename T>
    static void run(T& m) {
        using Tangent = typename LieGroupBase<Derived>::Tangent;
        m = Tangent::RandomHelper().Exp();
    }
};

//! @brief Assignment assert specialization for SO2Base objects
template <typename Derived>
struct AssignmentEvaluatorImpl<SO2Base<Derived>> {
    template <typename T>
    static void run_impl(const T& data) {
        using std::abs;
        HOLISTIC_MOTION_ASSERT(abs(data.norm() - typename SO2Base<Derived>::Scalar(1)) <
                            Constants<typename SO2Base<Derived>::Scalar>::eps,
                    "SO2 assigned data not normalized !", invalid_argument);
        HOLISTIC_MOTION_UNUSED_VARIABLE(data);
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_BASE_H_ */
