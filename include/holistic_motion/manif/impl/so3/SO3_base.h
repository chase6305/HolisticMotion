#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3_BASE_H_

#include "holistic_motion/manif/impl/lie_group_base.h"
#include "holistic_motion/manif/impl/so3/SO3_properties.h"
#include "holistic_motion/manif/impl/utils.h"

namespace holistic_motion {
namespace robotics {

//
// LieGroup
//

/**
 * @brief The base class of the SO3 group.
 * @note See Appendix B of the paper.
 */
template <typename _Derived>
struct SO3Base : LieGroupBase<_Derived> {
private:
    using Base = LieGroupBase<_Derived>;
    using Type = SO3Base<_Derived>;

public:
    HOLISTIC_MOTION_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_AUTO_API
    HOLISTIC_MOTION_INHERIT_GROUP_OPERATOR

    using Base::Coeffs;

    using Rotation = typename internal::traits<_Derived>::Rotation;
    using Transformation = typename internal::traits<_Derived>::Transformation;
    using QuaternionDataType = Eigen::Quaternion<Scalar>;

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SO3Base)

public:
    HOLISTIC_MOTION_GROUP_ML_ASSIGN_OP(SO3Base)

    // LieGroup common API

    /**
     * @brief Get the inverse of this.
     * @param[out] -optional- J_minv_m Jacobian of the inverse wrt this.
     * @note q^-1 = q*. See Eq. (140).
     */
    LieGroup Inverse(OptJacobianRef J_minv_m = {}) const;

    /**
     * @brief Get the SO3 corresponding Lie algebra element in vector form.
     * @param[out] -optional- J_t_m Jacobian of the tangent wrt to this.
     * @return The SO3 tangent of this.
     * @note This is the log() map in vector form.
     * @note See Eq. (133) & Eq. (144).
     * @see SO3Tangent.
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
     * @brief Composition of this and another SO3 element.
     * @param[in] m Another SO3 element.
     * @param[out] -optional- J_mc_ma Jacobian of the composition wrt this.
     * @param[out] -optional- J_mc_mb Jacobian of the composition wrt m.
     * @return The composition of 'this . m'.
     * @note Quaternion product.
     * @note See Eqs. (141,142).
     */
    template <typename _DerivedOther>
    LieGroup Compose(const LieGroupBase<_DerivedOther>& m,
                     OptJacobianRef J_mc_ma = {},
                     OptJacobianRef J_mc_mb = {}) const;

    /**
     * @brief Rotation action on a 3-vector.
     * @param  v A 2-vector.
     * @param[out] -optional- J_vout_m The Jacobian of the new object wrt this.
     * @param[out] -optional- J_vout_v The Jacobian of the new object wrt input
     * object.
     * @return The rotated 3-vector.
     * @note See Eq (136), Eqs. (150,151)
     */
    template <typename _EigenDerived>
    Eigen::Matrix<Scalar, 3, 1> Act(
            const Eigen::MatrixBase<_EigenDerived>& v,
            tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_m = {},
            tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_v = {})
            const;

    /**
     * @brief Get the adjoint of SO3 at this.
     * @note See Eq. (139).
     */
    Jacobian Adj() const;

    // SO3 specific functions

    /**
     * @brief Get the transformation matrix (3D isometry).
     * @note T = | R 0 |
     *           | 0 1 |
     */
    Transformation GetTransform() const;

    //! @brief Get a rotation matrix.
    Rotation GetRotation() const;

    //! @brief Get the x component of the quaternion.
    Scalar X() const;
    //! @brief Get the y component of the quaternion.
    Scalar Y() const;
    //! @brief Get the z component of the quaternion.
    Scalar Z() const;
    //! @brief Get the w component of the quaternion.
    Scalar W() const;

    //! @brief Get quaternion.
    QuaternionDataType GetQuat() const;

    /**
     * @brief Normalize the underlying quaternion.
     */
    void Normalize();

    /**
     * @brief Set the rotational as a quaternion.
     * @param quaternion a unitary quaternion
     */
    void SetQuat(const QuaternionDataType& quaternion);

    /**
     * @brief Set the rotational as a quaternion.
     * @param quaternion an Eigen::Vector representing a unitary quaternion
     */
    template <typename _EigenDerived>
    void SetQuat(const Eigen::MatrixBase<_EigenDerived>& quaternion);
};

template <typename _Derived>
typename SO3Base<_Derived>::Transformation SO3Base<_Derived>::GetTransform()
        const {
    Transformation T = Transformation::Identity();
    T.template topLeftCorner<3, 3>() = GetRotation();
    return T;
}

template <typename _Derived>
typename SO3Base<_Derived>::Rotation SO3Base<_Derived>::GetRotation() const {
    return GetQuat().matrix();
}

template <typename _Derived>
typename SO3Base<_Derived>::LieGroup SO3Base<_Derived>::Inverse(
        OptJacobianRef J_minv_m) const {
    if (J_minv_m) {
        *J_minv_m = -GetRotation();
    }

    /// @todo, conjugate doc :
    /// equal to the multiplicative inverse if
    /// the quaternion is normalized
    return LieGroup(GetQuat().conjugate());
}

template <typename _Derived>
typename SO3Base<_Derived>::Tangent SO3Base<_Derived>::Log(
        OptJacobianRef J_t_m) const {
    using std::atan2;
    using std::sqrt;

    Tangent tan;
    Scalar log_coeff;

    const Scalar sin_angle_squared = Coeffs().template head<3>().squaredNorm();
    if (sin_angle_squared > Constants<Scalar>::eps) {
        const Scalar sin_angle = sqrt(sin_angle_squared);
        const Scalar cos_angle = W();

        /** @note If (cos_angle < 0) then angle >= pi/2 ,
         *  means : angle for angle_axis vector >= pi (== 2*angle)
         *   |-> results in correct rotation but not a normalized angle_axis
         * vector
         *
         * In that case we observe that 2 * angle ~ 2 * angle - 2 * pi,
         * which is equivalent saying
         *
         * angle - pi = atan(sin(angle - pi), cos(angle - pi))
         *            = atan(-sin(angle), -cos(angle))
         */
        const Scalar two_angle =
                Scalar(2.0) * ((cos_angle < Scalar(0.0))
                                       ? Scalar(atan2(-sin_angle, -cos_angle))
                                       : Scalar(atan2(sin_angle, cos_angle)));

        log_coeff = two_angle / sin_angle;
    } else {
        // small-angle approximation
        log_coeff = Scalar(2.0);
    }

    tan = Tangent(Coeffs().template head<3>() * log_coeff);

    //  using std::atan2;
    //  Scalar n = Coeffs().template head<3>().norm();
    //  Scalar angle(0);
    //  typename Tangent::DataType axis(1,0,0);
    //  if (n<Constants<Scalar>::eps)
    //    n = Coeffs().template head<3>().stableNorm();
    //  if (n > Scalar(0))
    //  {
    //    angle = Scalar(2)*atan2(n, W());
    //    axis  = Coeffs().template head<3>() / n;
    //  }

    //  tan = Tangent(axis*angle);

    if (J_t_m) {
        J_t_m->setIdentity();
        J_t_m->noalias() += Scalar(0.5) * tan.Hat();
        Scalar theta2 = tan.Coeffs().squaredNorm();
        if (theta2 > Constants<Scalar>::eps) {
            Scalar theta = sqrt(theta2);  // rotation angle
            J_t_m->noalias() += (Scalar(1) / theta2 -
                                 (Scalar(1) + cos(theta)) /
                                         (Scalar(2) * theta * sin(theta))) *
                                tan.Hat() * tan.Hat();
        }
    }

    return tan;
}

template <typename _Derived>
typename SO3Base<_Derived>::Tangent SO3Base<_Derived>::Lift(
        OptJacobianRef J_t_m) const {
    return Log(J_t_m);
}

template <typename _Derived>
template <typename _DerivedOther>
typename SO3Base<_Derived>::LieGroup SO3Base<_Derived>::Compose(
        const LieGroupBase<_DerivedOther>& m,
        OptJacobianRef J_mc_ma,
        OptJacobianRef J_mc_mb) const {
    using std::abs;

    static_assert(std::is_base_of<SO3Base<_DerivedOther>, _DerivedOther>::value,
                  "Argument does not inherit from S03Base !");

    const auto& m_SO3 = static_cast<const SO3Base<_DerivedOther>&>(m);

    if (J_mc_ma) {
        *J_mc_ma = m_SO3.GetRotation().transpose();
    }

    if (J_mc_mb) J_mc_mb->setIdentity();

    QuaternionDataType ret_q = GetQuat() * m_SO3.GetQuat();

    const Scalar ret_sqnorm = ret_q.squaredNorm();

    if (abs(ret_sqnorm - Scalar(1)) > Constants<Scalar>::eps) {
        ret_q.coeffs() *= ApproxSqrtInv(ret_sqnorm);
    }

    return LieGroup(ret_q);
}

template <typename _Derived>
template <typename _EigenDerived>
Eigen::Matrix<typename SO3Base<_Derived>::Scalar, 3, 1> SO3Base<_Derived>::Act(
        const Eigen::MatrixBase<_EigenDerived>& v,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_m,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_v) const {
    assert_vector_dim(v, 3);
    const Rotation R(GetRotation());

    if (J_vout_m) {
        J_vout_m->noalias() = -R * skew(v);
    }

    if (J_vout_v) {
        (*J_vout_v) = R;
    }

    return R * v;
}

template <typename _Derived>
typename SO3Base<_Derived>::Jacobian SO3Base<_Derived>::Adj() const {
    return GetRotation();
}

// SO3 specific

template <typename _Derived>
typename SO3Base<_Derived>::Scalar SO3Base<_Derived>::X() const {
    return Coeffs().x();
}

template <typename _Derived>
typename SO3Base<_Derived>::Scalar SO3Base<_Derived>::Y() const {
    return Coeffs().y();
}

template <typename _Derived>
typename SO3Base<_Derived>::Scalar SO3Base<_Derived>::Z() const {
    return Coeffs().z();
}

template <typename _Derived>
typename SO3Base<_Derived>::Scalar SO3Base<_Derived>::W() const {
    return Coeffs().w();
}

template <typename _Derived>
typename SO3Base<_Derived>::QuaternionDataType SO3Base<_Derived>::GetQuat()
        const {
    return QuaternionDataType(Coeffs());
}

template <typename _Derived>
void SO3Base<_Derived>::Normalize() {
    Coeffs().normalize();
}

template <typename _Derived>
void SO3Base<_Derived>::SetQuat(const QuaternionDataType& quaternion) {
    SetQuat(quaternion.Coeffs());
}

template <typename _Derived>
template <typename _EigenDerived>
void SO3Base<_Derived>::SetQuat(
        const Eigen::MatrixBase<_EigenDerived>& quaternion) {
    using std::abs;
    assert_vector_dim(quaternion, 4);
    HOLISTIC_MOTION_ASSERT(abs(quaternion.norm() - Scalar(1)) < Constants<Scalar>::eps,
                "The quaternion is not normalized !", invalid_argument);

    Coeffs() = quaternion;
}

namespace internal {

//! @brief Random specialization for SO3Base objects.
template <typename Derived>
struct RandomEvaluatorImpl<SO3Base<Derived>> {
    template <typename T>
    static void run(T& m) {
        using Scalar = typename SO3Base<Derived>::Scalar;
        using LieGroup = typename SO3Base<Derived>::LieGroup;

        m = LieGroup(randQuat<Scalar>());
    }
};

//! @brief Assignment assert specialization for SE2Base objects
template <typename Derived>
struct AssignmentEvaluatorImpl<SO3Base<Derived>> {
    template <typename T>
    static void run_impl(const T& data) {
        using std::abs;
        HOLISTIC_MOTION_ASSERT(abs(data.norm() - typename SO3Base<Derived>::Scalar(1)) <
                            Constants<typename SO3Base<Derived>::Scalar>::eps,
                    "SO3 assigned data not normalized !",
                    holistic_motion::robotics::invalid_argument);
        HOLISTIC_MOTION_UNUSED_VARIABLE(data);
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3_BASE_H_ */
