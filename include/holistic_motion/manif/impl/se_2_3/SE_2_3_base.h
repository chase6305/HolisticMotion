#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_BASE_H_

#include "holistic_motion/manif/impl/lie_group_base.h"
#include "holistic_motion/manif/impl/se3/SE3_map.h"
#include "holistic_motion/manif/impl/se_2_3/SE_2_3_properties.h"
#include "holistic_motion/manif/impl/so3/SO3_map.h"

namespace holistic_motion {
namespace robotics {
//
// LieGroup
//

/**
 * @brief The base class of the SE_2_3 group.
 * @note See Appendix A2 in the paper "The Invariant Extended Kalman filter as a
stable observer".
 * However, note that the serialization used in that paper is different from
that defined below
 * The paper uses a SE_2_3 definition as,
 *  X = |R v p|
 *      |  1  |
 *      |    1|
 * with a vector space serialization as (w, a, v)
 * Instead, here we define the SE_2_3 to be,
 *  X = |R p v|
 *      |  1  |
 *      |    1|
 * with a vector space serialization as (v, w, a)
 */
template <typename _Derived>
struct SE_2_3Base : LieGroupBase<_Derived> {
private:
    using Base = LieGroupBase<_Derived>;
    using Type = SE_2_3Base<_Derived>;

public:
    HOLISTIC_MOTION_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_AUTO_API
    HOLISTIC_MOTION_INHERIT_GROUP_OPERATOR

    using Base::Coeffs;

    using Rotation = typename internal::traits<_Derived>::Rotation;
    using Translation = typename internal::traits<_Derived>::Translation;
    using LinearVelocity = typename internal::traits<_Derived>::LinearVelocity;
    using Transformation = Eigen::Matrix<Scalar, 5, 5>;
    using Isometry =
            Eigen::Matrix<Scalar, 5, 5>; /**< Double direct spatial isometry*/
    using QuaternionDataType = Eigen::Quaternion<Scalar>;

    // LieGroup common API

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SE_2_3Base)

public:
    HOLISTIC_MOTION_GROUP_ML_ASSIGN_OP(SE_2_3Base)

    /**
     * @brief Get the inverse.
     * @param[out] -optional- J_minv_m Jacobian of the inverse wrt this.
     */
    LieGroup Inverse(OptJacobianRef J_minv_m = {}) const;

    /**
     * @brief Get the SE_2_3 corresponding Lie algebra element in vector form.
     * @param[out] -optional- J_t_m Jacobian of the tangent wrt to this.
     * @return The SE_2_3 tangent of this.
     * @note This is the log() map in vector form.
     * @see SE_2_3Tangent.
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
     * @brief Composition of this and another SE_2_3 element.
     * @param[in] m Another SE_2_3 element.
     * @param[out] -optional- J_mc_ma Jacobian of the composition wrt this.
     * @param[out] -optional- J_mc_mb Jacobian of the composition wrt m.
     * @return The composition of 'this . m'.
     */
    template <typename _DerivedOther>
    LieGroup Compose(const LieGroupBase<_DerivedOther>& m,
                     OptJacobianRef J_mc_ma = {},
                     OptJacobianRef J_mc_mb = {}) const;

    /**
     * @brief Get the action of the underlying SE(3) element on a 3d point
     * @note this method by default returns a rigid motion action on 3d points
     * and does not take into account the embedded linear velocity of total
     * SE_2(3) element
     * @param[in]  v A 3D point.
     * @param[out] -optional- J_vout_m The Jacobian of the new object wrt this.
     * @param[out] -optional- J_vout_v The Jacobian of the new object wrt input
     * object.
     */
    template <typename _EigenDerived>
    Eigen::Matrix<Scalar, 3, 1> Act(
            const Eigen::MatrixBase<_EigenDerived>& v,
            tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 9>>> J_vout_m = {},
            tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_v = {})
            const;

    /**
     * @brief Get the adjoint matrix of SE_2_3 at this.
     */
    Jacobian Adj() const;

    // SE_2_3 specific functions

    /**
     * Get the isometry object (double direct isometry).
     * @note T = | R t v|
     *           |   1  |
     *           |     1|
     */
    Transformation GetTransform() const;

    /**
     * Get the isometry object (double direct isometry).
     * @note T = | R t v|
     *           |   1  |
     *           |     1|
     */
    Isometry isometry() const;

    /**
     * @brief Get the rotational part of this as a rotation matrix.
     */
    Rotation GetRotation() const;

    /**
     * @brief Get the rotational part of this as a quaternion.
     */
    QuaternionDataType GetQuat() const;

    /**
     * @brief Get the translational part in vector form.
     */
    Translation GetTranslation() const;

    /**
     * @brief Get the x component of the translational part.
     */
    Scalar x() const;

    /**
     * @brief Get the y component of translational part.
     */
    Scalar y() const;

    /**
     * @brief Get the z component of translational part.
     */
    Scalar z() const;

    /**
     * @brief Get the linear velocity part in vector form.
     */
    LinearVelocity linearVelocity() const;

    /**
     * @brief Get the x component of the linear velocity part.
     */
    Scalar vx() const;

    /**
     * @brief Get the y component of linear velocity part.
     */
    Scalar vy() const;

    /**
     * @brief Get the z component of linear velocity part.
     */
    Scalar vz() const;

    /**
     * @brief Normalize the underlying quaternion.
     */
    void Normalize();

public:  /// @todo make protected
    Eigen::Map<const SO3<Scalar>> AsSO3() const {
        return Eigen::Map<const SO3<Scalar>>(Coeffs().data() + 3);
    }

    Eigen::Map<SO3<Scalar>> AsSO3() {
        return Eigen::Map<SO3<Scalar>>(Coeffs().data() + 3);
    }
};

template <typename _Derived>
typename SE_2_3Base<_Derived>::Transformation
SE_2_3Base<_Derived>::GetTransform() const {
    Eigen::Matrix<Scalar, 5, 5> T;
    T.template topLeftCorner<3, 3>() = GetRotation();
    T.template block<3, 1>(0, 3) = GetTranslation();
    T.template topRightCorner<3, 1>() = linearVelocity();
    T.template bottomLeftCorner<2, 3>().setZero();
    T.template bottomRightCorner<2, 2>().setIdentity();
    return T;
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Isometry SE_2_3Base<_Derived>::isometry() const {
    return Isometry(GetTransform());
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Rotation SE_2_3Base<_Derived>::GetRotation()
        const {
    return AsSO3().GetRotation();
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::QuaternionDataType
SE_2_3Base<_Derived>::GetQuat() const {
    return AsSO3().GetQuat();
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Translation
SE_2_3Base<_Derived>::GetTranslation() const {
    return Coeffs().template head<3>();
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::LinearVelocity
SE_2_3Base<_Derived>::linearVelocity() const {
    return Coeffs().template tail<3>();
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::LieGroup SE_2_3Base<_Derived>::Inverse(
        OptJacobianRef J_minv_m) const {
    if (J_minv_m) {
        (*J_minv_m) = -Adj();
    }

    const SO3<Scalar> so3inv = AsSO3().Inverse();

    return LieGroup(-so3inv.Act(GetTranslation()), so3inv,
                    -so3inv.Act(linearVelocity()));
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Tangent SE_2_3Base<_Derived>::Log(
        OptJacobianRef J_t_m) const {
    const SO3Tangent<Scalar> so3tan = AsSO3().Log();

    Tangent tan((typename Tangent::DataType()
                         << so3tan.Ljacinv() * GetTranslation(),
                 so3tan.Coeffs(), so3tan.Ljacinv() * linearVelocity())
                        .finished());

    if (J_t_m) {
        // Jr^-1
        (*J_t_m) = tan.Rjacinv();
    }

    return tan;
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Tangent SE_2_3Base<_Derived>::Lift(
        OptJacobianRef J_t_m) const {
    return Log(J_t_m);
}

template <typename _Derived>
template <typename _DerivedOther>
typename SE_2_3Base<_Derived>::LieGroup SE_2_3Base<_Derived>::Compose(
        const LieGroupBase<_DerivedOther>& m,
        OptJacobianRef J_mc_ma,
        OptJacobianRef J_mc_mb) const {
    static_assert(
            std::is_base_of<SE_2_3Base<_DerivedOther>, _DerivedOther>::value,
            "Argument does not inherit from SE_2_3Base !");

    const auto& m_se_2_3 = static_cast<const SE_2_3Base<_DerivedOther>&>(m);

    if (J_mc_ma) {
        (*J_mc_ma) = m.Inverse().Adj();
    }

    if (J_mc_mb) {
        J_mc_mb->setIdentity();
    }

    return LieGroup(
            GetRotation() * m_se_2_3.GetTranslation() + GetTranslation(),
            AsSO3().compose(m_se_2_3.AsSO3()).GetQuat(),
            GetRotation() * m_se_2_3.linearVelocity() + linearVelocity());
}

template <typename _Derived>
template <typename _EigenDerived>
Eigen::Matrix<typename SE_2_3Base<_Derived>::Scalar, 3, 1>
SE_2_3Base<_Derived>::Act(
        const Eigen::MatrixBase<_EigenDerived>& v,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 9>>> J_vout_m,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_v) const {
    assert_vector_dim(v, 3);

    const Rotation R(GetRotation());

    if (J_vout_m) {
        J_vout_m->template topLeftCorner<3, 3>() = R;
        J_vout_m->template block<3, 3>(0, 3).noalias() = -R * skew(v);
        J_vout_m->template topRightCorner<3, 3>().setZero();
    }

    if (J_vout_v) {
        (*J_vout_v) = R;
    }

    return GetTranslation() + R * v;
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Jacobian SE_2_3Base<_Derived>::Adj() const {
    ///
    /// this is
    ///       Ad(g) = | R T.R 0|
    ///               | 0  R  0|
    ///               | 0 V.R R|
    ///
    /// considering vee(log(g)) = (v;w;a)
    /// with T = [t]_x
    /// with V = [v]_x

    Jacobian Adj;
    Adj.template topLeftCorner<3, 3>() = GetRotation();
    Adj.template bottomRightCorner<3, 3>() = Adj.template topLeftCorner<3, 3>();
    Adj.template block<3, 3>(3, 3) = Adj.template topLeftCorner<3, 3>();

    Adj.template block<3, 3>(0, 3).noalias() =
            skew(GetTranslation()) * Adj.template topLeftCorner<3, 3>();
    Adj.template block<3, 3>(6, 3).noalias() =
            skew(linearVelocity()) * Adj.template topLeftCorner<3, 3>();

    Adj.template bottomLeftCorner<6, 3>().setZero();
    Adj.template topRightCorner<6, 3>().setZero();

    return Adj;
}

// SE_2_3 specific function

template <typename _Derived>
typename SE_2_3Base<_Derived>::Scalar SE_2_3Base<_Derived>::x() const {
    return Coeffs()(0);
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Scalar SE_2_3Base<_Derived>::y() const {
    return Coeffs()(1);
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Scalar SE_2_3Base<_Derived>::z() const {
    return Coeffs()(2);
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Scalar SE_2_3Base<_Derived>::vx() const {
    return Coeffs()(7);
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Scalar SE_2_3Base<_Derived>::vy() const {
    return Coeffs()(8);
}

template <typename _Derived>
typename SE_2_3Base<_Derived>::Scalar SE_2_3Base<_Derived>::vz() const {
    return Coeffs()(9);
}

template <typename _Derived>
void SE_2_3Base<_Derived>::Normalize() {
    Coeffs().template segment<4>(3).normalize();
}

namespace internal {

//! @brief Random specialization for SE_2_3Base objects.
template <typename Derived>
struct RandomEvaluatorImpl<SE_2_3Base<Derived>> {
    template <typename T>
    static void run(T& m) {
        using Scalar = typename SE_2_3Base<Derived>::Scalar;
        using Translation = typename SE_2_3Base<Derived>::Translation;
        using LinearVelocity = typename SE_2_3Base<Derived>::LinearVelocity;
        using LieGroup = typename SE_2_3Base<Derived>::LieGroup;

        m = LieGroup(Translation::Random(), randQuat<Scalar>(),
                     LinearVelocity::Random());
    }
};

//! @brief Assignment assert specialization for SE2Base objects
template <typename Derived>
struct AssignmentEvaluatorImpl<SE_2_3Base<Derived>> {
    template <typename T>
    static void run_impl(const T& data) {
        using std::abs;
        HOLISTIC_MOTION_ASSERT(
                abs(data.template segment<4>(3).norm() -
                    typename SE_2_3Base<Derived>::Scalar(1)) <
                        Constants<typename SE_2_3Base<Derived>::Scalar>::eps,
                "SE_2_3 assigned data not normalized !",
                holistic_motion::robotics::invalid_argument);
        HOLISTIC_MOTION_UNUSED_VARIABLE(data);
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_BASE_H_ */
