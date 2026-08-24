#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_BASE_H_

#include "holistic_motion/manif/impl/lie_group_base.h"
#include "holistic_motion/manif/impl/rn/Rn_map.h"
#include "holistic_motion/manif/impl/se3/SE3_properties.h"
#include "holistic_motion/manif/impl/so3/SO3_map.h"

namespace holistic_motion {
namespace robotics {
//
// LieGroup
//

/**
 * @brief The base class of the SE3 group.
 * @note See Appendix D of the paper.
 */
template <typename _Derived>
struct SE3Base : LieGroupBase<_Derived> {
private:
    using Base = LieGroupBase<_Derived>;
    using Type = SE3Base<_Derived>;

public:
    HOLISTIC_MOTION_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_AUTO_API
    HOLISTIC_MOTION_INHERIT_GROUP_OPERATOR

    using Base::Coeffs;

    using Rotation = typename internal::traits<_Derived>::Rotation;
    using Translation = typename internal::traits<_Derived>::Translation;
    using Transformation = typename internal::traits<_Derived>::Transformation;
    using Isometry = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

    using QuaternionDataType = Eigen::Quaternion<Scalar>;

    // LieGroup common API

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SE3Base)

public:
    HOLISTIC_MOTION_GROUP_ML_ASSIGN_OP(SE3Base)

    /**
     * @brief Get the inverse.
     * @param[out] -optional- J_minv_m Jacobian of the inverse wrt this.
     * @note See Eqs. (170,176).
     */
    LieGroup Inverse(OptJacobianRef J_minv_m = {}) const;

    /**
     * @brief Get the SE3 corresponding Lie algebra element in vector form.
     * @param[out] -optional- J_t_m Jacobian of the tangent wrt to this.
     * @return The SE3 tangent of this.
     * @note This is the log() map in vector form.
     * @note See Eq. (173) & Eq. (79,179,180) and following notes.
     * @see SE3Tangent.
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
     * @brief Composition of this and another SE3 element.
     * @param[in] m Another SE3 element.
     * @param[out] -optional- J_mc_ma Jacobian of the composition wrt this.
     * @param[out] -optional- J_mc_mb Jacobian of the composition wrt m.
     * @return The composition of 'this . m'.
     * @note See Eq. (171) and Eqs. (177,178).
     */
    template <typename _DerivedOther>
    LieGroup Compose(const LieGroupBase<_DerivedOther>& m,
                     OptJacobianRef J_mc_ma = {},
                     OptJacobianRef J_mc_mb = {}) const;

    /**
     * @brief Rigid motion action on a 3D point.
     * @param  v A 3D point.
     * @param[out] -optional- J_vout_m The Jacobian of the new object wrt this.
     * @param[out] -optional- J_vout_v The Jacobian of the new object wrt input
     * object.
     * @return The transformed 3D point.
     * @note See Eq. (181) & Eqs. (182,183).
     */
    template <typename _EigenDerived>
    Eigen::Matrix<Scalar, 3, 1> Act(
            const Eigen::MatrixBase<_EigenDerived>& v,
            tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 6>>> J_vout_m = {},
            tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_v = {})
            const;

    /**
     * @brief Get the adjoint matrix of SE3 at this.
     * @note See Eq. (175).
     */
    Jacobian Adj() const;

    // SE3 specific functions

    /**
     * Get the transformation matrix (3D isometry).
     * @note T = | R t |
     *           | 0 1 |
     */
    Transformation GetTransform() const;

    /**
     * Get the isometry object (Eigen 3D isometry).
     * @note T = | R t |
     *           | 0 1 |
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

    // Scalar roll() const;
    // Scalar pitch() const;
    // Scalar yaw() const;

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

    /**
     * @brief Set the rotational as a SO3 object.
     * @param so3 a holistic_motion::robotics::SO3 object
     */
    void SetQuat(const SO3<Scalar>& so3);

    /**
     * @brief Set the translation of the SE3 object
     * @param translation, 3d-vector representing the translation
     */
    void SetTranslation(const Translation& translation);

public:  /// @todo make protected
    Eigen::Map<const Rn<Scalar, 3>> AsR3() const {
        return Eigen::Map<const Rn<Scalar, 3>>(Coeffs().data());
    }

    Eigen::Map<Rn<Scalar, 3>> AsR3() {
        return Eigen::Map<Rn<Scalar, 3>>(Coeffs().data());
    }

    Eigen::Map<const SO3<Scalar>> AsSO3() const {
        return Eigen::Map<const SO3<Scalar>>(Coeffs().data() + 3);
    }

    Eigen::Map<SO3<Scalar>> AsSO3() {
        return Eigen::Map<SO3<Scalar>>(Coeffs().data() + 3);
    }
};

template <typename _Derived>
typename SE3Base<_Derived>::Transformation SE3Base<_Derived>::GetTransform()
        const {
    Transformation T = Transformation::Identity();
    T.template topLeftCorner<3, 3>() = GetRotation();
    T.template topRightCorner<3, 1>() = GetTranslation();
    return T;
}

template <typename _Derived>
typename SE3Base<_Derived>::Isometry SE3Base<_Derived>::isometry() const {
    return Isometry(GetTransform());
}

template <typename _Derived>
typename SE3Base<_Derived>::Rotation SE3Base<_Derived>::GetRotation() const {
    return AsSO3().GetRotation();
}

template <typename _Derived>
typename SE3Base<_Derived>::QuaternionDataType SE3Base<_Derived>::GetQuat()
        const {
    return AsSO3().GetQuat();
}

template <typename _Derived>
typename SE3Base<_Derived>::Translation SE3Base<_Derived>::GetTranslation()
        const {
    return Coeffs().template head<3>();
}

template <typename _Derived>
void SE3Base<_Derived>::SetQuat(const QuaternionDataType& quaternion) {
    SetQuat(quaternion.coeffs());
}

template <typename _Derived>
template <typename _EigenDerived>
void SE3Base<_Derived>::SetQuat(
        const Eigen::MatrixBase<_EigenDerived>& quaternion) {
    using std::abs;
    assert_vector_dim(quaternion, 4);
    HOLISTIC_MOTION_ASSERT(abs(quaternion.norm() - Scalar(1)) < Constants<Scalar>::eps,
                "The quaternion is not normalized !", invalid_argument);

    AsSO3().Coeffs() = quaternion;
}

template <typename _Derived>
void SE3Base<_Derived>::SetQuat(const SO3<Scalar>& so3) {
    SetQuat(so3.Coeffs());
}

template <typename _Derived>
void SE3Base<_Derived>::SetTranslation(const Translation& translation) {
    Coeffs().template head<3>() = translation;
}

template <typename _Derived>
typename SE3Base<_Derived>::LieGroup SE3Base<_Derived>::Inverse(
        OptJacobianRef J_minv_m) const {
    if (J_minv_m) {
        (*J_minv_m) = -Adj();
    }

    const SO3<Scalar> so3inv = AsSO3().Inverse();

    return LieGroup(-so3inv.Act(GetTranslation()), so3inv);
}

template <typename _Derived>
typename SE3Base<_Derived>::Tangent SE3Base<_Derived>::Log(
        OptJacobianRef J_t_m) const {
    using std::abs;
    using std::sqrt;

    const SO3Tangent<Scalar> so3tan = AsSO3().Log();

    Tangent tan((typename Tangent::DataType()
                         << so3tan.Ljacinv() * GetTranslation(),
                 so3tan.Coeffs())
                        .finished());

    if (J_t_m) {
        // Jr^-1
        (*J_t_m) = tan.Rjacinv();
    }

    return tan;
}

template <typename _Derived>
typename SE3Base<_Derived>::Tangent SE3Base<_Derived>::Lift(
        OptJacobianRef J_t_m) const {
    return Log(J_t_m);
}

template <typename _Derived>
template <typename _DerivedOther>
typename SE3Base<_Derived>::LieGroup SE3Base<_Derived>::Compose(
        const LieGroupBase<_DerivedOther>& m,
        OptJacobianRef J_mc_ma,
        OptJacobianRef J_mc_mb) const {
    static_assert(std::is_base_of<SE3Base<_DerivedOther>, _DerivedOther>::value,
                  "Argument does not inherit from SE3Base !");

    const auto& m_se3 = static_cast<const SE3Base<_DerivedOther>&>(m);

    if (J_mc_ma) {
        (*J_mc_ma) = m.Inverse().Adj();
    }

    if (J_mc_mb) {
        J_mc_mb->setIdentity();
    }

    return LieGroup(GetRotation() * m_se3.GetTranslation() + GetTranslation(),
                    AsSO3().Compose(m_se3.AsSO3()).GetQuat());
}

template <typename _Derived>
template <typename _EigenDerived>
Eigen::Matrix<typename SE3Base<_Derived>::Scalar, 3, 1> SE3Base<_Derived>::Act(
        const Eigen::MatrixBase<_EigenDerived>& v,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 6>>> J_vout_m,
        tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_v) const {
    assert_vector_dim(v, 3);
    const Rotation R(GetRotation());

    if (J_vout_m) {
        J_vout_m->template topLeftCorner<3, 3>() = R;
        J_vout_m->template topRightCorner<3, 3>() = -R * skew(v);
    }

    if (J_vout_v) {
        (*J_vout_v) = R;
    }

    return GetTranslation() + R * v;
}

template <typename _Derived>
typename SE3Base<_Derived>::Jacobian SE3Base<_Derived>::Adj() const {
    /// @note Chirikjian (close to Eq.10.94)
    /// says
    ///       Ad(g) = |  R  0 |
    ///               | T.R R |
    ///
    /// considering vee(log(g)) = (w;v)
    /// with T = [t]_x
    ///
    /// but this is
    ///       Ad(g) = | R T.R |
    ///               | 0  R  |
    ///
    /// considering vee(log(g)) = (v;w)

    Jacobian Adj;
    Adj.template topLeftCorner<3, 3>() = GetRotation();
    Adj.template bottomRightCorner<3, 3>() = Adj.template topLeftCorner<3, 3>();
    Adj.template topRightCorner<3, 3>().noalias() =
            skew(GetTranslation()) * Adj.template topLeftCorner<3, 3>();
    Adj.template bottomLeftCorner<3, 3>().setZero();

    return Adj;
}

// SE3 specific function

template <typename _Derived>
typename SE3Base<_Derived>::Scalar SE3Base<_Derived>::x() const {
    return Coeffs().x();
}

template <typename _Derived>
typename SE3Base<_Derived>::Scalar SE3Base<_Derived>::y() const {
    return Coeffs().y();
}

template <typename _Derived>
typename SE3Base<_Derived>::Scalar SE3Base<_Derived>::z() const {
    return Coeffs().z();
}

template <typename _Derived>
void SE3Base<_Derived>::Normalize() {
    Coeffs().template tail<4>().normalize();
}

namespace internal {

//! @brief Random specialization for SE3Base objects.
template <typename Derived>
struct RandomEvaluatorImpl<SE3Base<Derived>> {
    template <typename T>
    static void run(T& m) {
        using Scalar = typename SE3Base<Derived>::Scalar;
        using Translation = typename SE3Base<Derived>::Translation;
        using LieGroup = typename SE3Base<Derived>::LieGroup;

        m = LieGroup(Translation::Random(), randQuat<Scalar>());
    }
};

//! @brief Assignment assert specialization for SE2Base objects
template <typename Derived>
struct AssignmentEvaluatorImpl<SE3Base<Derived>> {
    template <typename T>
    static void run_impl(const T& data) {
        using std::abs;
        HOLISTIC_MOTION_ASSERT(abs(data.template tail<4>().norm() -
                        typename SE3Base<Derived>::Scalar(1)) <
                            Constants<typename SE3Base<Derived>::Scalar>::eps,
                    "SE3 assigned data not normalized !",
                    holistic_motion::robotics::invalid_argument);
        HOLISTIC_MOTION_UNUSED_VARIABLE(data);
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_BASE_H_ */
