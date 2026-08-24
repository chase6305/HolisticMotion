#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_H_

#include "holistic_motion/manif/impl/se3/SE3_base.h"

namespace holistic_motion {
namespace robotics {
// Forward declare for type traits specialization

template <typename _Scalar>
struct SE3;
template <typename _Scalar>
struct SE3Tangent;

namespace internal {

//! Traits specialization
template <typename _Scalar>
struct traits<SE3<_Scalar>> {
    using Scalar = _Scalar;

    using LieGroup = SE3<_Scalar>;
    using Tangent = SE3Tangent<_Scalar>;

    using Base = SE3Base<SE3<_Scalar>>;

    static constexpr int Dim = LieGroupProperties<Base>::Dim;
    static constexpr int DoF = LieGroupProperties<Base>::DoF;
    static constexpr int RepSize = 7;

    /// @todo would be nice to concat vec3 + quaternion
    using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

    using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
    using Transformation = Eigen::Matrix<Scalar, 4, 4>;
    using Rotation = Eigen::Matrix<Scalar, Dim, Dim>;
    using Translation = Eigen::Matrix<Scalar, Dim, 1>;
    using Vector = Eigen::Matrix<Scalar, Dim, 1>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace holistic_motion {
namespace robotics {
//
// LieGroup
//

/**
 * @brief Represent an element of SE3.
 */
template <typename _Scalar>
struct SE3 : SE3Base<SE3<_Scalar>> {
private:
    using Base = SE3Base<SE3<_Scalar>>;
    using Type = SE3<_Scalar>;

protected:
    using Base::Derived;

public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND

    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    using Translation = typename Base::Translation;
    using Quaternion = Eigen::Quaternion<Scalar>;

    HOLISTIC_MOTION_INHERIT_GROUP_API
    using Base::GetRotation;
    using Base::GetTransform;
    using Base::Normalize;

    SE3() {
        data_ << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                Scalar(0), Scalar(1.);
    }

    ~SE3() = default;

    HOLISTIC_MOTION_COPY_CONSTRUCTOR(SE3)
    HOLISTIC_MOTION_MOVE_CONSTRUCTOR(SE3)

    template <typename _DerivedOther>
    SE3(const LieGroupBase<_DerivedOther>& o);

    HOLISTIC_MOTION_GROUP_ASSIGN_OP(SE3)

    /**
     * @brief Constructor given a translation and a unit quaternion.
     * @param[in] t A translation vector.
     * @param[in] q A unit quaternion.
     * @throws holistic_motion::invalid_argument on un-normalized complex number.
     */
    SE3(const Translation& t, const Eigen::Quaternion<Scalar>& q);

    /**
     * @brief Constructor given a translation and an angle axis.
     * @param[in] t A translation vector.
     * @param[in] angle_axis An angle-axis.
     */
    SE3(const Translation& t, const Eigen::AngleAxis<Scalar>& angle_axis);

    /**
     * @brief Constructor given a translation and SO3 element.
     * @param[in] t A translation vector.
     * @param[in] SO3 An element of SO3.
     */
    SE3(const Translation& t, const SO3<Scalar>& SO3);

    /**
     * @brief Constructor given a SO3 element.
     * @param[in] SO3 An element of SO3.
     */
    SE3(const SO3<Scalar>& SO3);

    /**
     * @brief Constructor given translation components and
     * roll-pitch-yaw angles.
     * @param[in] x The x component of the translation.
     * @param[in] y The y component of the translation.
     * @param[in] z The z component of the translation.
     * @param[in] roll The roll angle.
     * @param[in] pitch The pitch angle.
     * @param[in] yaw The yaw angle.
     */
    SE3(const Scalar x,
        const Scalar y,
        const Scalar z,
        const Scalar roll,
        const Scalar pitch,
        const Scalar yaw);

    /**
     * @brief Constructor given translation components and a unit quaternion.
     *
     * @param[in] x The x component of the translation.
     * @param[in] y The y component of the translation.
     * @param[in] z The z component of the translation.
     * @param[in] q1 x.
     * @param[in] q2 y.
     * @param[in] q3 z.
     * @param[in] q4 w.
     */
    SE3(const Scalar x,
        const Scalar y,
        const Scalar z,
        const Scalar q1,
        const Scalar q2,
        const Scalar q3,
        const Scalar q4);

    /**
     * @brief Constructor given a 4x4 matrix.
     * @param[in] mat 4x4 matrix
     * @throws holistic_motion::invalid_argument on un-normalized complex number.
     */
    SE3(const Eigen::Matrix<Scalar, 4, 4>& mat);

    /**
     * @brief Constructor given a translation vector and 3x3 rotation matrix.
     * @param[in] t A translation vector.
     * @param[in] rot_mat 3x3 rotation matrix
     * @throws holistic_motion::invalid_argument on un-normalized complex number.
     */
    SE3(const Translation& t, const Eigen::Matrix<Scalar, 3, 3>& rot_mat);

    /**
     * @brief Constructor given a 3x3 rotation matrix, translation vector
     * default is (0.0, 0.0, 0.0).
     * @param[in] rot_mat 3x3 rotation matrix
     * @throws holistic_motion::invalid_argument on un-normalized complex number.
     */
    SE3(const Eigen::Matrix<Scalar, 3, 3>& rot_mat);

    /**
     * @brief Constructor from a 3D Eigen::Isometry<Scalar>
     * @param[in] h an isometry object from Eigen
     *
     * Isometry is a typedef from Eigen::Transform, in which the linear part is
     * assumed a rotation matrix. This is used to speed up certain methods of
     * Transform, especially inverse().
     */
    SE3(const Eigen::Transform<_Scalar, 3, Eigen::Isometry>& h);

    // LieGroup common API

    DataType& Coeffs();
    const DataType& Coeffs() const;

    // SE3 specific API

protected:
    DataType data_;
};

HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(SE3)

template <typename _Scalar>
template <typename _DerivedOther>
SE3<_Scalar>::SE3(const LieGroupBase<_DerivedOther>& o) : SE3(o.Coeffs()) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Translation& t, const Eigen::Quaternion<Scalar>& q)
    : SE3((DataType() << t, q.coeffs()).finished()) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Translation& t, const Eigen::AngleAxis<Scalar>& a)
    : SE3(t, Quaternion(a)) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Translation& t, const SO3<Scalar>& so3)
    : SE3(t, so3.GetQuat()) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const SO3<Scalar>& so3)
    : SE3(Translation(.0, .0, .0), so3.GetQuat()) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Scalar x,
                  const Scalar y,
                  const Scalar z,
                  const Scalar roll,
                  const Scalar pitch,
                  const Scalar yaw)
    : SE3(Translation(x, y, z),
          Eigen::Quaternion<Scalar>(
                  Eigen::AngleAxis<Scalar>(
                          yaw, Eigen::Matrix<Scalar, 3, 1>::UnitZ()) *
                  Eigen::AngleAxis<Scalar>(
                          pitch, Eigen::Matrix<Scalar, 3, 1>::UnitY()) *
                  Eigen::AngleAxis<Scalar>(
                          roll, Eigen::Matrix<Scalar, 3, 1>::UnitX()))) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Scalar x,
                  const Scalar y,
                  const Scalar z,
                  const Scalar q1,
                  const Scalar q2,
                  const Scalar q3,
                  const Scalar q4)
    : SE3(Translation(x, y, z), Eigen::Quaternion<Scalar>(q4, q1, q2, q3)) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Eigen::Matrix<Scalar, 4, 4>& mat)
    : SE3(mat.block(0, 3, 3, 1),
          Eigen::Quaternion<Scalar>(
                  Eigen::Matrix<Scalar, 3, 3>(mat.block(0, 0, 3, 3)))
                  .normalized()) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Translation& t,
                  const Eigen::Matrix<Scalar, 3, 3>& rot_mat)
    : SE3(t, Eigen::Quaternion<Scalar>(rot_mat).normalized()) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Eigen::Matrix<Scalar, 3, 3>& rot_mat)
    : SE3(Translation(.0, .0, .0),
          Eigen::Quaternion<Scalar>(rot_mat).normalized()) {
    //
}

template <typename _Scalar>
SE3<_Scalar>::SE3(const Eigen::Transform<_Scalar, 3, Eigen::Isometry>& h)
    : SE3(h.translation(), Eigen::Quaternion<_Scalar>(h.rotation())) {
    //
}

template <typename _Scalar>
typename SE3<_Scalar>::DataType& SE3<_Scalar>::Coeffs() {
    return data_;
}

template <typename _Scalar>
const typename SE3<_Scalar>::DataType& SE3<_Scalar>::Coeffs() const {
    return data_;
}

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_H_ */
