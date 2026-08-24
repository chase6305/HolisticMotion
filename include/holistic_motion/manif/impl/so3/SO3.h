#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3_H_

#include "holistic_motion/manif/impl/so3/SO3_base.h"

namespace holistic_motion {
namespace robotics {

// Forward declare for type traits specialization

template <typename _Scalar>
struct SO3;
template <typename _Scalar>
struct SO3Tangent;

namespace internal {

//! Traits specialization
template <typename _Scalar>
struct traits<SO3<_Scalar>> {
    using Scalar = _Scalar;

    using LieGroup = SO3<_Scalar>;
    using Tangent = SO3Tangent<_Scalar>;

    using Base = SO3Base<SO3<_Scalar>>;

    static constexpr int Dim = LieGroupProperties<Base>::Dim;
    static constexpr int DoF = LieGroupProperties<Base>::DoF;
    static constexpr int RepSize = 4;

    using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

    using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
    using Transformation = Eigen::Matrix<Scalar, 4, 4>;
    using Rotation = Eigen::Matrix<Scalar, Dim, Dim>;
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
 * @brief Represents an element of SO3.
 */
template <typename _Scalar>
struct SO3 : SO3Base<SO3<_Scalar>> {
private:
    using Base = SO3Base<SO3<_Scalar>>;
    using Type = SO3<_Scalar>;

    using QuaternionDataType = Eigen::Quaternion<_Scalar>;

protected:
    using Base::Derived;

public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND

    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API
    using Base::GetQuat;
    using Base::GetRotation;
    using Base::GetTransform;
    using Base::Normalize;

    SO3() { data_ << Scalar(0), Scalar(0), Scalar(0), Scalar(1.); }

    ~SO3() = default;

    HOLISTIC_MOTION_COPY_CONSTRUCTOR(SO3)
    HOLISTIC_MOTION_MOVE_CONSTRUCTOR(SO3)

    // Copy constructor given base
    template <typename _DerivedOther>
    SO3(const LieGroupBase<_DerivedOther>& o);

    HOLISTIC_MOTION_GROUP_ASSIGN_OP(SO3)

    /**
     * @brief Constructor given a unit quaternion.
     * @param[in] q A unit quaternion.
     * @throws holistic_motion::invalid_argument on un-normalized quaternion.
     */
    SO3(const QuaternionDataType& q);

    /**
     * @brief Constructor given the quaternion's coefficients.
     * @param[in] x The x-components of a unit quaternion.
     * @param[in] y The x-components of a unit quaternion.
     * @param[in] z The x-components of a unit quaternion.
     * @param[in] w The x-components of a unit quaternion.
     * @throws holistic_motion::invalid_argument on un-normalized quaternion.
     */
    SO3(const Scalar x, const Scalar y, const Scalar z, const Scalar w);

    /**
     * @brief Constructor given an angle axis.
     */
    SO3(const Eigen::AngleAxis<Scalar>& angle_axis);

    /**
     * @brief Constructor given Euler angles.
     */
    SO3(const Scalar roll, const Scalar pitch, const Scalar yaw);

    /**
     * @brief Constructor given angles and euler type.
     */
    SO3(const Scalar angle1,
        const Scalar angle2,
        const Scalar angle3,
        const EulerAnglesType& type);

    /**
     * @brief Constructor given Transformation matrix.
     */
    SO3(const Eigen::Matrix<Scalar, 3, 3>& mat);

    /**
     * @brief Constructor given Euler angles vector.
     */
    SO3(const Eigen::Matrix<Scalar, 3, 1>& euler_vec);

    /**
     * @brief Constructor given quaternion vector.
     */
    // SO3(const Eigen::Matrix<Scalar, 4, 1>& euler_vec);

    DataType& Coeffs();
    const DataType& Coeffs() const;

protected:
    DataType data_;
};

HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(SO3)

template <typename _Scalar>
template <typename _DerivedOther>
SO3<_Scalar>::SO3(const LieGroupBase<_DerivedOther>& o) : SO3(o.Coeffs()) {
    //
}

template <typename _Scalar>
SO3<_Scalar>::SO3(const QuaternionDataType& q) : SO3(q.coeffs()) {
    //
}

template <typename _Scalar>
SO3<_Scalar>::SO3(const Scalar x,
                  const Scalar y,
                  const Scalar z,
                  const Scalar w)
    : SO3((DataType() << x, y, z, w).finished()) {
    //
}

template <typename _Scalar>
SO3<_Scalar>::SO3(const Eigen::AngleAxis<Scalar>& angle_axis)
    : SO3(QuaternionDataType(angle_axis).coeffs()) {}

template <typename _Scalar>
SO3<_Scalar>::SO3(const Scalar roll, const Scalar pitch, const Scalar yaw)
    : SO3(Eigen::AngleAxis<Scalar>(yaw, Eigen::Matrix<Scalar, 3, 1>::UnitZ()) *
          Eigen::AngleAxis<Scalar>(pitch,
                                   Eigen::Matrix<Scalar, 3, 1>::UnitY()) *
          Eigen::AngleAxis<Scalar>(roll,
                                   Eigen::Matrix<Scalar, 3, 1>::UnitX())) {
    //
}

template <typename _Scalar>
SO3<_Scalar>::SO3(const Scalar angle1,
                  const Scalar angle2,
                  const Scalar angle3,
                  const EulerAnglesType& type) {
    switch (type) {
        case EulerAnglesType::XYZ:
            *this = SO3(RotX(angle1) * RotY(angle2) * RotZ(angle3));
            break;
        case EulerAnglesType::xyz:
            *this = SO3(RotZ(angle3) * RotY(angle2) * RotX(angle1));
            break;
        case EulerAnglesType::ZYX:
            *this = SO3(RotZ(angle1) * RotY(angle2) * RotX(angle3));
            break;
        case EulerAnglesType::zyx:
            *this = SO3(RotX(angle3) * RotY(angle2) * RotZ(angle1));
            break;
        case EulerAnglesType::ZYZ:
            *this = SO3(RotZ(angle1) * RotY(angle2) * RotZ(angle3));
            break;
        case EulerAnglesType::zyz:
            *this = SO3(RotZ(angle3) * RotY(angle2) * RotZ(angle1));
            break;
        case EulerAnglesType::ZXZ:
            *this = SO3(RotZ(angle1) * RotX(angle2) * RotZ(angle3));
            break;
        case EulerAnglesType::zxz:
            *this = SO3(RotZ(angle3) * RotX(angle2) * RotZ(angle1));
            break;
        case EulerAnglesType::YXY:
            *this = SO3(RotY(angle1) * RotX(angle2) * RotY(angle3));
            break;
        case EulerAnglesType::yxy:
            *this = SO3(RotY(angle3) * RotX(angle2) * RotY(angle1));
            break;
        case EulerAnglesType::YZY:
            *this = SO3(RotY(angle1) * RotZ(angle2) * RotY(angle3));
            break;
        case EulerAnglesType::yzy:
            *this = SO3(RotY(angle3) * RotZ(angle2) * RotY(angle1));
            break;
        case EulerAnglesType::XZX:
            *this = SO3(RotX(angle1) * RotZ(angle2) * RotX(angle3));
            break;
        case EulerAnglesType::xzx:
            *this = SO3(RotX(angle3) * RotZ(angle2) * RotX(angle1));
            break;
        case EulerAnglesType::XYX:
            *this = SO3(RotX(angle1) * RotY(angle2) * RotX(angle3));
            break;
        case EulerAnglesType::xyx:
            *this = SO3(RotX(angle3) * RotY(angle2) * RotX(angle1));
            break;
        case EulerAnglesType::rota_vec:
            Eigen::Matrix<Scalar, 3, 1> rotVec(angle1, angle2, angle3);
            *this = SO3(Eigen::AngleAxis<Scalar>(rotVec.norm(),
                                                 rotVec.normalized()));
            break;
    }
}

template <typename _Scalar>
SO3<_Scalar>::SO3(const Eigen::Matrix<Scalar, 3, 3>& mat)
    : SO3(QuaternionDataType(mat).normalized().coeffs()) {}

template <typename _Scalar>
SO3<_Scalar>::SO3(const Eigen::Matrix<Scalar, 3, 1>& euler_vec)
    : SO3(euler_vec[0], euler_vec[1], euler_vec[2]) {
    //
}

// template <typename _Scalar>
// SO3<_Scalar>::SO3(const Eigen::Matrix<Scalar, 4, 1>& euler_vec)
//     : SO3(QuaternionDataType(euler_vec).normalized().Coeffs()) {}

template <typename _Scalar>
typename SO3<_Scalar>::DataType& SO3<_Scalar>::Coeffs() {
    return data_;
}

template <typename _Scalar>
const typename SO3<_Scalar>::DataType& SO3<_Scalar>::Coeffs() const {
    return data_;
}
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3_H_ */
