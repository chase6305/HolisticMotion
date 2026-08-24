#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_H_

#include "holistic_motion/manif/impl/so2/SO2_base.h"

namespace holistic_motion {
namespace robotics {
// Forward declare for type traits specialization

template <typename _Scalar>
struct SO2;
template <typename _Scalar>
struct SO2Tangent;

namespace internal {

//! Traits specialization
template <typename _Scalar>
struct traits<SO2<_Scalar>> {
    using Scalar = _Scalar;

    using LieGroup = SO2<_Scalar>;
    using Tangent = SO2Tangent<_Scalar>;

    using Base = SO2Base<SO2<_Scalar>>;

    static constexpr int Dim = LieGroupProperties<Base>::Dim;
    static constexpr int DoF = LieGroupProperties<Base>::DoF;
    static constexpr int RepSize = 2;

    using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

    using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
    using Transformation = Eigen::Matrix<Scalar, 3, 3>;
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
 * @brief Represents an element of SO2.
 */
template <typename _Scalar>
struct SO2 : SO2Base<SO2<_Scalar>> {
private:
    using Base = SO2Base<SO2<_Scalar>>;
    using Type = SO2<_Scalar>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API
    using Base::GetRotation;
    using Base::GetTransform;
    using Base::Normalize;

protected:
    using Base::Derived;

public:
    SO2() { data_ << Scalar(1), Scalar(0); }

    ~SO2() = default;

    HOLISTIC_MOTION_COPY_CONSTRUCTOR(SO2)
    HOLISTIC_MOTION_MOVE_CONSTRUCTOR(SO2)

    // Copy constructor given base
    template <typename _DerivedOther>
    SO2(const LieGroupBase<_DerivedOther>& o);

    HOLISTIC_MOTION_GROUP_ASSIGN_OP(SO2)

    /**
     * @brief Constructor given the real and imaginary part
     * of a unit complex number representing the angle.
     * @param[in] real The real of a unitary complex number.
     * @param[in] imag The imaginary of a unitary complex number.
     * @throws holistic_motion::invalid_argument on un-normalized complex number.
     */
    SO2(const Scalar real, const Scalar imag);

    //! @brief Constructor given an angle (rad.)
    SO2(const Scalar theta);

    // LieGroup common API

    //! Get a const reference to the underlying DataType.
    DataType& Coeffs();
    const DataType& Coeffs() const;

    // SO2 specific API

    using Base::angle;

protected:
    DataType data_;
};

HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(SO2)

template <typename _Scalar>
template <typename _DerivedOther>
SO2<_Scalar>::SO2(const LieGroupBase<_DerivedOther>& o) : SO2(o.Coeffs()) {
    //
}

template <typename _Scalar>
SO2<_Scalar>::SO2(const Scalar real, const Scalar imag)
    : SO2(DataType(real, imag)) {
    //
}

template <typename _Scalar>
SO2<_Scalar>::SO2(const Scalar theta) : SO2(cos(theta), sin(theta)) {
    using std::cos;
    using std::sin;
}

template <typename _Scalar>
typename SO2<_Scalar>::DataType& SO2<_Scalar>::Coeffs() {
    return data_;
}

template <typename _Scalar>
const typename SO2<_Scalar>::DataType& SO2<_Scalar>::Coeffs() const {
    return data_;
}
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_H_ */
