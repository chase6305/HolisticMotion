#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3TANGENT_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3TANGENT_H_

#include "holistic_motion/manif/impl/so3/SO3Tangent_base.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! Traits specialization
template <typename _Scalar>
struct traits<SO3Tangent<_Scalar>> {
    using Scalar = _Scalar;

    using LieGroup = SO3<_Scalar>;
    using Tangent = SO3Tangent<_Scalar>;

    using Base = SO3TangentBase<Tangent>;

    static constexpr int Dim = LieGroupProperties<Base>::Dim;
    static constexpr int DoF = LieGroupProperties<Base>::DoF;
    static constexpr int RepSize = DoF;

    using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

    using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
    using LieAlg = Eigen::Matrix<Scalar, 3, 3>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace holistic_motion {
namespace robotics {

//
// Tangent
//

/**
 * @brief Represents an element of tangent space of SO3.
 */
template <typename _Scalar>
struct SO3Tangent : SO3TangentBase<SO3Tangent<_Scalar>> {
private:
    using Base = SO3TangentBase<SO3Tangent<_Scalar>>;
    using Type = SO3Tangent<_Scalar>;

protected:
    using Base::Derived;

public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND

    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    SO3Tangent() = default;
    ~SO3Tangent() = default;

    HOLISTIC_MOTION_COPY_CONSTRUCTOR(SO3Tangent)
    HOLISTIC_MOTION_MOVE_CONSTRUCTOR(SO3Tangent)

    // Copy constructor given base
    template <typename _DerivedOther>
    SO3Tangent(const TangentBase<_DerivedOther>& o);

    HOLISTIC_MOTION_TANGENT_ASSIGN_OP(SO3Tangent)

    // Tangent common API

    DataType& Coeffs();
    const DataType& Coeffs() const;

    // SO3Tangent specific API

protected:
    DataType data_;
};

HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(SO3Tangent);

template <typename _Scalar>
template <typename _DerivedOther>
SO3Tangent<_Scalar>::SO3Tangent(const TangentBase<_DerivedOther>& o)
    : data_(o.Coeffs()) {
    //
}

template <typename _Scalar>
typename SO3Tangent<_Scalar>::DataType& SO3Tangent<_Scalar>::Coeffs() {
    return data_;
}

template <typename _Scalar>
const typename SO3Tangent<_Scalar>::DataType& SO3Tangent<_Scalar>::Coeffs()
        const {
    return data_;
}

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3TANGENT_H_ */
