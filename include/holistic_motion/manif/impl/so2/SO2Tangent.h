#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2TANGENT_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2TANGENT_H_

#include "holistic_motion/manif/impl/so2/SO2Tangent_base.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! Traits specialization
template <typename _Scalar>
struct traits<SO2Tangent<_Scalar>> {
    using Scalar = _Scalar;

    using LieGroup = SO2<_Scalar>;
    using Tangent = SO2Tangent<_Scalar>;

    using Base = SO2TangentBase<Tangent>;

    static constexpr int Dim = LieGroupProperties<Base>::Dim;
    static constexpr int DoF = LieGroupProperties<Base>::DoF;
    static constexpr int RepSize = DoF;

    using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

    using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
    using LieAlg = Eigen::Matrix<Scalar, 2, 2>;
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
 * @brief Represents an element of tangent space of SO2.
 */
template <typename _Scalar>
struct SO2Tangent : SO2TangentBase<SO2Tangent<_Scalar>> {
private:
    using Base = SO2TangentBase<SO2Tangent<_Scalar>>;
    using Type = SO2Tangent<_Scalar>;

protected:
    using Base::Derived;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    SO2Tangent() = default;
    ~SO2Tangent() = default;

    HOLISTIC_MOTION_COPY_CONSTRUCTOR(SO2Tangent)
    HOLISTIC_MOTION_MOVE_CONSTRUCTOR(SO2Tangent)

    // Copy constructor given base
    template <typename _DerivedOther>
    SO2Tangent(const TangentBase<_DerivedOther>& o);

    HOLISTIC_MOTION_TANGENT_ASSIGN_OP(SO2Tangent)

    //! @brief Constructor given an angle (rad.).
    SO2Tangent(const Scalar theta);

    // Tangent common API

    DataType& Coeffs();
    const DataType& Coeffs() const;

    // SO2Tangent specific API

    using Base::angle;

protected:
    DataType data_;
};

HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(SO2Tangent);

template <typename _Scalar>
template <typename _DerivedOther>
SO2Tangent<_Scalar>::SO2Tangent(const TangentBase<_DerivedOther>& o)
    : data_(o.Coeffs()) {
    //
}

template <typename _Scalar>
SO2Tangent<_Scalar>::SO2Tangent(const Scalar theta) : data_(theta) {
    //
}

template <typename _Scalar>
typename SO2Tangent<_Scalar>::DataType& SO2Tangent<_Scalar>::Coeffs() {
    return data_;
}

template <typename _Scalar>
const typename SO2Tangent<_Scalar>::DataType& SO2Tangent<_Scalar>::Coeffs()
        const {
    return data_;
}
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2TANGENT_H_ */
