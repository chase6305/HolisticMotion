#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3TANGENT_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3TANGENT_H_

#include "holistic_motion/manif/impl/se_2_3/SE_2_3Tangent_base.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! Traits specialization
template <typename _Scalar>
struct traits<SE_2_3Tangent<_Scalar>> {
    using Scalar = _Scalar;

    using LieGroup = SE_2_3<_Scalar>;
    using Tangent = SE_2_3Tangent<_Scalar>;

    using Base = SE_2_3TangentBase<Tangent>;

    static constexpr int Dim = LieGroupProperties<Base>::Dim;
    static constexpr int DoF = LieGroupProperties<Base>::DoF;
    static constexpr int RepSize = DoF;

    using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

    using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
    using LieAlg = Eigen::Matrix<Scalar, 5, 5>;
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
 * @brief Represents an element of tangent space of SE_2_3.
 */
template <typename _Scalar>
struct SE_2_3Tangent : SE_2_3TangentBase<SE_2_3Tangent<_Scalar>> {
private:
    using Base = SE_2_3TangentBase<SE_2_3Tangent<_Scalar>>;
    using Type = SE_2_3Tangent<_Scalar>;

protected:
    using Base::Derived;

public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND

    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    SE_2_3Tangent() = default;
    ~SE_2_3Tangent() = default;

    HOLISTIC_MOTION_COPY_CONSTRUCTOR(SE_2_3Tangent)
    HOLISTIC_MOTION_MOVE_CONSTRUCTOR(SE_2_3Tangent)

    template <typename _DerivedOther>
    SE_2_3Tangent(const TangentBase<_DerivedOther>& o);

    HOLISTIC_MOTION_TANGENT_ASSIGN_OP(SE_2_3Tangent)

    // Tangent common API

    DataType& Coeffs();
    const DataType& Coeffs() const;

    // SE_2_3Tangent specific API

protected:
    DataType data_;
};

HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(SE_2_3Tangent);

template <typename _Scalar>
template <typename _DerivedOther>
SE_2_3Tangent<_Scalar>::SE_2_3Tangent(const TangentBase<_DerivedOther>& o)
    : data_(o.Coeffs()) {
    //
}

template <typename _Scalar>
typename SE_2_3Tangent<_Scalar>::DataType& SE_2_3Tangent<_Scalar>::Coeffs() {
    return data_;
}

template <typename _Scalar>
const typename SE_2_3Tangent<_Scalar>::DataType&
SE_2_3Tangent<_Scalar>::Coeffs() const {
    return data_;
}
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3TANGENT_H_ */
