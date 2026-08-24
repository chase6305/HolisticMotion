#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_H_

#include "holistic_motion/manif/impl/rn/RnTangent_base.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! Traits specialization
template <typename _Scalar, unsigned int _N>
struct traits<RnTangent<_Scalar, _N>> {
    using Scalar = _Scalar;

    using LieGroup = Rn<_Scalar, _N>;
    using Tangent = RnTangent<_Scalar, _N>;

    using Base = RnTangentBase<Tangent>;

    static constexpr int Dim = _N;
    static constexpr int DoF = _N;
    static constexpr int RepSize = _N;

    using DataType = Eigen::Matrix<Scalar, DoF, 1>;

    using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
    using LieAlg = Eigen::Matrix<Scalar, DoF + 1, DoF + 1>;
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
 * @brief Represents an element of tangent space of Rn.
 */
template <typename _Scalar, unsigned int _N>
struct RnTangent : RnTangentBase<RnTangent<_Scalar, _N>> {
private:
    static_assert(_N > 0, "N must be greater than 0 !");

    using Base = RnTangentBase<RnTangent<_Scalar, _N>>;
    using Type = RnTangent<_Scalar, _N>;

protected:
    using Base::Derived;

public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND

    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    RnTangent() = default;
    ~RnTangent() = default;

    HOLISTIC_MOTION_COPY_CONSTRUCTOR(RnTangent)
    HOLISTIC_MOTION_MOVE_CONSTRUCTOR(RnTangent)

    // Copy constructor given base
    template <typename _DerivedOther>
    RnTangent(const TangentBase<_DerivedOther>& o);

    HOLISTIC_MOTION_TANGENT_ASSIGN_OP(RnTangent)

    // Tangent common API

    DataType& Coeffs();
    const DataType& Coeffs() const;

protected:
    DataType data_;
};

template <typename _Scalar>
using R1Tangent = RnTangent<_Scalar, 1>;
template <typename _Scalar>
using R2Tangent = RnTangent<_Scalar, 2>;
template <typename _Scalar>
using R3Tangent = RnTangent<_Scalar, 3>;
template <typename _Scalar>
using R4Tangent = RnTangent<_Scalar, 4>;
template <typename _Scalar>
using R5Tangent = RnTangent<_Scalar, 5>;
template <typename _Scalar>
using R6Tangent = RnTangent<_Scalar, 6>;
template <typename _Scalar>
using R7Tangent = RnTangent<_Scalar, 7>;
template <typename _Scalar>
using R8Tangent = RnTangent<_Scalar, 8>;
template <typename _Scalar>
using R9Tangent = RnTangent<_Scalar, 9>;
template <typename _Scalar>
using R10Tangent = RnTangent<_Scalar, 10>;
template <typename _Scalar>
using R11Tangent = RnTangent<_Scalar, 11>;
template <typename _Scalar>
using R12Tangent = RnTangent<_Scalar, 12>;
template <typename _Scalar>
using R13Tangent = RnTangent<_Scalar, 13>;
template <typename _Scalar>
using R14Tangent = RnTangent<_Scalar, 14>;
template <typename _Scalar>
using R15Tangent = RnTangent<_Scalar, 15>;
template <typename _Scalar>
using R16Tangent = RnTangent<_Scalar, 16>;
template <typename _Scalar>
using R17Tangent = RnTangent<_Scalar, 17>;
template <typename _Scalar>
using R18Tangent = RnTangent<_Scalar, 18>;
template <typename _Scalar>
using R19Tangent = RnTangent<_Scalar, 19>;
template <typename _Scalar>
using R20Tangent = RnTangent<_Scalar, 20>;
template <typename _Scalar>
using R21Tangent = RnTangent<_Scalar, 21>;
template <typename _Scalar>
using R22Tangent = RnTangent<_Scalar, 22>;
template <typename _Scalar>
using R23Tangent = RnTangent<_Scalar, 23>;
template <typename _Scalar>
using R24Tangent = RnTangent<_Scalar, 24>;
template <typename _Scalar>
using R25Tangent = RnTangent<_Scalar, 25>;
template <typename _Scalar>
using R26Tangent = RnTangent<_Scalar, 26>;
template <typename _Scalar>
using R27Tangent = RnTangent<_Scalar, 27>;
template <typename _Scalar>
using R28Tangent = RnTangent<_Scalar, 28>;
template <typename _Scalar>
using R29Tangent = RnTangent<_Scalar, 29>;
template <typename _Scalar>
using R30Tangent = RnTangent<_Scalar, 30>;
template <typename _Scalar>
using R31Tangent = RnTangent<_Scalar, 31>;
template <typename _Scalar>
using R32Tangent = RnTangent<_Scalar, 32>;

HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R1Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R2Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R3Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R4Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R5Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R6Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R7Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R8Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R9Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R10Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R11Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R12Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R13Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R14Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R15Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R16Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R17Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R18Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R19Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R20Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R21Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R22Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R23Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R24Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R25Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R26Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R27Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R28Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R29Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R30Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R31Tangent)
HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(R32Tangent)

template <typename _Scalar, unsigned int _N>
template <typename _DerivedOther>
RnTangent<_Scalar, _N>::RnTangent(const TangentBase<_DerivedOther>& o)
    : data_(o.Coeffs()) {
    //
}

template <typename _Scalar, unsigned int _N>
typename RnTangent<_Scalar, _N>::DataType& RnTangent<_Scalar, _N>::Coeffs() {
    return data_;
}

template <typename _Scalar, unsigned int _N>
const typename RnTangent<_Scalar, _N>::DataType&
RnTangent<_Scalar, _N>::Coeffs() const {
    return data_;
}
}  // namespace robotics
}  // namespace holistic_motion

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_H_
