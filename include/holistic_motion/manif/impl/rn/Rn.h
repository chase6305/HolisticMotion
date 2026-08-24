#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_H_

#include "holistic_motion/manif/impl/rn/Rn_base.h"

namespace holistic_motion {
namespace robotics {
// Forward declare for type traits specialization

template <typename _Scalar, unsigned int N>
struct Rn;
template <typename _Scalar, unsigned int N>
struct RnTangent;

namespace internal {

//! Traits specialization
template <typename _Scalar, unsigned int _N>
struct traits<Rn<_Scalar, _N>> {
    using Scalar = _Scalar;

    using LieGroup = Rn<_Scalar, _N>;
    using Tangent = RnTangent<_Scalar, _N>;

    using Base = RnBase<Rn<_Scalar, _N>>;

    static constexpr int Dim = _N;
    static constexpr int DoF = _N;
    static constexpr int RepSize = _N;

    using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

    using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
    using Transformation = Eigen::Matrix<Scalar, DoF, DoF>;
    using Vector = Eigen::Matrix<Scalar, DoF, 1>;
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
 * @brief Represents an element of Rn.
 */
template <typename _Scalar, unsigned int _N>
struct Rn : RnBase<Rn<_Scalar, _N>> {
private:
    static_assert(_N > 0, "N must be greater than 0 !");

    using Base = RnBase<Rn<_Scalar, _N>>;
    using Type = Rn<_Scalar, _N>;

protected:
    using Base::Derived;

public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND

    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API

    Rn() { this->SetZero(); }

    ~Rn() = default;

    HOLISTIC_MOTION_COPY_CONSTRUCTOR(Rn)
    HOLISTIC_MOTION_MOVE_CONSTRUCTOR(Rn)

    // Copy constructor given base
    template <typename _DerivedOther>
    Rn(const LieGroupBase<_DerivedOther>& o);

    HOLISTIC_MOTION_GROUP_ASSIGN_OP(Rn)

    // LieGroup common API

    //! Get a reference to the underlying DataType.
    DataType& Coeffs();

    //! Get a const reference to the underlying DataType.
    const DataType& Coeffs() const;

    // Rn specific API

protected:
    DataType data_;
};

template <typename _Scalar>
using R1 = Rn<_Scalar, 1>;
template <typename _Scalar>
using R2 = Rn<_Scalar, 2>;
template <typename _Scalar>
using R3 = Rn<_Scalar, 3>;
template <typename _Scalar>
using R4 = Rn<_Scalar, 4>;
template <typename _Scalar>
using R5 = Rn<_Scalar, 5>;
template <typename _Scalar>
using R6 = Rn<_Scalar, 6>;
template <typename _Scalar>
using R7 = Rn<_Scalar, 7>;
template <typename _Scalar>
using R8 = Rn<_Scalar, 8>;
template <typename _Scalar>
using R9 = Rn<_Scalar, 9>;
template <typename _Scalar>
using R10 = Rn<_Scalar, 10>;
template <typename _Scalar>
using R11 = Rn<_Scalar, 11>;
template <typename _Scalar>
using R12 = Rn<_Scalar, 12>;
template <typename _Scalar>
using R13 = Rn<_Scalar, 13>;
template <typename _Scalar>
using R14 = Rn<_Scalar, 14>;
template <typename _Scalar>
using R15 = Rn<_Scalar, 15>;
template <typename _Scalar>
using R16 = Rn<_Scalar, 16>;
template <typename _Scalar>
using R17 = Rn<_Scalar, 17>;
template <typename _Scalar>
using R18 = Rn<_Scalar, 18>;
template <typename _Scalar>
using R19 = Rn<_Scalar, 19>;
template <typename _Scalar>
using R20 = Rn<_Scalar, 20>;
template <typename _Scalar>
using R21 = Rn<_Scalar, 21>;
template <typename _Scalar>
using R22 = Rn<_Scalar, 22>;
template <typename _Scalar>
using R23 = Rn<_Scalar, 23>;
template <typename _Scalar>
using R24 = Rn<_Scalar, 24>;
template <typename _Scalar>
using R25 = Rn<_Scalar, 25>;
template <typename _Scalar>
using R26 = Rn<_Scalar, 26>;
template <typename _Scalar>
using R27 = Rn<_Scalar, 27>;
template <typename _Scalar>
using R28 = Rn<_Scalar, 28>;
template <typename _Scalar>
using R29 = Rn<_Scalar, 29>;
template <typename _Scalar>
using R30 = Rn<_Scalar, 30>;
template <typename _Scalar>
using R31 = Rn<_Scalar, 31>;
template <typename _Scalar>
using R32 = Rn<_Scalar, 32>;

HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R1)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R2)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R3)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R4)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R5)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R6)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R7)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R8)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R9)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R10)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R11)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R12)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R13)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R14)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R15)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R16)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R17)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R18)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R19)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R20)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R21)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R22)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R23)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R24)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R25)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R26)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R27)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R28)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R29)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R30)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R31)
HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(R32)

template <typename _Scalar, unsigned int _N>
template <typename _DerivedOther>
Rn<_Scalar, _N>::Rn(const LieGroupBase<_DerivedOther>& o) : Rn(o.Coeffs()) {
    //
}

template <typename _Scalar, unsigned int _N>
typename Rn<_Scalar, _N>::DataType& Rn<_Scalar, _N>::Coeffs() {
    return data_;
}

template <typename _Scalar, unsigned int _N>
const typename Rn<_Scalar, _N>::DataType& Rn<_Scalar, _N>::Coeffs() const {
    return data_;
}

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_H_ */
