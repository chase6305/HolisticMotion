#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_MAP_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_MAP_H_

#include "holistic_motion/manif/impl/rn/RnTangent.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename _Scalar, unsigned int _N>
struct traits<Eigen::Map<RnTangent<_Scalar, _N>, 0>>
    : public traits<RnTangent<_Scalar, _N>> {
    using typename traits<RnTangent<_Scalar, _N>>::Scalar;
    using traits<RnTangent<_Scalar, _N>>::DoF;
    using DataType = ::Eigen::Map<Eigen::Matrix<Scalar, DoF, 1>, 0>;
    using Base = RnTangentBase<Eigen::Map<RnTangent<Scalar, _N>, 0>>;
};

//! @brief traits specialization for Eigen Map const
template <typename _Scalar, unsigned int _N>
struct traits<Eigen::Map<const RnTangent<_Scalar, _N>, 0>>
    : public traits<const RnTangent<_Scalar, _N>> {
    using typename traits<const RnTangent<_Scalar, _N>>::Scalar;
    using traits<const RnTangent<_Scalar, _N>>::DoF;
    using DataType = ::Eigen::Map<const Eigen::Matrix<Scalar, DoF, 1>, 0>;
    using Base = RnTangentBase<const Eigen::Map<RnTangent<Scalar, _N>, 0>>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace Eigen {

//! @brief Specialization of Map for holistic_motion::robotics::RnTangent
template <class _Scalar, unsigned int _N>
class Map<holistic_motion::robotics::RnTangent<_Scalar, _N>, 0>
    : public holistic_motion::robotics::RnTangentBase<
              Map<holistic_motion::robotics::RnTangent<_Scalar, _N>, 0>> {
    using Base = holistic_motion::robotics::RnTangentBase<
            Map<holistic_motion::robotics::RnTangent<_Scalar, _N>, 0>>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    Map(Scalar* coeffs) : data_(coeffs) {}

    HOLISTIC_MOTION_TANGENT_MAP_ASSIGN_OP(RnTangent)

    DataType& Coeffs() { return data_; }
    const DataType& Coeffs() const { return data_; }

protected:
    DataType data_;
};

//! @brief Specialization of Map for const holistic_motion::robotics::RnTangent
template <class _Scalar, unsigned int _N>
class Map<const holistic_motion::robotics::RnTangent<_Scalar, _N>, 0>
    : public holistic_motion::robotics::RnTangentBase<
              Map<const holistic_motion::robotics::RnTangent<_Scalar, _N>, 0>> {
    using Base = holistic_motion::robotics::RnTangentBase<
            Map<const holistic_motion::robotics::RnTangent<_Scalar, _N>, 0>>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    Map(const Scalar* coeffs) : data_(coeffs) {}

    const DataType& Coeffs() const { return data_; }

protected:
    const DataType data_;
};

}  // namespace Eigen

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_MAP_H_
