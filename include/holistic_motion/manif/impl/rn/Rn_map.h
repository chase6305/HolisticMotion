#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_MAP_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_MAP_H_

#include "holistic_motion/manif/impl/rn/Rn.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename _Scalar, unsigned int _N>
struct traits<Eigen::Map<Rn<_Scalar, _N>, 0>> : public traits<Rn<_Scalar, _N>> {
    using typename traits<Rn<_Scalar, _N>>::Scalar;
    using traits<Rn<Scalar, _N>>::RepSize;
    using Base = RnBase<Eigen::Map<Rn<Scalar, _N>, 0>>;
    using DataType = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template <typename _Scalar, unsigned int _N>
struct traits<Eigen::Map<const Rn<_Scalar, _N>, 0>>
    : public traits<const Rn<_Scalar, _N>> {
    using typename traits<const Rn<_Scalar, _N>>::Scalar;
    using traits<const Rn<Scalar, _N>>::RepSize;
    using Base = RnBase<Eigen::Map<const Rn<Scalar, _N>, 0>>;
    using DataType = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace Eigen {

/**
 * @brief Specialization of Map for holistic_motion::robotics::Rn
 */
template <class _Scalar, unsigned int _N>
class Map<holistic_motion::robotics::Rn<_Scalar, _N>, 0>
    : public holistic_motion::robotics::RnBase<Map<holistic_motion::robotics::Rn<_Scalar, _N>, 0>> {
    using Base =
            holistic_motion::robotics::RnBase<Map<holistic_motion::robotics::Rn<_Scalar, _N>, 0>>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API

    Map(Scalar* coeffs) : data_(coeffs) {}

    HOLISTIC_MOTION_GROUP_MAP_ASSIGN_OP(Rn)

    DataType& Coeffs() { return data_; }
    const DataType& Coeffs() const { return data_; }

protected:
    DataType data_;
};

/**
 * @brief Specialization of Map for const holistic_motion::robotics::Rn
 */
template <class _Scalar, unsigned int _N>
class Map<const holistic_motion::robotics::Rn<_Scalar, _N>, 0>
    : public holistic_motion::robotics::RnBase<
              Map<const holistic_motion::robotics::Rn<_Scalar, _N>, 0>> {
    using Base = holistic_motion::robotics::RnBase<
            Map<const holistic_motion::robotics::Rn<_Scalar, _N>, 0>>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API

    Map(const Scalar* coeffs) : data_(coeffs) {}

    const DataType& Coeffs() const { return data_; }

protected:
    const DataType data_;
};

}  // namespace Eigen

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_RN_MAP_H_
