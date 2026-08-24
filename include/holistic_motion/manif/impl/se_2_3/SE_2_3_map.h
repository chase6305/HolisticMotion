#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_MAP_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_MAP_H_

#include "holistic_motion/manif/impl/se_2_3/SE_2_3.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename _Scalar>
struct traits<Eigen::Map<SE_2_3<_Scalar>, 0>> : public traits<SE_2_3<_Scalar>> {
    using typename traits<SE_2_3<_Scalar>>::Scalar;
    using traits<SE_2_3<Scalar>>::RepSize;
    using Base = SE_2_3Base<Eigen::Map<SE_2_3<Scalar>, 0>>;
    using DataType = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template <typename _Scalar>
struct traits<Eigen::Map<const SE_2_3<_Scalar>, 0>>
    : public traits<const SE_2_3<_Scalar>> {
    using typename traits<const SE_2_3<_Scalar>>::Scalar;
    using traits<const SE_2_3<Scalar>>::RepSize;
    using Base = SE_2_3Base<Eigen::Map<const SE_2_3<Scalar>, 0>>;
    using DataType = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace Eigen {

/**
 * @brief Specialization of Map for holistic_motion::robotics::SE_2_3
 */
template <class _Scalar>
class Map<holistic_motion::robotics::SE_2_3<_Scalar>, 0>
    : public holistic_motion::robotics::SE_2_3Base<
              Map<holistic_motion::robotics::SE_2_3<_Scalar>, 0>> {
    using Base =
            holistic_motion::robotics::SE_2_3Base<Map<holistic_motion::robotics::SE_2_3<_Scalar>, 0>>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API
    using Base::GetRotation;

    Map(Scalar* coeffs) : data_(coeffs) {}

    HOLISTIC_MOTION_GROUP_MAP_ASSIGN_OP(SE_2_3)

    DataType& Coeffs() { return data_; }
    const DataType& Coeffs() const { return data_; }

protected:
    DataType data_;
};

/**
 * @brief Specialization of Map for const holistic_motion::robotics::SE_2_3
 */
template <class _Scalar>
class Map<const holistic_motion::robotics::SE_2_3<_Scalar>, 0>
    : public holistic_motion::robotics::SE_2_3Base<
              Map<const holistic_motion::robotics::SE_2_3<_Scalar>, 0>> {
    using Base = holistic_motion::robotics::SE_2_3Base<
            Map<const holistic_motion::robotics::SE_2_3<_Scalar>, 0>>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API
    using Base::GetRotation;

    Map(const Scalar* coeffs) : data_(coeffs) {}

    const DataType& Coeffs() const { return data_; }

protected:
    const DataType data_;
};

}  // namespace Eigen

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_MAP_H_ */
