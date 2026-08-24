#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_MAP_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_MAP_H_

#include "holistic_motion/manif/impl/so2/SO2.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename _Scalar>
struct traits<Eigen::Map<SO2<_Scalar>, 0>> : public traits<SO2<_Scalar>> {
    using typename traits<SO2<_Scalar>>::Scalar;
    using traits<SO2<Scalar>>::RepSize;
    using Base = SO2Base<Eigen::Map<SO2<Scalar>, 0>>;
    using DataType = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template <typename _Scalar>
struct traits<Eigen::Map<const SO2<_Scalar>, 0>>
    : public traits<const SO2<_Scalar>> {
    using typename traits<const SO2<_Scalar>>::Scalar;
    using traits<const SO2<Scalar>>::RepSize;
    using Base = SO2Base<Eigen::Map<const SO2<Scalar>, 0>>;
    using DataType = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace Eigen {

/**
 * @brief Specialization of Map for holistic_motion::robotics::SO2
 */
template <class _Scalar>
class Map<holistic_motion::robotics::SO2<_Scalar>, 0>
    : public holistic_motion::robotics::SO2Base<Map<holistic_motion::robotics::SO2<_Scalar>, 0>> {
    using Base = holistic_motion::robotics::SO2Base<Map<holistic_motion::robotics::SO2<_Scalar>, 0>>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API
    using Base::GetRotation;
    using Base::GetTransform;

    Map(Scalar* coeffs) : data_(coeffs) {}

    HOLISTIC_MOTION_GROUP_MAP_ASSIGN_OP(SO2)

    DataType& Coeffs() { return data_; }
    const DataType& Coeffs() const { return data_; }

protected:
    DataType data_;
};

/**
 * @brief Specialization of Map for const holistic_motion::robotics::SO2
 */
template <class _Scalar>
class Map<const holistic_motion::robotics::SO2<_Scalar>, 0>
    : public holistic_motion::robotics::SO2Base<
              Map<const holistic_motion::robotics::SO2<_Scalar>, 0>> {
    using Base =
            holistic_motion::robotics::SO2Base<Map<const holistic_motion::robotics::SO2<_Scalar>, 0>>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API
    using Base::GetRotation;
    using Base::GetTransform;

    Map(const Scalar* coeffs) : data_(coeffs) {}

    const DataType& Coeffs() const { return data_; }

protected:
    const DataType data_;
};

}  // namespace Eigen

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_MAP_H_ */
