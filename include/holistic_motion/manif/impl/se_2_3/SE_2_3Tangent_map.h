#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3TANGENT_MAP_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3TANGENT_MAP_H_

#include "holistic_motion/manif/impl/se_2_3/SE_2_3Tangent.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename _Scalar>
struct traits<Eigen::Map<SE_2_3Tangent<_Scalar>, 0>>
    : public traits<SE_2_3Tangent<_Scalar>> {
    using typename traits<SE_2_3Tangent<_Scalar>>::Scalar;
    using traits<SE_2_3Tangent<_Scalar>>::DoF;
    using DataType = Eigen::Map<Eigen::Matrix<Scalar, DoF, 1>, 0>;
    using Base = SE_2_3TangentBase<Eigen::Map<SE_2_3Tangent<Scalar>, 0>>;
};

//! @brief traits specialization for Eigen Map
template <typename _Scalar>
struct traits<Eigen::Map<const SE_2_3Tangent<_Scalar>, 0>>
    : public traits<const SE_2_3Tangent<_Scalar>> {
    using typename traits<const SE_2_3Tangent<_Scalar>>::Scalar;
    using traits<const SE_2_3Tangent<_Scalar>>::DoF;
    using DataType = Eigen::Map<const Eigen::Matrix<Scalar, DoF, 1>, 0>;
    using Base = SE_2_3TangentBase<Eigen::Map<const SE_2_3Tangent<Scalar>, 0>>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace Eigen {

/**
 * @brief Specialization of Map for holistic_motion::robotics::SE_2_3
 */
template <class _Scalar>
class Map<holistic_motion::robotics::SE_2_3Tangent<_Scalar>, 0>
    : public holistic_motion::robotics::SE_2_3TangentBase<
              Map<holistic_motion::robotics::SE_2_3Tangent<_Scalar>, 0>> {
    using Base = holistic_motion::robotics::SE_2_3TangentBase<
            Map<holistic_motion::robotics::SE_2_3Tangent<_Scalar>, 0>>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    Map(Scalar* coeffs) : data_(coeffs) {}

    HOLISTIC_MOTION_TANGENT_MAP_ASSIGN_OP(SE_2_3Tangent)

    DataType& Coeffs() { return data_; }
    const DataType& Coeffs() const { return data_; }

protected:
    DataType data_;
};

/**
 * @brief Specialization of Map for const holistic_motion::robotics::SE_2_3
 */
template <class _Scalar>
class Map<const holistic_motion::robotics::SE_2_3Tangent<_Scalar>, 0>
    : public holistic_motion::robotics::SE_2_3TangentBase<
              Map<const holistic_motion::robotics::SE_2_3Tangent<_Scalar>, 0>> {
    using Base = holistic_motion::robotics::SE_2_3TangentBase<
            Map<const holistic_motion::robotics::SE_2_3Tangent<_Scalar>, 0>>;

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

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3TANGENT_MAP_H_ */
