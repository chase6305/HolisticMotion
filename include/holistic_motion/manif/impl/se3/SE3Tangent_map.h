#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3TANGENT_MAP_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3TANGENT_MAP_H_

#include "holistic_motion/manif/impl/se3/SE3Tangent.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename _Scalar>
struct traits<Eigen::Map<SE3Tangent<_Scalar>, 0>>
    : public traits<SE3Tangent<_Scalar>> {
    using typename traits<SE3Tangent<_Scalar>>::Scalar;
    using traits<SE3Tangent<_Scalar>>::DoF;
    using DataType = Eigen::Map<Eigen::Matrix<Scalar, DoF, 1>, 0>;
    using Base = SE3TangentBase<Eigen::Map<SE3Tangent<Scalar>, 0>>;
};

//! @brief traits specialization for Eigen Map
template <typename _Scalar>
struct traits<Eigen::Map<const SE3Tangent<_Scalar>, 0>>
    : public traits<const SE3Tangent<_Scalar>> {
    using typename traits<const SE3Tangent<_Scalar>>::Scalar;
    using traits<const SE3Tangent<_Scalar>>::DoF;
    using DataType = Eigen::Map<const Eigen::Matrix<Scalar, DoF, 1>, 0>;
    using Base = SE3TangentBase<Eigen::Map<const SE3Tangent<Scalar>, 0>>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace Eigen {

/**
 * @brief Specialization of Map for holistic_motion::robotics::SE3
 */
template <class _Scalar>
class Map<holistic_motion::robotics::SE3Tangent<_Scalar>, 0>
    : public holistic_motion::robotics::SE3TangentBase<
              Map<holistic_motion::robotics::SE3Tangent<_Scalar>, 0>> {
    using Base = holistic_motion::robotics::SE3TangentBase<
            Map<holistic_motion::robotics::SE3Tangent<_Scalar>, 0>>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    Map(Scalar* coeffs) : data_(coeffs) {}

    HOLISTIC_MOTION_TANGENT_MAP_ASSIGN_OP(SE3Tangent)

    DataType& Coeffs() { return data_; }
    const DataType& Coeffs() const { return data_; }

protected:
    DataType data_;
};

/**
 * @brief Specialization of Map for const holistic_motion::robotics::SE3
 */
template <class _Scalar>
class Map<const holistic_motion::robotics::SE3Tangent<_Scalar>, 0>
    : public holistic_motion::robotics::SE3TangentBase<
              Map<const holistic_motion::robotics::SE3Tangent<_Scalar>, 0>> {
    using Base = holistic_motion::robotics::SE3TangentBase<
            Map<const holistic_motion::robotics::SE3Tangent<_Scalar>, 0>>;

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

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3TANGENT_MAP_H_ */
