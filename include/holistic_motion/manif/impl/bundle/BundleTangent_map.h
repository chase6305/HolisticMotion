#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLETANGENT_MAP_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLETANGENT_MAP_H_

#include "holistic_motion/manif/impl/bundle/BundleTangent.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

/**
 * @brief traits specialization for Eigen Map
 */
template <typename _Scalar, template <typename> class... T>
struct traits<Eigen::Map<BundleTangent<_Scalar, T...>, 0>>
    : public traits<BundleTangent<_Scalar, T...>> {
    using typename traits<BundleTangent<_Scalar, T...>>::Scalar;
    using traits<BundleTangent<Scalar, T...>>::DoF;
    using Base = BundleTangentBase<Eigen::Map<BundleTangent<Scalar, T...>, 0>>;
    using DataType = Eigen::Map<Eigen::Matrix<Scalar, DoF, 1>, 0>;
};

/**
 * @brief traits specialization for Eigen const Map
 */
template <typename _Scalar, template <typename> class... T>
struct traits<Eigen::Map<const BundleTangent<_Scalar, T...>, 0>>
    : public traits<const BundleTangent<_Scalar, T...>> {
    using typename traits<const BundleTangent<_Scalar, T...>>::Scalar;
    using traits<const BundleTangent<Scalar, T...>>::DoF;
    using Base =
            BundleTangentBase<Eigen::Map<const BundleTangent<Scalar, T...>, 0>>;
    using DataType = Eigen::Map<const Eigen::Matrix<Scalar, DoF, 1>, 0>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace Eigen {

/**
 * @brief Specialization of Map for holistic_motion::robotics::Bundle
 */
template <class _Scalar, template <typename> class... T>
class Map<holistic_motion::robotics::BundleTangent<_Scalar, T...>, 0>
    : public holistic_motion::robotics::BundleTangentBase<
              Map<holistic_motion::robotics::BundleTangent<_Scalar, T...>, 0>> {
    using Base = holistic_motion::robotics::BundleTangentBase<
            Map<holistic_motion::robotics::BundleTangent<_Scalar, T...>, 0>>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    using Base::BundleSize;

    Map(Scalar* coeffs) : data_(coeffs) {}

    HOLISTIC_MOTION_TANGENT_MAP_ASSIGN_OP(BundleTangent)

    DataType& Coeffs() { return data_; }

    const DataType& Coeffs() const { return data_; }

protected:
    DataType data_;
};

/**
 * @brief Specialization of Map for const holistic_motion::robotics::BundleTangent
 */
template <class _Scalar, template <typename> class... T>
class Map<const holistic_motion::robotics::BundleTangent<_Scalar, T...>, 0>
    : public holistic_motion::robotics::BundleTangentBase<
              Map<const holistic_motion::robotics::BundleTangent<_Scalar, T...>, 0>> {
    using Base = holistic_motion::robotics::BundleTangentBase<
            Map<const holistic_motion::robotics::BundleTangent<_Scalar, T...>, 0>>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    using Base::BundleSize;

    Map(const Scalar* coeffs) : data_(coeffs) {}

    const DataType& Coeffs() const { return data_; }

protected:
    const DataType data_;
};

}  // namespace Eigen

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLETANGENT_MAP_H_
