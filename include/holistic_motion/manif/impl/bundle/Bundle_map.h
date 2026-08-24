#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLE_MAP_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLE_MAP_H_

#include "holistic_motion/manif/impl/bundle/Bundle.h"

namespace holistic_motion {
namespace robotics {
namespace internal {

/**
 * @brief traits specialization for Eigen Map
 */
template <typename _Scalar, template <typename> class... T>
struct traits<Eigen::Map<Bundle<_Scalar, T...>, 0>>
    : public traits<Bundle<_Scalar, T...>> {
    using typename traits<Bundle<_Scalar, T...>>::Scalar;
    using traits<Bundle<Scalar, T...>>::RepSize;
    using Base = BundleBase<Eigen::Map<Bundle<Scalar, T...>, 0>>;
    using DataType = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

/**
 * @brief traits specialization for Eigen const Map
 */
template <typename _Scalar, template <typename> class... T>
struct traits<Eigen::Map<const Bundle<_Scalar, T...>, 0>>
    : public traits<const Bundle<_Scalar, T...>> {
    using typename traits<const Bundle<_Scalar, T...>>::Scalar;
    using traits<const Bundle<Scalar, T...>>::RepSize;
    using Base = BundleBase<Eigen::Map<const Bundle<Scalar, T...>, 0>>;
    using DataType = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

namespace Eigen {

/**
 * @brief Specialization of Map for holistic_motion::robotics::Bundle
 */
template <class _Scalar, template <typename> class... T>
class Map<holistic_motion::robotics::Bundle<_Scalar, T...>, 0>
    : public holistic_motion::robotics::BundleBase<
              Map<holistic_motion::robotics::Bundle<_Scalar, T...>, 0>> {
    using Base = holistic_motion::robotics::BundleBase<
            Map<holistic_motion::robotics::Bundle<_Scalar, T...>, 0>>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API

    using Base::BundleSize;

    Map(Scalar* coeffs) : data_(coeffs) {}

    HOLISTIC_MOTION_GROUP_MAP_ASSIGN_OP(Bundle)

    DataType& Coeffs() { return data_; }

    const DataType& Coeffs() const { return data_; }

protected:
    DataType data_;
};

/**
 * @brief Specialization of Map for const holistic_motion::robotics::Bundle
 */
template <class _Scalar, template <typename> class... T>
class Map<const holistic_motion::robotics::Bundle<_Scalar, T...>, 0>
    : public holistic_motion::robotics::BundleBase<
              Map<const holistic_motion::robotics::Bundle<_Scalar, T...>, 0>> {
    using Base = holistic_motion::robotics::BundleBase<
            Map<const holistic_motion::robotics::Bundle<_Scalar, T...>, 0>>;

public:
    HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF
    HOLISTIC_MOTION_INHERIT_GROUP_API

    using Base::BundleSize;

    Map(const Scalar* coeffs) : data_(coeffs) {}

    const DataType& Coeffs() const { return data_; }

protected:
    const DataType data_;
};

}  // namespace Eigen

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLE_MAP_H_
