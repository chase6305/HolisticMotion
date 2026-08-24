#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_PROPERTIES_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_PROPERTIES_H_

#include "holistic_motion/manif/impl/traits.h"

namespace holistic_motion {
namespace robotics {
// Forward declaration
template <typename _Derived>
struct SE3Base;
template <typename _Derived>
struct SE3TangentBase;

namespace internal {

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<SE3Base<_Derived>> {
    static constexpr int Dim = 3;  /// @brief Space dimension
    static constexpr int DoF = 6;  /// @brief Degrees of freedom
};

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<SE3TangentBase<_Derived>> {
    static constexpr int Dim = 3;  /// @brief Space dimension
    static constexpr int DoF = 6;  /// @brief Degrees of freedom
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_PROPERTIES_H_ */
