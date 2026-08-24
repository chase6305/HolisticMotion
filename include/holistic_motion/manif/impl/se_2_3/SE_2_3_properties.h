#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_PROPERTIES_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_PROPERTIES_H_

#include "holistic_motion/manif/impl/traits.h"

namespace holistic_motion {
namespace robotics {
// Forward declaration
template <typename _Derived>
struct SE_2_3Base;
template <typename _Derived>
struct SE_2_3TangentBase;

namespace internal {

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<SE_2_3Base<_Derived>> {
    static constexpr int Dim = 3;  /// @brief Space dimension
    static constexpr int DoF = 9;  /// @brief Degrees of freedom
};

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<SE_2_3TangentBase<_Derived>> {
    static constexpr int Dim = 3;  /// @brief Space dimension
    static constexpr int DoF = 9;  /// @brief Degrees of freedom
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_PROPERTIES_H_ */
