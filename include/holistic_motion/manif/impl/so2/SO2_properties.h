#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_PROPERTIES_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_PROPERTIES_H_

#include "holistic_motion/manif/impl/traits.h"

namespace holistic_motion {
namespace robotics {
// Forward declaration
template <typename _Derived>
struct SO2Base;
template <typename _Derived>
struct SO2TangentBase;

namespace internal {

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<SO2Base<_Derived>> {
    static constexpr int Dim = 2;  /// @brief Space dimension
    static constexpr int DoF = 1;  /// @brief Degrees of freedom
};

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<SO2TangentBase<_Derived>> {
    static constexpr int Dim = 2;  /// @brief Space dimension
    static constexpr int DoF = 1;  /// @brief Degrees of freedom
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_PROPERTIES_H_ */
