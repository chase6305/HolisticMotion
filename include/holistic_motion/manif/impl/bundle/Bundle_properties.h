#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLE_PROPERTIES_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLE_PROPERTIES_H_

#include "holistic_motion/manif/impl/traits.h"

namespace holistic_motion {
namespace robotics {

// Forward declaration for type traits specialization
template <typename _Derived>
struct BundleBase;
template <typename _Derived>
struct BundleTangentBase;

namespace internal {

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<BundleBase<_Derived>> {
    static constexpr int Dim =
            traits<_Derived>::Dim;  /// @brief Space dimension
    static constexpr int DoF =
            traits<_Derived>::DoF;  /// @brief Degrees of freedom
};

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<BundleTangentBase<_Derived>> {
    static constexpr int Dim =
            traits<_Derived>::Dim;  /// @brief Space dimension
    static constexpr int DoF =
            traits<_Derived>::DoF;  /// @brief Degrees of freedom
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLE_PROPERTIES_H_
