#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CONSTANTS_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CONSTANTS_H_

#include <ceres/jet.h>

#include "holistic_motion/manif/constants.h"

namespace holistic_motion {
namespace robotics {
/// @brief Specialize Constants traits
/// for the ceres::Jet type
template <typename _Scalar, int N>
struct Constants<ceres::Jet<_Scalar, N>> {
    static const ceres::Jet<_Scalar, N> eps;
};

template <typename _Scalar, int N>
const ceres::Jet<_Scalar, N> Constants<ceres::Jet<_Scalar, N>>::eps =
        ceres::Jet<_Scalar, N>(Constants<_Scalar>::eps);

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CONSTANTS_H_ */
