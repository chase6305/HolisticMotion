#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CERES_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CERES_H_

#include "holistic_motion/manif/ceres/ceres_utils.h"
#include "holistic_motion/manif/ceres/constants.h"
#include "holistic_motion/manif/ceres/constraint.h"
#include "holistic_motion/manif/ceres/local_parametrization.h"
#include "holistic_motion/manif/ceres/objective.h"

namespace holistic_motion {
namespace robotics {
namespace internal {
struct YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS {};

template <typename _Scalar, int _N>
struct is_ad<ceres::Jet<_Scalar, _N>> : std::integral_constant<bool, true> {};

}  // namespace internal

#ifdef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_H_
using CeresConstraintSO2 = CeresConstraintFunctor<SO2d>;
using CeresLocalParameterizationSO2 = CeresLocalParameterizationFunctor<SO2d>;
using CeresObjectiveSO2 = CeresObjectiveFunctor<SO2d>;
#else
using CeresConstraintSO2 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
using CeresLocalParameterizationSO2 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
using CeresObjectiveSO2 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
#endif

#ifdef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3_H_
using CeresConstraintSO3 = CeresConstraintFunctor<SO3d>;
using CeresLocalParameterizationSO3 = CeresLocalParameterizationFunctor<SO3d>;
using CeresObjectiveSO3 = CeresObjectiveFunctor<SO3d>;
#else
using CeresConstraintSO3 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
using CeresLocalParameterizationSO3 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
using CeresObjectiveSO3 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
#endif

#ifdef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE2_H_
using CeresConstraintSE2 = CeresConstraintFunctor<SE2d>;
using CeresLocalParameterizationSE2 = CeresLocalParameterizationFunctor<SE2d>;
using CeresObjectiveSE2 = CeresObjectiveFunctor<SE2d>;
#else
using CeresConstraintSE2 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
using CeresLocalParameterizationSE2 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
using CeresObjectiveSE2 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
#endif

#ifdef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_H_
using CeresConstraintSE3 = CeresConstraintFunctor<SE3d>;
using CeresLocalParameterizationSE3 = CeresLocalParameterizationFunctor<SE3d>;
using CeresObjectiveSE3 = CeresObjectiveFunctor<SE3d>;
#else
using CeresConstraintSE3 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
using CeresLocalParameterizationSE3 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
using CeresObjectiveSE3 =
        internal::YOU_MUST_INCLUDE_LIEGROUP_BEFORE_CERES_HELPER_HEADERS;
#endif

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_CERES_H_ */
