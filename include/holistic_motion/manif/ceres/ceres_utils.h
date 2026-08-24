#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_UTILS_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_UTILS_H_

#include <ceres/autodiff_cost_function.h>
#include <ceres/autodiff_local_parameterization.h>

#include "holistic_motion/manif/ceres/constraint.h"
#include "holistic_motion/manif/ceres/local_parametrization.h"
#include "holistic_motion/manif/ceres/objective.h"

namespace holistic_motion {
namespace robotics {
/**
 * @brief Helper function to create a Ceres autodiff local parameterization
 * wrapper.
 * @see CeresLocalParameterizationFunctor
 */
template <typename _LieGroup>
std::shared_ptr<ceres::AutoDiffLocalParameterization<
        CeresLocalParameterizationFunctor<_LieGroup>,
        _LieGroup::RepSize,
        _LieGroup::DoF>>
make_local_parameterization_autodiff() {
    return std::make_shared<ceres::AutoDiffLocalParameterization<
            CeresLocalParameterizationFunctor<_LieGroup>, _LieGroup::RepSize,
            _LieGroup::DoF>>();
}

/**
 * @brief Helper function to create a Ceres autodiff objective wrapper.
 * @see CeresObjectiveFunctor
 */
template <typename _LieGroup, typename... Args>
std::shared_ptr<ceres::AutoDiffCostFunction<CeresObjectiveFunctor<_LieGroup>,
                                            1,
                                            _LieGroup::RepSize>>
make_objective_autodiff(Args&&... args) {
    return std::make_shared<ceres::AutoDiffCostFunction<
            CeresObjectiveFunctor<_LieGroup>, 1, _LieGroup::RepSize>>(
            new CeresObjectiveFunctor<_LieGroup>(std::forward<Args>(args)...));
}

/**
 * @brief Helper function to create a Ceres autodiff constraint wrapper.
 * @see CeresConstraintFunctor
 */
template <typename _LieGroup, typename... Args>
std::shared_ptr<ceres::AutoDiffCostFunction<CeresConstraintFunctor<_LieGroup>,
                                            _LieGroup::DoF,
                                            _LieGroup::RepSize,
                                            _LieGroup::RepSize>>
make_constraint_autodiff(Args&&... args) {
    return std::make_shared<ceres::AutoDiffCostFunction<
            CeresConstraintFunctor<_LieGroup>, _LieGroup::DoF,
            _LieGroup::RepSize, _LieGroup::RepSize>>(
            new CeresConstraintFunctor<_LieGroup>(std::forward<Args>(args)...));
}

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_UTILS_H_ */
