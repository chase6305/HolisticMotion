#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_OBJECTIVE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_OBJECTIVE_H_

#include <Eigen/Core>
#include <type_traits>
#include <utility>

#include "holistic_motion/manif/impl/lie_group_base.h"

namespace holistic_motion {
namespace robotics {
template <typename _LieGroup> class CeresObjectiveFunctor {
    using LieGroup = _LieGroup;

    template <typename _Scalar>
    using LieGroupTemplate = typename LieGroup::template LieGroupTemplate<_Scalar>;

  public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND_TYPE(LieGroup)

    template <typename... Args,
              std::enable_if_t<std::is_constructible_v<LieGroup, Args...>, int> = 0>
    CeresObjectiveFunctor(Args &&...args) : target_state_(std::forward<Args>(args)...) {}

    CeresObjectiveFunctor(const LieGroup &target_state, const double weight = 1)
        : weight_(weight), target_state_(target_state) {}

    virtual ~CeresObjectiveFunctor() = default;

    template <typename T> bool operator()(const T *const state_raw, T *residuals_raw) const {
        const Eigen::Map<const LieGroupTemplate<T>> state(state_raw);

        const auto target = target_state_.template cast<T>();
        if ((target.Coeffs().array() == state.Coeffs().array()).all() &&
            target_state_.Coeffs().allFinite()) {
            residuals_raw[0] = T(0) * T(weight_);
            return true;
        }
        const auto error = (target - state).Coeffs().eval();
        // At the target the squared objective has zero derivative. Avoid
        // autodiff's undefined sqrt derivative for an exactly zero norm.
        residuals_raw[0] = ((error.array() == T(0)).all() ? T(0) : error.norm()) * T(weight_);

        return true;
    }

    LieGroup GetTargetState() const;
    void SetTargetState(const LieGroup &target_state);

    inline void Weight(const double weight) { weight_ = weight; }
    inline double Weight() const noexcept { return weight_; }

  protected:
    double weight_ = 1;
    LieGroup target_state_;
};

template <typename _LieGroup>
typename CeresObjectiveFunctor<_LieGroup>::LieGroup
CeresObjectiveFunctor<_LieGroup>::GetTargetState() const {
    return target_state_;
}

template <typename _LieGroup>
void CeresObjectiveFunctor<_LieGroup>::SetTargetState(const LieGroup &target_state) {
    target_state_ = target_state;
}

} // namespace robotics
} // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_OBJECTIVE_H_ */
