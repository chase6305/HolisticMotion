#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_OBJECTIVE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_OBJECTIVE_H_

#include <Eigen/Core>

namespace holistic_motion {
namespace robotics {
template <typename _LieGroup>
class CeresObjectiveFunctor {
    using LieGroup = _LieGroup;
    using Tangent = typename _LieGroup::Tangent;

    template <typename _Scalar>
    using LieGroupTemplate =
            typename LieGroup::template LieGroupTemplate<_Scalar>;

public:
    HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND_TYPE(LieGroup)

    template <typename... Args>
    CeresObjectiveFunctor(Args&&... args)
        : target_state_(std::forward<Args>(args)...) {
        //
    }

    CeresObjectiveFunctor(const LieGroup& target_state, const double weight = 1)
        : weight_(weight), target_state_(target_state) {
        //
    }

    virtual ~CeresObjectiveFunctor() = default;

    template <typename T>
    bool operator()(const T* const state_raw, T* residuals_raw) const {
        const Eigen::Map<const LieGroupTemplate<T>> state(state_raw);

        residuals_raw[0] =
                (target_state_.template cast<T>() - state).Coeffs().norm() *
                T(weight_);

        /// @todo
        ///
        /// Jacobian G = q.rjac().transpose() * q.rjac();
        /// residual = (q.Coeffs().transpose() * G * q.Coeffs());
        ///
        /// or
        ///
        /// residual = (q.Coeffs().transpose() * W * q.Coeffs());

        //    std::cout << "State:"
        //              << state_raw[0] << "," << state_raw[1]
        //              << "\n";

        //    std::cout << "Target:"
        //              << target_state_.Coeffs().transpose()
        //              << "\n";

        //    std::cout << "residual: " << residuals_raw[0] << "\n";

        return true;
    }

    LieGroup GetTargetState() const;
    void SetTargetState(const LieGroup& target_state) const;

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
void CeresObjectiveFunctor<_LieGroup>::SetTargetState(
        const LieGroup& target_state) const {
    target_state_ = target_state;
}

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_CERES_OBJECTIVE_H_ */
