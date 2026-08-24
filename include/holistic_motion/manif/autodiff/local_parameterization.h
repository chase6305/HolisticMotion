#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_AUTODIFF_LOCAL_PARAMETRIZATION_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_AUTODIFF_LOCAL_PARAMETRIZATION_H_

namespace holistic_motion {
namespace robotics {

template <typename Ad, typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RepSize, Derived::DoF>
autodiffLocalParameterizationJacobian(
        const holistic_motion::robotics::LieGroupBase<Derived>& _state) {
    using Scalar = typename Derived::Scalar;
    using LieGroup = typename Derived::template LieGroupTemplate<Ad>;
    using Tangent = typename Derived::Tangent::template TangentTemplate<Ad>;

    using Jac = Eigen::Matrix<Scalar, Derived::RepSize, Derived::DoF>;

    LieGroup state = _state.template cast<Ad>();
    Tangent delta = Tangent::Zero();

    LieGroup state_plus_delta;

    auto f = [](const auto& s, const auto& t) { return s + t; };

    Jac J_so_t =
            autodiff::jacobian(f, autodiff::wrt(delta),
                               autodiff::at(state, delta), state_plus_delta);

    HOLISTIC_MOTION_ASSERT(state.IsApprox(state_plus_delta));
    HOLISTIC_MOTION_ASSERT(Derived::RepSize == J_so_t.rows());
    HOLISTIC_MOTION_ASSERT(Derived::DoF == J_so_t.cols());

    return J_so_t;
};

}  // namespace robotics
}  // namespace holistic_motion
#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_AUTODIFF_LOCAL_PARAMETRIZATION_H_
