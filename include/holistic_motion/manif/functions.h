#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_FUNCTIONS_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_FUNCTIONS_H_

#include "holistic_motion/manif/impl/lie_group_base.h"

namespace holistic_motion {
namespace robotics {
template <typename _Derived>
const typename _Derived::DataType& Coeffs(
        const LieGroupBase<_Derived>& lie_group) {
    return lie_group.Coeffs();
}

template <typename _Derived>
const typename _Derived::DataType& Coeffs(
        const TangentBase<_Derived>& tangent) {
    return tangent.Coeffs();
}

template <typename _Derived>
const typename _Derived::Scalar* Data(const LieGroupBase<_Derived>& lie_group) {
    return lie_group.Data();
}

template <typename _Derived>
typename _Derived::Scalar* Data(LieGroupBase<_Derived>& lie_group) {
    return lie_group.Data();
}

template <typename _Derived>
const typename _Derived::Scalar* Data(const TangentBase<_Derived>& tangent) {
    return tangent.Data();
}

template <typename _Derived>
typename _Derived::Scalar* Data(TangentBase<_Derived>& tangent) {
    return tangent.Data();
}

template <typename _Derived>
void Identity(LieGroupBase<_Derived>& lie_group) {
    lie_group.identity();
}

template <typename _LieGroup>
_LieGroup Identity() {
    return _LieGroup::Identity();
}

template <typename _Derived>
void Zero(TangentBase<_Derived>& tangent) {
    tangent.zero();
}

template <typename _Tangent>
_Tangent Zero() {
    return _Tangent::ZeroHelper();
}

template <typename _Derived>
void Random(LieGroupBase<_Derived>& lie_group) {
    lie_group.random();
}

template <typename _Type>
_Type Random() {
    return _Type::Random();
}

template <typename _Derived>
void Random(TangentBase<_Derived>& tangent) {
    tangent.random();
}

template <typename _Derived>
typename _Derived::LieGroup Inverse(
        const LieGroupBase<_Derived>& lie_group,
        typename _Derived::OpJacobianRef J_minv_m = {}) {
    return lie_group.Inverse(J_minv_m);
}

template <typename _DerivedMan, typename _DerivedTan>
typename _DerivedMan::LieGroup Rplus(
        const LieGroupBase<_DerivedMan>& lie_group,
        const TangentBase<_DerivedTan>& tangent,
        typename _DerivedMan::OpJacobianRef J_mout_m = {},
        typename _DerivedMan::OpJacobianRef J_mout_t = {}) {
    return lie_group.Rplus(tangent, J_mout_m, J_mout_t);
}

template <typename _DerivedMan, typename _DerivedTan>
typename _DerivedMan::LieGroup Lplus(
        const LieGroupBase<_DerivedMan>& lie_group,
        const TangentBase<_DerivedTan>& tangent,
        typename _DerivedMan::OpJacobianRef J_mout_m = {},
        typename _DerivedMan::OpJacobianRef J_mout_t = {}) {
    return lie_group.Lplus(tangent, J_mout_m, J_mout_t);
}

template <typename _DerivedMan, typename _DerivedTan>
typename _DerivedMan::LieGroup Plus(
        const LieGroupBase<_DerivedMan>& lie_group,
        const TangentBase<_DerivedTan>& tangent,
        typename _DerivedMan::OpJacobianRef J_mout_m = {},
        typename _DerivedMan::OpJacobianRef J_mout_t = {}) {
    return lie_group.Plus(tangent, J_mout_m, J_mout_t);
}

template <typename _Derived0, typename _Derived1>
typename _Derived0::Tangent Rminus(
        const LieGroupBase<_Derived0>& lie_group_lhs,
        const LieGroupBase<_Derived1>& lie_group_rhs,
        typename _Derived0::OptJacobianRef J_t_ma = {},
        typename _Derived0::OptJacobianRef J_t_mb = {}) {
    return lie_group_lhs.Rminus(lie_group_rhs, J_t_ma, J_t_mb);
}

template <typename _Derived0, typename _Derived1>
typename _Derived0::Tangent Lminus(
        const LieGroupBase<_Derived0>& lie_group_lhs,
        const LieGroupBase<_Derived1>& lie_group_rhs,
        typename _Derived0::OptJacobianRef J_t_ma = {},
        typename _Derived0::OptJacobianRef J_t_mb = {}) {
    return lie_group_lhs.Lminus(lie_group_rhs, J_t_ma, J_t_mb);
}

template <typename _Derived0, typename _Derived1>
typename _Derived0::Tangent Minus(
        const LieGroupBase<_Derived0>& lie_group_lhs,
        const LieGroupBase<_Derived1>& lie_group_rhs,
        typename _Derived0::OptJacobianRef J_t_ma = {},
        typename _Derived0::OptJacobianRef J_t_mb = {}) {
    return lie_group_lhs.Minus(lie_group_rhs, J_t_ma, J_t_mb);
}

template <typename _Derived>
HOLISTIC_MOTION_DEPRECATED typename _Derived::Tangent Lift(
        const LieGroupBase<_Derived>& lie_group,
        typename _Derived::OptJacobianRef J_l_m = {}) {
    return lie_group.Log(J_l_m);
}

template <typename _Derived>
typename _Derived::Tangent Log(const LieGroupBase<_Derived>& lie_group,
                               typename _Derived::OptJacobianRef J_l_m = {}) {
    return lie_group.Log(J_l_m);
}

template <typename _Derived>
HOLISTIC_MOTION_DEPRECATED typename _Derived::LieGroup Retract(
        const TangentBase<_Derived>& tangent,
        typename _Derived::OptJacobianRef J_r_t = {}) {
    return tangent.Exp(J_r_t);
}

template <typename _Derived>
typename _Derived::LieGroup Exp(const TangentBase<_Derived>& tangent,
                                typename _Derived::OptJacobianRef J_e_t = {}) {
    return tangent.Exp(J_e_t);
}

template <typename _Derived0, typename _Derived1>
typename _Derived0::LieGroup Compose(
        const LieGroupBase<_Derived0>& lie_group_lhs,
        const LieGroupBase<_Derived1>& lie_group_rhs,
        typename _Derived0::OptJacobianRef J_mc_ma = {},
        typename _Derived0::OptJacobianRef J_mc_mb = {}) {
    return lie_group_lhs.Compose(lie_group_rhs, J_mc_ma, J_mc_mb);
}

template <typename _Derived0, typename _Derived1>
typename _Derived0::LieGroup Between(
        const LieGroupBase<_Derived0>& lie_group_lhs,
        const LieGroupBase<_Derived1>& lie_group_rhs,
        typename _Derived0::OptJacobianRef J_mc_ma = {},
        typename _Derived0::OptJacobianRef J_mc_mb = {}) {
    return lie_group_lhs.Between(lie_group_rhs, J_mc_ma, J_mc_mb);
}

template <typename _Derived>
typename _Derived::Vector Act(const LieGroupBase<_Derived>& lie_group,
                              typename _Derived::Vector v,
                              typename _Derived::OptJacobianRef J_vout_m = {},
                              typename _Derived::OptJacobianRef J_vout_v = {}) {
    return lie_group.Act(v, J_vout_m, J_vout_v);
}

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_FUNCTIONS_H_ */
