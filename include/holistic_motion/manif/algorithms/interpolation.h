#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_INTERPOLATION_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_INTERPOLATION_H_

#include "holistic_motion/manif/SE3.h"

namespace holistic_motion::robotics {

/// Binomial coefficient for n >= k >= 0.
template <typename T> constexpr T BinomialCoefficient(const T n, const T k) {
    return (n >= k) ? (k >= 0) ? (k * 2 > n) ? BinomialCoefficient(n, n - k)
                                 : k         ? BinomialCoefficient(n, k - 1) * (n - k + 1) / k
                                             : 1
                               : (throw std::logic_error("k >= 0 !"))
                    : (throw std::logic_error("n >= k !"));
}

/// Integer power; nonpositive exponents return the supplied carry.
template <typename T> constexpr T Ipow(T base, int exp, T carry = 1) {
    while (exp > 0) {
        if (exp % 2)
            carry *= base;
        exp /= 2;
        if (exp > 0)
            base *= base;
    }
    return carry;
}

/// Bernstein polynomial of order n, basis index i.
template <typename T> constexpr T PolynomialBernstein(const T n, const T i, const T t) {
    return BinomialCoefficient(n, i) * Ipow(T(1) - t, n - i) * Ipow(t, i);
}

/// Polynomial blend with zero derivatives through the requested degree at 0/1.
template <typename T> T SmoothingPhi(const T t, const std::size_t degree) {
    switch (degree) {
    case 1:
        return t * t * (T(3) - T(2) * t);
    case 2:
        return t * t * t * (T(10) + t * (T(-15) + T(6) * t));
    case 3:
        return t * t * t * t * (T(35) + t * (T(-84) + t * (T(70) - T(20) * t)));
    case 4:
        return t * t * t * t * t *
               (T(126) + t * (T(-420) + t * (T(540) + t * (T(-315) + T(70) * t))));
    default:
        throw std::logic_error("Not implemented yet !");
    }
}

/// Left-trivialized interpolation from ma to mb for t in [0, 1].
template <typename _Derived, typename _Scalar>
static typename LieGroupBase<_Derived>::LieGroup InterpolateLinear(const LieGroupBase<_Derived> &ma,
                                                                   const LieGroupBase<_Derived> &mb,
                                                                   const _Scalar t) {
    HOLISTIC_MOTION_CHECK(t >= _Scalar(0) && t <= _Scalar(1), "s must be be in [0, 1].");
    return ma.Lplus(t * mb.Lminus(ma));
}

/// Linear translation and rotation interpolation in SE3, rather than screw motion.
template <typename _Scalar>
SE3<_Scalar> InterpolateLinearInSE3(const SE3<_Scalar> &ma, const SE3<_Scalar> &mb,
                                    const _Scalar t) {
    HOLISTIC_MOTION_CHECK(t >= _Scalar(0) && t <= _Scalar(1), "s must be be in [0, 1].");
    return SE3<_Scalar>(ma.GetTranslation() + t * (mb.GetTranslation() - ma.GetTranslation()),
                        InterpolateLinear(ma.AsSO3(), mb.AsSO3(), t));
}

/// Right-trivialized geodesic interpolation from ma to mb for t in [0, 1].
template <typename _Derived, typename _Scalar>
static typename LieGroupBase<_Derived>::LieGroup InterpolateSlerp(const LieGroupBase<_Derived> &ma,
                                                                  const LieGroupBase<_Derived> &mb,
                                                                  const _Scalar t) {
    HOLISTIC_MOTION_CHECK(t >= _Scalar(0) && t <= _Scalar(1), "s must be be in [0, 1].");
    return ma.Rplus(mb.Rminus(ma) * t);
}

/// Cubic Bezier interpolation with body tangents ta/tb per unit parameter.
/// The Euclidean case is the cubic Hermite interpolant. Rotational control
/// increments must stay on the intended logarithm branch.
template <typename _Derived, typename _Scalar>
static typename LieGroupBase<_Derived>::LieGroup
InterpolateCubic(const LieGroupBase<_Derived> &ma, const LieGroupBase<_Derived> &mb,
                 const _Scalar t,
                 const typename LieGroupBase<_Derived>::Tangent &ta =
                     LieGroupBase<_Derived>::Tangent::ZeroHelper(),
                 const typename LieGroupBase<_Derived>::Tangent &tb =
                     LieGroupBase<_Derived>::Tangent::ZeroHelper()) {
    using Group = typename LieGroupBase<_Derived>::LieGroup;
    using Scalar = typename LieGroupBase<_Derived>::Scalar;
    HOLISTIC_MOTION_CHECK(t >= _Scalar(0) && t <= _Scalar(1), "s must be be in [0, 1].");
    const Group start(ma), end(mb);
    const Group first = start.Rplus(ta * (Scalar(1) / Scalar(3)));
    const Group second = end.Rplus(tb * (-Scalar(1) / Scalar(3)));
    const Group left = InterpolateSlerp(start, first, t);
    const Group middle = InterpolateSlerp(first, second, t);
    const Group right = InterpolateSlerp(second, end, t);
    return InterpolateSlerp(InterpolateSlerp(left, middle, t), InterpolateSlerp(middle, right, t),
                            t);
}

/// Blend endpoint tangent extrapolations with m zero endpoint derivatives.
template <typename _Derived, typename _Scalar>
static typename LieGroupBase<_Derived>::LieGroup
InterpolateSmooth(const LieGroupBase<_Derived> &ma, const LieGroupBase<_Derived> &mb,
                  const _Scalar t, const unsigned int m,
                  const typename LieGroupBase<_Derived>::Tangent &ta =
                      LieGroupBase<_Derived>::Tangent::ZeroHelper(),
                  const typename LieGroupBase<_Derived>::Tangent &tb =
                      LieGroupBase<_Derived>::Tangent::ZeroHelper()) {
    using Scalar = typename LieGroupBase<_Derived>::Scalar;
    HOLISTIC_MOTION_CHECK(t >= _Scalar(0) && t <= _Scalar(1), "s must be be in [0, 1].");
    const auto phi = SmoothingPhi(t, m);
    const auto right = mb.Rplus(tb * (t - Scalar(1)));
    const auto left = ma.Rplus(ta * t);
    return left.Lplus(right.Lminus(left) * phi);
}

enum class INTERP_METHOD { SLERP, CUBIC, CNSMOOTH };

/// Select geodesic, cubic Bezier, or third-order smooth interpolation.
template <typename _Derived, typename _Scalar>
typename LieGroupBase<_Derived>::LieGroup
Interpolate(const LieGroupBase<_Derived> &ma, const LieGroupBase<_Derived> &mb, const _Scalar t,
            const INTERP_METHOD method = INTERP_METHOD::SLERP,
            const typename LieGroupBase<_Derived>::Tangent &ta =
                LieGroupBase<_Derived>::Tangent::ZeroHelper(),
            const typename LieGroupBase<_Derived>::Tangent &tb =
                LieGroupBase<_Derived>::Tangent::ZeroHelper()) {
    switch (method) {
    case INTERP_METHOD::SLERP:
        return InterpolateSlerp(ma, mb, t);
    case INTERP_METHOD::CUBIC:
        return InterpolateCubic(ma, mb, t, ta, tb);
    case INTERP_METHOD::CNSMOOTH:
        return InterpolateSmooth(ma, mb, t, 3, ta, tb);
    default:
        throw runtime_error("Unknown interpolation method!");
    }
}

} // namespace holistic_motion::robotics

#endif
