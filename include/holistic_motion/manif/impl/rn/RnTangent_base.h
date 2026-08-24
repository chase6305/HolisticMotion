#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_BASE_H_

#include "holistic_motion/manif/impl/rn/Rn_properties.h"
#include "holistic_motion/manif/impl/tangent_base.h"

namespace holistic_motion {
namespace robotics {
//
// Tangent
//

/**
 * @brief The base class of the R^n tangent.
 * @note See Appendix E.
 */
template <typename _Derived>
struct RnTangentBase : TangentBase<_Derived> {
private:
    using Base = TangentBase<_Derived>;
    using Type = RnTangentBase<_Derived>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR
    using Base::Coeffs;

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(RnTangentBase)

public:
    HOLISTIC_MOTION_TANGENT_ML_ASSIGN_OP(RnTangentBase)

    // Tangent common API

    /**
     * @brief Hat operator of Rn.
     * @return An element of the Lie algebra rn.
     * @note See Appendix E.
     */
    LieAlg Hat() const;

    /**
     * @brief Get the Rn element.
     * @param[out] -optional- J_m_t Jacobian of the Rn element wrt this.
     * @return The Rn element.
     * @note This is the Exp() map with the argument in vector form.
     * @note See Eqs. (184) and Eq. (191).
     */
    LieGroup Exp(OptJacobianRef J_m_t = {}) const;

    /**
     * @brief This function is deprecated.
     * Please considere using
     * @ref exp instead.
     */
    HOLISTIC_MOTION_DEPRECATED
    LieGroup Retract(OptJacobianRef J_m_t = {}) const;

    /**
     * @brief Get the right Jacobian of Rn.
     * @note See Eq. (191).
     */
    Jacobian Rjac() const;

    /**
     * @brief Get the left Jacobian of Rn.
     * @note See Eq. (191).
     */
    Jacobian Ljac() const;

    /**
     * @brief Get the inverse of the right Jacobian of Rn.
     * @note See Eq. (191).
     * @see rjac.
     */
    Jacobian Rjacinv() const;

    /**
     * @brief Get the inverse of the left Jacobian of Rn.
     * @note See Eq. (191).
     * @see ljac.
     */
    Jacobian Ljacinv() const;

    /**
     * @brief
     * @return
     */
    Jacobian SmallAdj() const;

    // RnTangent specific API
};

template <typename _Derived>
typename RnTangentBase<_Derived>::LieGroup RnTangentBase<_Derived>::Exp(
        OptJacobianRef J_m_t) const {
    if (J_m_t) {
        J_m_t->setIdentity();
    }

    return LieGroup(Coeffs());
}

template <typename _Derived>
typename RnTangentBase<_Derived>::LieGroup RnTangentBase<_Derived>::Retract(
        OptJacobianRef J_m_t) const {
    return Exp(J_m_t);
}

template <typename _Derived>
typename RnTangentBase<_Derived>::LieAlg RnTangentBase<_Derived>::Hat() const {
    LieAlg t_hat = LieAlg::Zero();
    t_hat.template topRightCorner<Dim, 1>() = Coeffs();
    return t_hat;
}

template <typename _Derived>
typename RnTangentBase<_Derived>::Jacobian RnTangentBase<_Derived>::Rjac()
        const {
    static const Jacobian Jr = Jacobian::Identity();
    return Jr;
}

template <typename _Derived>
typename RnTangentBase<_Derived>::Jacobian RnTangentBase<_Derived>::Ljac()
        const {
    static const Jacobian Jl = Jacobian::Identity();
    return Jl;
}

template <typename _Derived>
typename RnTangentBase<_Derived>::Jacobian RnTangentBase<_Derived>::Rjacinv()
        const {
    return Rjac();
}

template <typename _Derived>
typename RnTangentBase<_Derived>::Jacobian RnTangentBase<_Derived>::Ljacinv()
        const {
    return Ljac();
}

template <typename _Derived>
typename RnTangentBase<_Derived>::Jacobian RnTangentBase<_Derived>::SmallAdj()
        const {
    static const Jacobian smallAdj = Jacobian::Zero();
    return smallAdj;
}

// RnTangent specific API

namespace internal {

/**
 * @brief Generator specialization for RnTangentBase objects.
 */
template <typename Derived>
struct GeneratorEvaluator<RnTangentBase<Derived>> {
    static typename RnTangentBase<Derived>::LieAlg run(const unsigned int i) {
        HOLISTIC_MOTION_CHECK(i < RnTangentBase<Derived>::DoF,
                   "Index i must less than DoF!", invalid_argument);

        using LieAlg = typename RnTangentBase<Derived>::LieAlg;

        LieAlg Ei = LieAlg::Zero();

        Ei(i, RnTangentBase<Derived>::DoF) = 1;

        return Ei;
    }
};

//! @brief Random specialization for RnTangentBase objects.
template <typename Derived>
struct RandomEvaluatorImpl<RnTangentBase<Derived>> {
    static void run(RnTangentBase<Derived>& m) { m.Coeffs().setRandom(); }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_RNTANGENT_BASE_H_
