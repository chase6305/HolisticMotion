#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2TANGENT_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2TANGENT_BASE_H_

#include "holistic_motion/manif/impl/so2/SO2_properties.h"
#include "holistic_motion/manif/impl/tangent_base.h"

namespace holistic_motion {
namespace robotics {
//
// Tangent
//

/**
 * @brief The base class of the SO2 tangent.
 * @note See Appendix A.
 */
template <typename _Derived>
struct SO2TangentBase : TangentBase<_Derived> {
private:
    using Base = TangentBase<_Derived>;
    using Type = SO2TangentBase<_Derived>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    using Base::Coeffs;

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SO2TangentBase)

public:
    HOLISTIC_MOTION_TANGENT_ML_ASSIGN_OP(SO2TangentBase)

    // Tangent common API

    /**
     * @brief Hat operator of SO2.
     * @return An element of the Lie algebra so2 (skew-symmetric matrix).
     * @note See Eqs. (112, 113).
     */
    LieAlg Hat() const;

    /**
     * @brief Get the SO2 element.
     * @param[out] -optional- J_m_t Jacobian of the SO2 element wrt this.
     * @return The SO2 element.
     * @note This is the Exp() map with the argument in vector form.
     * @note See Eqs. (114, 116) and Eq. (126).
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
     * @brief Get the right Jacobian of SO2.
     * @note See Eq. (126).
     */
    Jacobian Rjac() const;

    /**
     * @brief Get the left Jacobian of SO2.
     * @note See Eq. (126).
     */
    Jacobian Ljac() const;

    /**
     * @brief Get the inverse of the right Jacobian of SO2.
     * @note See Eq. (126).
     * @see rjac.
     */
    Jacobian Rjacinv() const;

    /**
     * @brief Get the inverse of the right Jacobian of SO2.
     * @note See Eq. (126).
     * @see ljac.
     */
    Jacobian Ljacinv() const;

    /**
     * @brief
     * @return
     */
    Jacobian SmallAdj() const;

    // SO2Tangent specific API

    // const Scalar& angle() const;

    //! @brief Get the angle (rad.).
    Scalar Angle() const;
};

template <typename _Derived>
typename SO2TangentBase<_Derived>::LieGroup SO2TangentBase<_Derived>::Exp(
        OptJacobianRef J_m_t) const {
    using std::cos;
    using std::sin;

    if (J_m_t) {
        (*J_m_t) = Rjac();
    }

    return LieGroup(cos(Angle()), sin(Angle()));
}

template <typename _Derived>
typename SO2TangentBase<_Derived>::LieGroup SO2TangentBase<_Derived>::Retract(
        OptJacobianRef J_m_t) const {
    return Exp(J_m_t);
}

template <typename _Derived>
typename SO2TangentBase<_Derived>::LieAlg SO2TangentBase<_Derived>::Hat()
        const {
    return (LieAlg() << Scalar(0), Scalar(-Angle()), Scalar(Angle()), Scalar(0))
            .finished();
}

template <typename _Derived>
typename SO2TangentBase<_Derived>::Jacobian SO2TangentBase<_Derived>::Rjac()
        const {
    static const Jacobian Jr = Jacobian::Constant(Scalar(1));
    return Jr;
}

template <typename _Derived>
typename SO2TangentBase<_Derived>::Jacobian SO2TangentBase<_Derived>::Ljac()
        const {
    static const Jacobian Jl = Jacobian::Constant(Scalar(1));
    return Jl;
}

template <typename _Derived>
typename SO2TangentBase<_Derived>::Jacobian SO2TangentBase<_Derived>::Rjacinv()
        const {
    return Rjac();
}

template <typename _Derived>
typename SO2TangentBase<_Derived>::Jacobian SO2TangentBase<_Derived>::Ljacinv()
        const {
    return Ljac();
}

template <typename _Derived>
typename SO2TangentBase<_Derived>::Jacobian SO2TangentBase<_Derived>::SmallAdj()
        const {
    static const Jacobian smallAdj = Jacobian::Zero();
    return smallAdj;
}

// SO2Tangent specific API

// template <typename _Derived>
// const typename SO2TangentBase<_Derived>::Scalar&
// SO2TangentBase<_Derived>::angle() const
//{
//  return Coeffs().x();
//}

template <typename _Derived>
typename SO2TangentBase<_Derived>::Scalar SO2TangentBase<_Derived>::Angle()
        const {
    return Coeffs()(0);
}

namespace internal {

/**
 * @brief Generator specialization for SO2TangentBase objects.
 * E = | 0 -1 |
 *     | 1  0 |
 */
template <typename Derived>
struct GeneratorEvaluator<SO2TangentBase<Derived>> {
    static typename SO2TangentBase<Derived>::LieAlg run(const unsigned int i) {
        HOLISTIC_MOTION_CHECK(i == 0, "Index i must be 0!", invalid_argument);

        const static typename SO2TangentBase<Derived>::LieAlg E0 =
                skew(typename SO2TangentBase<Derived>::Scalar(1));

        return E0;
    }
};

//! @brief Random specialization for SO2TangentBase objects.
template <typename Derived>
struct RandomEvaluatorImpl<SO2TangentBase<Derived>> {
    static void run(SO2TangentBase<Derived>& m) {
        // in [-1,1]  /  in [-PI,PI]
        m.Coeffs().setRandom() *= HOLISTIC_MOTION_PI;
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO2_BASE_H_ */
