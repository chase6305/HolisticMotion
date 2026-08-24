#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3TANGENT_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3TANGENT_BASE_H_

#include "holistic_motion/manif/impl/so3/SO3_properties.h"
#include "holistic_motion/manif/impl/tangent_base.h"

namespace holistic_motion {
namespace robotics {
//
// Tangent
//

/**
 * @brief The base class of the SO3 tangent.
 * @note See Appendix B.
 */
template <typename _Derived>
struct SO3TangentBase : TangentBase<_Derived> {
private:
    using Base = TangentBase<_Derived>;
    using Type = SO3TangentBase<_Derived>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    using AngBlock =
            typename DataType::template FixedSegmentReturnType<3>::Type;
    using ConstAngBlock =
            typename DataType::template ConstFixedSegmentReturnType<3>::Type;

    // Tangent common API

    using Base::Coeffs;

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SO3TangentBase)

public:
    HOLISTIC_MOTION_TANGENT_ML_ASSIGN_OP(SO3TangentBase)

    /**
     * @brief Hat operator of SO3.
     * @return An element of the Lie algebra so3 (skew-symmetric matrix).
     * @note See example 3 of the paper.
     */
    LieAlg Hat() const;

    /**
     * @brief Get the SO3 element.
     * @param[out] -optional- J_m_t Jacobian of the SO3 element wrt this.
     * @return The SO3 element.
     * @note This is the Exp() map with the argument in vector form.
     * @note See Eq. (132) and Eq. (143).
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
     * Get the right Jacobian of SO3.
     * @note See Eq. (143).
     */
    Jacobian Rjac() const;

    /**
     * Get the left Jacobian of SO3.
     * @note See Eq. (145).
     */
    Jacobian Ljac() const;

    /**
     * Get the inverse of the right Jacobian of SO3.
     * @note See Eq. (144).
     * @see rjac.
     */
    Jacobian Rjacinv() const;

    /**
     * Get the inverse of the left Jacobian of SO3.
     * @note See Eq. (146).
     * @see ljac.
     */
    Jacobian Ljacinv() const;

    /**
     * @brief
     * @return
     */
    Jacobian SmallAdj() const;

    // SO3Tangent specific API

    //! @brief
    Scalar x() const;
    //! @brief
    Scalar y() const;
    //! @brief
    Scalar z() const;

    //! @brief Get the angular part.
    AngBlock ang();
    const ConstAngBlock ang() const;
};

template <typename _Derived>
typename SO3TangentBase<_Derived>::LieGroup SO3TangentBase<_Derived>::Exp(
        OptJacobianRef J_m_t) const {
    using std::cos;
    using std::sin;
    using std::sqrt;

    const DataType& theta_vec = Coeffs();
    const Scalar theta_sq = theta_vec.squaredNorm();

    if (theta_sq > Constants<Scalar>::eps) {
        const Scalar theta = sqrt(theta_sq);
        if (J_m_t) {
            const LieAlg W = Hat();

            J_m_t->setIdentity();
            J_m_t->noalias() -= (Scalar(1.0) - cos(theta)) / theta_sq * W;
            J_m_t->noalias() +=
                    (theta - sin(theta)) / (theta_sq * theta) * W * W;
        }

        return LieGroup(
                Eigen::AngleAxis<Scalar>(theta, theta_vec.normalized()));
    } else {
        if (J_m_t) {
            J_m_t->setIdentity();
            J_m_t->noalias() -= Scalar(0.5) * Hat();
        }

        return LieGroup(x() / Scalar(2), y() / Scalar(2), z() / Scalar(2),
                        Scalar(1));
    }
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::LieGroup SO3TangentBase<_Derived>::Retract(
        OptJacobianRef J_m_t) const {
    return Exp(J_m_t);
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::Jacobian SO3TangentBase<_Derived>::Rjac()
        const {
    return Ljac().transpose();
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::Jacobian SO3TangentBase<_Derived>::Ljac()
        const {
    using std::cos;
    using std::sin;
    using std::sqrt;

    const Scalar theta_sq = Coeffs().squaredNorm();

    const LieAlg W = Hat();

    // Small angle approximation
    if (theta_sq <= Constants<Scalar>::eps)
        return Jacobian::Identity() + Scalar(0.5) * W;

    const Scalar theta = sqrt(theta_sq);  // rotation angle

    return Jacobian::Identity() + (Scalar(1) - cos(theta)) / theta_sq * W +
           (theta - sin(theta)) / (theta_sq * theta) * W * W;
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::Jacobian SO3TangentBase<_Derived>::Rjacinv()
        const {
    return Ljacinv().transpose();
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::Jacobian SO3TangentBase<_Derived>::Ljacinv()
        const {
    using std::cos;
    using std::sin;
    using std::sqrt;

    const Scalar theta_sq = Coeffs().squaredNorm();

    const LieAlg W = Hat();

    if (theta_sq <= Constants<Scalar>::eps)
        return Jacobian::Identity() - Scalar(0.5) * W;

    const Scalar theta = sqrt(theta_sq);  // rotation angle

    return Jacobian::Identity() - Scalar(0.5) * W +
           (Scalar(1) / theta_sq -
            (Scalar(1) + cos(theta)) / (Scalar(2) * theta * sin(theta))) *
                   W * W;
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::Jacobian SO3TangentBase<_Derived>::SmallAdj()
        const {
    return Hat();
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::LieAlg SO3TangentBase<_Derived>::Hat()
        const {
    return skew(Coeffs());
}

// SO3Tangent specifics

template <typename _Derived>
typename SO3TangentBase<_Derived>::Scalar SO3TangentBase<_Derived>::x() const {
    return Coeffs()(0);
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::Scalar SO3TangentBase<_Derived>::y() const {
    return Coeffs()(1);
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::Scalar SO3TangentBase<_Derived>::z() const {
    return Coeffs()(2);
}

template <typename _Derived>
typename SO3TangentBase<_Derived>::AngBlock SO3TangentBase<_Derived>::ang() {
    return Coeffs().template tail<3>();
}

template <typename _Derived>
const typename SO3TangentBase<_Derived>::ConstAngBlock
SO3TangentBase<_Derived>::ang() const {
    return Coeffs().template tail<3>();
}

namespace internal {

//! @brief Generator specialization for SO3TangentBase objects.
template <typename Derived>
struct GeneratorEvaluator<SO3TangentBase<Derived>> {
    static typename SO3TangentBase<Derived>::LieAlg run(const unsigned int i) {
        using LieAlg = typename SO3TangentBase<Derived>::LieAlg;
        using Scalar = typename SO3TangentBase<Derived>::Scalar;

        switch (i) {
            case 0: {
                static const LieAlg E0(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(-1), Scalar(0), Scalar(1), Scalar(0))
                                .finished());
                return E0;
            }
            case 1: {
                static const LieAlg E1(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(1), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(-1), Scalar(0), Scalar(0))
                                .finished());
                return E1;
            }
            case 2: {
                static const LieAlg E2((LieAlg() << Scalar(0), Scalar(-1),
                                        Scalar(0), Scalar(1), Scalar(0),
                                        Scalar(0), Scalar(0), Scalar(0),
                                        Scalar(0))
                                               .finished());
                return E2;
            }
            default:
                HOLISTIC_MOTION_THROW("Index i must be in [0,2]!", invalid_argument);
                break;
        }

        return LieAlg{};
    }
};

//! @brief Random specialization for SO3TangentBase objects.
template <typename Derived>
struct RandomEvaluatorImpl<SO3TangentBase<Derived>> {
    static void run(SO3TangentBase<Derived>& m) {
        // In ball of radius PI
        m.Coeffs() = randPointInBall(HOLISTIC_MOTION_PI)
                             .template cast<typename Derived::Scalar>();
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SO3TANGENT_BASE_H_ */
