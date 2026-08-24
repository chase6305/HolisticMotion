#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE2TANGENT_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE2TANGENT_BASE_H_

#include "holistic_motion/manif/impl/se2/SE2_properties.h"
#include "holistic_motion/manif/impl/tangent_base.h"

namespace holistic_motion {
namespace robotics {
//
// Tangent
//

/**
 * @brief The base class of the SE2 tangent.
 * @note See Appendix C.
 */
template <typename _Derived>
struct SE2TangentBase : TangentBase<_Derived> {
private:
    using Base = TangentBase<_Derived>;
    using Type = SE2TangentBase<_Derived>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    using Base::Coeffs;
    using Base::Data;

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SE2TangentBase)

public:
    HOLISTIC_MOTION_TANGENT_ML_ASSIGN_OP(SE2TangentBase)

    // Tangent common API

    /**
     * @brief Hat operator of SE2.
     * @return An element of the Lie algebra se2 (skew-symmetric matrix).
     * @note See Eq. (153).
     */
    LieAlg Hat() const;

    /**
     * @brief Get the SE2 element.
     * @param[out] -optional- J_m_t Jacobian of the SE2 element wrt this.
     * @return The SE2 element.
     * @note This is the Exp() map with the argument in vector form.
     * @note See Eqs. (156,158) & Eq. (163).
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
     * @brief Get the right Jacobian of SE2.
     * @note See Eq. (163).
     */
    Jacobian Rjac() const;

    /**
     * @brief Get the inverse right Jacobian of SE2.
     */
    Jacobian Rjacinv() const;

    /**
     * @brief Get the left Jacobian of SE2.
     * @note See Eq. (164).
     */
    Jacobian Ljac() const;

    /**
     * @brief Get the inverse left Jacobian of SE2.
     */
    Jacobian Ljacinv() const;

    /**
     * @brief
     * @return
     */
    Jacobian SmallAdj() const;

    // SE2Tangent specific API

    //! @brief Get the x component of the translational part.
    Scalar x() const;
    //! @brief Get the y component of the translational part.
    Scalar y() const;
    //! @brief Get the rotational part.
    Scalar Angle() const;
};

template <typename _Derived>
typename SE2TangentBase<_Derived>::LieAlg SE2TangentBase<_Derived>::Hat()
        const {
    return (LieAlg() << Scalar(0), -Angle(), x(), Angle(), Scalar(0), y(),
            Scalar(0), Scalar(0), Scalar(0))
            .finished();
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::LieGroup SE2TangentBase<_Derived>::Exp(
        OptJacobianRef J_m_t) const {
    using std::abs;
    using std::cos;
    using std::sin;

    const Scalar theta = Angle();
    const Scalar cos_theta = cos(theta);
    const Scalar sin_theta = sin(theta);
    const Scalar theta_sq = theta * theta;

    Scalar A,   // sin_theta_by_theta
            B;  // one_minus_cos_theta_by_theta

    if (theta_sq < Constants<Scalar>::eps) {
        // Taylor approximation
        A = Scalar(1) - Scalar(1. / 6.) * theta_sq;
        B = Scalar(.5) * theta - Scalar(1. / 24.) * theta * theta_sq;
    } else {
        // Euler
        A = sin_theta / theta;
        B = (Scalar(1) - cos_theta) / theta;
    }

    if (J_m_t) {
        // Jr
        J_m_t->setIdentity();
        (*J_m_t)(0, 0) = A;
        (*J_m_t)(0, 1) = B;
        (*J_m_t)(1, 0) = -B;
        (*J_m_t)(1, 1) = A;

        if (theta_sq < Constants<Scalar>::eps) {
            (*J_m_t)(0, 2) = -y() / Scalar(2) + theta * x() / Scalar(6);
            (*J_m_t)(1, 2) = x() / Scalar(2) + theta * y() / Scalar(6);
        } else {
            (*J_m_t)(0, 2) =
                    (-y() + theta * x() + y() * cos_theta - x() * sin_theta) /
                    theta_sq;
            (*J_m_t)(1, 2) =
                    (x() + theta * y() - x() * cos_theta - y() * sin_theta) /
                    theta_sq;
        }
    }

    return LieGroup(A * x() - B * y(), B * x() + A * y(), cos_theta, sin_theta);
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::LieGroup SE2TangentBase<_Derived>::Retract(
        OptJacobianRef J_m_t) const {
    return Exp(J_m_t);
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::Jacobian SE2TangentBase<_Derived>::Rjac()
        const {
    //  const Scalar theta = Angle();
    //  const Scalar cos_theta = cos(theta);
    //  const Scalar sin_theta = sin(theta);
    //  const Scalar theta_sq = theta * theta;

    //  Scalar A,  // sin_theta_by_theta
    //         B;  // one_minus_cos_theta_by_theta

    //  if (abs(theta) < Constants<Scalar>::eps)
    //  {
    //    // Taylor approximation
    //    A = Scalar(1) - Scalar(1. / 6.) * theta_sq;
    //    B = Scalar(.5) * theta - Scalar(1. / 24.) * theta * theta_sq;
    //  }
    //  else
    //  {
    //    // Euler
    //    A = sin_theta / theta;
    //    B = (Scalar(1) - cos_theta) / theta;
    //  }

    //  Jacobian Jr = Jacobian::Identity();
    //  Jr(0,0) =  A;
    //  Jr(0,1) =  B;
    //  Jr(1,0) = -B;
    //  Jr(1,1) =  A;

    //  Jr(0,2) = (-y() + theta*x() + y()*cos_theta - x()*sin_theta)/theta_sq;
    //  Jr(1,2) = ( x() + theta*y() - x()*cos_theta - y()*sin_theta)/theta_sq;

    //  return Jr;

    Jacobian Jr;
    Exp(Jr);

    return Jr;
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::Jacobian SE2TangentBase<_Derived>::Rjacinv()
        const {
    using std::abs;
    using std::cos;
    using std::sin;

    const Scalar theta = Angle();
    const Scalar cos_theta = cos(theta);
    const Scalar sin_theta = sin(theta);
    const Scalar theta_sq = theta * theta;

    Scalar A,   // theta_sin_theta
            B;  // theta_cos_theta

    A = theta * sin_theta;
    B = theta * cos_theta;

    Jacobian Jrinv;

    Jrinv(0, 1) = -theta * Scalar(0.5);
    Jrinv(1, 0) = -Jrinv(0, 1);

    if (theta_sq > Constants<Scalar>::eps) {
        Jrinv(0, 0) = -A / (Scalar(2) * cos_theta - Scalar(2));
        Jrinv(1, 1) = Jrinv(0, 0);

        Scalar den = Scalar(2) * theta * (cos_theta - Scalar(1));
        Jrinv(0, 2) = (A * x() + B * y() - theta * y() +
                       Scalar(2) * x() * cos_theta - Scalar(2) * x()) /
                      den;
        Jrinv(1, 2) = (-B * x() + A * y() + theta * x() +
                       Scalar(2) * y() * cos_theta - Scalar(2) * y()) /
                      den;
    } else {
        Jrinv(0, 0) = Scalar(1) - theta_sq / Scalar(12);
        Jrinv(1, 1) = Jrinv(0, 0);

        Jrinv(0, 2) = y() / Scalar(2) + theta * x() / Scalar(12);
        Jrinv(1, 2) = -x() / Scalar(2) + theta * y() / Scalar(12);
    }

    Jrinv(2, 0) = Scalar(0);
    Jrinv(2, 1) = Scalar(0);
    Jrinv(2, 2) = Scalar(1);

    return Jrinv;
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::Jacobian SE2TangentBase<_Derived>::Ljac()
        const {
    using std::cos;
    using std::sin;

    const Scalar theta = Angle();
    const Scalar cos_theta = cos(theta);
    const Scalar sin_theta = sin(theta);
    const Scalar theta_sq = theta * theta;

    Scalar A,   // sin_theta_by_theta
            B;  // one_minus_cos_theta_by_theta

    if (theta_sq < Constants<Scalar>::eps) {
        // Taylor approximation
        A = Scalar(1) - Scalar(1. / 6.) * theta_sq;
        B = Scalar(.5) * theta - Scalar(1. / 24.) * theta * theta_sq;
    } else {
        // Euler
        A = sin_theta / theta;
        B = (Scalar(1) - cos_theta) / theta;
    }

    Jacobian Jl = Jacobian::Identity();
    Jl(0, 0) = A;
    Jl(0, 1) = -B;
    Jl(1, 0) = B;
    Jl(1, 1) = A;

    if (theta_sq < Constants<Scalar>::eps) {
        Jl(0, 2) = y() / Scalar(2) + theta * x() / Scalar(6);
        Jl(1, 2) = -x() / Scalar(2) + theta * y() / Scalar(6);
    } else {
        Jl(0, 2) = (y() + theta * x() - y() * cos_theta - x() * sin_theta) /
                   theta_sq;
        Jl(1, 2) = (-x() + theta * y() + x() * cos_theta - y() * sin_theta) /
                   theta_sq;
    }

    return Jl;
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::Jacobian SE2TangentBase<_Derived>::Ljacinv()
        const {
    using std::abs;
    using std::cos;
    using std::sin;

    const Scalar theta = Angle();
    const Scalar cos_theta = cos(theta);
    const Scalar sin_theta = sin(theta);
    const Scalar theta_sq = theta * theta;

    Scalar A,   // theta_sin_theta
            B;  // theta_cos_theta

    A = theta * sin_theta;
    B = theta * cos_theta;

    Jacobian Jlinv;

    Jlinv(0, 1) = theta * Scalar(0.5);
    Jlinv(1, 0) = -Jlinv(0, 1);

    if (theta_sq > Constants<Scalar>::eps) {
        Jlinv(0, 0) = -A / (Scalar(2) * cos_theta - Scalar(2));
        Jlinv(1, 1) = Jlinv(0, 0);

        Scalar den = Scalar(2) * theta * (cos_theta - Scalar(1));
        Jlinv(0, 2) = (A * x() - B * y() + theta * y() +
                       Scalar(2) * x() * cos_theta - Scalar(2) * x()) /
                      den;
        Jlinv(1, 2) = (B * x() + A * y() - theta * x() +
                       Scalar(2) * y() * cos_theta - Scalar(2) * y()) /
                      den;
    } else {
        Jlinv(0, 0) = Scalar(1) - theta_sq / Scalar(12);
        Jlinv(1, 1) = Jlinv(0, 0);

        Jlinv(0, 2) = -y() / Scalar(2) + theta * x() / Scalar(12);
        Jlinv(1, 2) = x() / Scalar(2) + theta * y() / Scalar(12);
    }

    Jlinv(2, 0) = Scalar(0);
    Jlinv(2, 1) = Scalar(0);
    Jlinv(2, 2) = Scalar(1);

    return Jlinv;
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::Jacobian SE2TangentBase<_Derived>::SmallAdj()
        const {
    Jacobian smallAdj = Jacobian::Zero();

    smallAdj(0, 1) = -Angle();
    smallAdj(1, 0) = Angle();
    smallAdj(0, 2) = y();
    smallAdj(1, 2) = -x();

    return smallAdj;
}

// SE2Tangent specific API

template <typename _Derived>
typename SE2TangentBase<_Derived>::Scalar SE2TangentBase<_Derived>::x() const {
    return Coeffs().x();
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::Scalar SE2TangentBase<_Derived>::y() const {
    return Coeffs().y();
}

template <typename _Derived>
typename SE2TangentBase<_Derived>::Scalar SE2TangentBase<_Derived>::Angle()
        const {
    return Coeffs().z();
}

namespace internal {

//! @brief Generator specialization for SE2TangentBase objects.
template <typename Derived>
struct GeneratorEvaluator<SE2TangentBase<Derived>> {
    static typename SE2TangentBase<Derived>::LieAlg run(const unsigned int i) {
        using LieAlg = typename SE2TangentBase<Derived>::LieAlg;
        using Scalar = typename SE2TangentBase<Derived>::Scalar;

        switch (i) {
            case 0: {
                const static LieAlg E0(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(1), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0))
                                .finished());
                return E0;
            }
            case 1: {
                const static LieAlg E1(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(0))
                                .finished());
                return E1;
            }
            case 2: {
                const static LieAlg E2((LieAlg() << Scalar(0), Scalar(-1),
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

//! @brief Inner weight matrix specialization for SE2TangentBase objects.
template <typename Derived>
struct InnerWeightsEvaluator<SE2TangentBase<Derived>> {
    static typename Derived::InnerWeightsMatrix run() {
        using InnerWeightsMatrix =
                typename SE2TangentBase<Derived>::InnerWeightsMatrix;
        using Scalar = typename SE2TangentBase<Derived>::Scalar;

        const static InnerWeightsMatrix W((InnerWeightsMatrix() << Scalar(1),
                                           Scalar(0), Scalar(0), Scalar(0),
                                           Scalar(1), Scalar(0), Scalar(0),
                                           Scalar(0), Scalar(2))
                                                  .finished());

        return W;
    }
};

//! @brief Random specialization for SE2TangentBase objects.
template <typename Derived>
struct RandomEvaluatorImpl<SE2TangentBase<Derived>> {
    static void run(SE2TangentBase<Derived>& m) {
        m.Coeffs().setRandom();             // in [-1,1]
        m.Coeffs().coeffRef(2) *= HOLISTIC_MOTION_PI;  // in [-PI,PI]
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE2_BASE_H_ */
