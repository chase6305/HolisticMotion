#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3TANGENT_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3TANGENT_BASE_H_

#include "holistic_motion/manif/impl/se3/SE3_properties.h"
#include "holistic_motion/manif/impl/so3/SO3Tangent_map.h"
#include "holistic_motion/manif/impl/tangent_base.h"

namespace holistic_motion {
namespace robotics {
//
// Tangent
//

/**
 * @brief The base class of the SE3 tangent.
 * @note See Appendix D of the paper.
 */
template <typename _Derived>
struct SE3TangentBase : TangentBase<_Derived> {
private:
    using Base = TangentBase<_Derived>;
    using Type = SE3TangentBase<_Derived>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    using LinBlock =
            typename DataType::template FixedSegmentReturnType<3>::Type;
    using AngBlock =
            typename DataType::template FixedSegmentReturnType<3>::Type;
    using ConstLinBlock =
            typename DataType::template ConstFixedSegmentReturnType<3>::Type;
    using ConstAngBlock =
            typename DataType::template ConstFixedSegmentReturnType<3>::Type;

    using Base::Coeffs;
    using Base::Data;

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SE3TangentBase)

public:
    HOLISTIC_MOTION_TANGENT_ML_ASSIGN_OP(SE3TangentBase)

    // Tangent common API

    /**
     * @brief Hat operator of SE3.
     * @return An element of the Lie algebra se3.
     * @note See Eq. (169).
     */
    LieAlg Hat() const;

    /**
     * @brief Get the SE3 element.
     * @param[out] -optional- J_m_t Jacobian of the SE3 element wrt this.
     * @return The SE3 element.
     * @note This is the Exp() map with the argument in vector form.
     * @note See Eq. (172) & Eqs. (179,180).
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
     * @brief Get the right Jacobian of SE3.
     * @note See note after Eqs. (179,180).
     */
    Jacobian Rjac() const;

    /**
     * @brief Get the left Jacobian of SE3.
     * @note See Eqs. (179,180).
     */
    Jacobian Ljac() const;

    /**
     * @brief Get the inverse right Jacobian of SE3.
     * @note See note after Eqs. (179,180).
     */
    Jacobian Rjacinv() const;

    /**
     * @brief Get the inverse left Jacobian of SE3.
     * @note See Eqs. (179,180).
     */
    Jacobian Ljacinv() const;

    /**
     * @brief
     * @return
     */
    Jacobian SmallAdj() const;

    // SE3Tangent specific API

    //! @brief Get the linear part.
    LinBlock lin();
    const ConstLinBlock lin() const;

    //! @brief Get the angular part.
    AngBlock ang();
    const ConstAngBlock ang() const;

    //  Scalar x() const;
    //  Scalar y() const;
    //  Scalar z() const;

    // Scalar roll() const;
    // Scalar pitch() const;
    // Scalar yaw() const;

public:  /// @todo make protected
    const Eigen::Map<const SO3Tangent<Scalar>> AsSO3() const {
        return Eigen::Map<const SO3Tangent<Scalar>>(Coeffs().data() + 3);
    }

    Eigen::Map<SO3Tangent<Scalar>> AsSO3() {
        return Eigen::Map<SO3Tangent<Scalar>>(Coeffs().data() + 3);
    }

    // private:

    template <typename _EigenDerived>
    static void FillQ(Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> Q,
                      const Eigen::MatrixBase<_EigenDerived>& c);
};

template <typename _Derived>
typename SE3TangentBase<_Derived>::LieGroup SE3TangentBase<_Derived>::Exp(
        OptJacobianRef J_m_t) const {
    using std::cos;
    using std::sin;
    using std::sqrt;

    if (J_m_t) {
        *J_m_t = Rjac();
    }

    /// @note Eq. 10.93
    return LieGroup(AsSO3().Ljac() * lin(), AsSO3().Exp().GetQuat());
}

template <typename _Derived>
typename SE3TangentBase<_Derived>::LieGroup SE3TangentBase<_Derived>::Retract(
        OptJacobianRef J_m_t) const {
    return Exp(J_m_t);
}

template <typename _Derived>
typename SE3TangentBase<_Derived>::LieAlg SE3TangentBase<_Derived>::Hat()
        const {
    return (LieAlg() << Scalar(0), Scalar(-Coeffs()(5)), Scalar(Coeffs()(4)),
            Scalar(Coeffs()(0)), Scalar(Coeffs()(5)), Scalar(0),
            Scalar(-Coeffs()(3)), Scalar(Coeffs()(1)), Scalar(-Coeffs()(4)),
            Scalar(Coeffs()(3)), Scalar(0), Scalar(Coeffs()(2)), Scalar(0),
            Scalar(0), Scalar(0), Scalar(0))
            .finished();
}

/// @note Eq. 10.95
/// @note barfoot14tro Eq. 102
template <typename _Derived>
typename SE3TangentBase<_Derived>::Jacobian SE3TangentBase<_Derived>::Rjac()
        const {
    /// @note Eq. 10.95
    Jacobian Jr;
    Jr.template bottomLeftCorner<3, 3>().setZero();
    Jr.template topLeftCorner<3, 3>() = AsSO3().Rjac();
    Jr.template bottomRightCorner<3, 3>() = Jr.template topLeftCorner<3, 3>();
    FillQ(Jr.template topRightCorner<3, 3>(), -Coeffs());

    return Jr;
}

template <typename _Derived>
typename SE3TangentBase<_Derived>::Jacobian SE3TangentBase<_Derived>::Ljac()
        const {
    /// @note Eq. 10.95
    Jacobian Jl;
    Jl.template bottomLeftCorner<3, 3>().setZero();
    Jl.template topLeftCorner<3, 3>() = AsSO3().Ljac();
    Jl.template bottomRightCorner<3, 3>() = Jl.template topLeftCorner<3, 3>();
    FillQ(Jl.template topRightCorner<3, 3>(), Coeffs());

    return Jl;
}

/// @note barfoot14tro Eq. 102
template <typename _Derived>
typename SE3TangentBase<_Derived>::Jacobian SE3TangentBase<_Derived>::Rjacinv()
        const {
    /// @note Eq. 10.95
    Jacobian Jr_inv;
    FillQ(Jr_inv.template bottomLeftCorner<3, 3>(),
          -Coeffs());  // serves as temporary Q
    Jr_inv.template topLeftCorner<3, 3>() = AsSO3().Rjacinv();
    Jr_inv.template bottomRightCorner<3, 3>() =
            Jr_inv.template topLeftCorner<3, 3>();
    Jr_inv.template topRightCorner<3, 3>().noalias() =
            -Jr_inv.template topLeftCorner<3, 3>() *
            Jr_inv.template bottomLeftCorner<3, 3>() *
            Jr_inv.template topLeftCorner<3, 3>();
    Jr_inv.template bottomLeftCorner<3, 3>().setZero();

    return Jr_inv;
}

template <typename _Derived>
typename SE3TangentBase<_Derived>::Jacobian SE3TangentBase<_Derived>::Ljacinv()
        const {
    Jacobian Jl_inv;
    FillQ(Jl_inv.template bottomLeftCorner<3, 3>(),
          Coeffs());  // serves as temporary Q
    Jl_inv.template topLeftCorner<3, 3>() = AsSO3().Ljacinv();
    Jl_inv.template bottomRightCorner<3, 3>() =
            Jl_inv.template topLeftCorner<3, 3>();
    Jl_inv.template topRightCorner<3, 3>().noalias() =
            -Jl_inv.template topLeftCorner<3, 3>() *
            Jl_inv.template bottomLeftCorner<3, 3>() *
            Jl_inv.template topLeftCorner<3, 3>();
    Jl_inv.template bottomLeftCorner<3, 3>().setZero();

    return Jl_inv;
}

template <typename _Derived>
template <typename _EigenDerived>
void SE3TangentBase<_Derived>::FillQ(
        Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> Q,
        const Eigen::MatrixBase<_EigenDerived>& c) {
    using std::cos;
    using std::sin;
    using std::sqrt;

    const Scalar theta_sq = c.template tail<3>().squaredNorm();

    Scalar A(0.5), B, C, D;

    // Small angle approximation
    if (theta_sq <= Constants<Scalar>::eps) {
        B = Scalar(1. / 6.) + Scalar(1. / 120.) * theta_sq;
        C = -Scalar(1. / 24.) + Scalar(1. / 720.) * theta_sq;
        D = -Scalar(1. / 60.);
    } else {
        const Scalar theta = sqrt(theta_sq);
        const Scalar sin_theta = sin(theta);
        const Scalar cos_theta = cos(theta);

        B = (theta - sin_theta) / (theta_sq * theta);
        C = (Scalar(1) - theta_sq / Scalar(2) - cos_theta) /
            (theta_sq * theta_sq);
        D = (C - Scalar(3) *
                         (theta - sin_theta - theta_sq * theta / Scalar(6)) /
                         (theta_sq * theta_sq * theta));

        // http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17_identities.pdf
        //    C = (theta_sq+Scalar(2)*cos_theta-Scalar(2)) /
        //    (Scalar(2)*theta_sq*theta_sq); D = (Scalar(2)*theta -
        //    Scalar(3)*sin_theta + theta*cos_theta) /
        //    (Scalar(2)*theta_sq*theta_sq*theta);
    }

    /// @note Barfoot14tro Eq. 102
    const Eigen::Matrix<Scalar, 3, 3> V = skew(c.template head<3>());
    const Eigen::Matrix<Scalar, 3, 3> W = skew(c.template tail<3>());
    const Eigen::Matrix<Scalar, 3, 3> VW = V * W;
    const Eigen::Matrix<Scalar, 3, 3> WV =
            VW.transpose();  // Note on this change wrt. Barfoot: it happens
                             // that V*W = (W*V).transpose() !!!
    const Eigen::Matrix<Scalar, 3, 3> WVW = WV * W;
    const Eigen::Matrix<Scalar, 3, 3> VWW = VW * W;
    Q.noalias() =
            +A * V + B * (WV + VW + WVW) -
            C * (VWW - VWW.transpose() -
                 Scalar(3) *
                         WVW)  // Note on this change wrt. Barfoot: it happens
                               // that V*W*W = -(W*W*V).transpose() !!!
            - D * WVW * W;  // Note on this change wrt. Barfoot: it happens that
                            // W*V*W*W = W*W*V*W !!!
}

template <typename _Derived>
typename SE3TangentBase<_Derived>::Jacobian SE3TangentBase<_Derived>::SmallAdj()
        const {
    /// @note Chirikjian (close to Eq.10.94)
    /// says
    ///       ad(g) = |  Omega  0   |
    ///               |   V   Omega |
    ///
    /// considering vee(log(g)) = (w;v)
    ///
    /// but this is
    ///       ad(g) = |  Omega  V   |
    ///               |   0   Omega |
    ///
    /// considering vee(log(g)) = (v;w)

    Jacobian smallAdj;
    smallAdj.template topRightCorner<3, 3>() = skew(lin());
    smallAdj.template topLeftCorner<3, 3>() = skew(ang());
    smallAdj.template bottomRightCorner<3, 3>() =
            smallAdj.template topLeftCorner<3, 3>();
    smallAdj.template bottomLeftCorner<3, 3>().setZero();

    return smallAdj;
}

// SE3Tangent specific API

template <typename _Derived>
typename SE3TangentBase<_Derived>::LinBlock SE3TangentBase<_Derived>::lin() {
    return Coeffs().template head<3>();
}

template <typename _Derived>
const typename SE3TangentBase<_Derived>::ConstLinBlock
SE3TangentBase<_Derived>::lin() const {
    return Coeffs().template head<3>();
}

template <typename _Derived>
typename SE3TangentBase<_Derived>::AngBlock SE3TangentBase<_Derived>::ang() {
    return Coeffs().template tail<3>();
}

template <typename _Derived>
const typename SE3TangentBase<_Derived>::ConstAngBlock
SE3TangentBase<_Derived>::ang() const {
    return Coeffs().template tail<3>();
}

// template <typename _Derived>
// typename SE3TangentBase<_Derived>::Scalar
// SE3TangentBase<_Derived>::x() const
//{
//  return Data()->x();
//}

// template <typename _Derived>
// typename SE3TangentBase<_Derived>::Scalar
// SE3TangentBase<_Derived>::y() const
//{
//  return Data()->y();
//}

// template <typename _Derived>
// typename SE3TangentBase<_Derived>::Scalar
// SE3TangentBase<_Derived>::z() const
//{
//  return Data()->z();
//}

namespace internal {

//! @brief Generator specialization for SE3TangentBase objects.
template <typename Derived>
struct GeneratorEvaluator<SE3TangentBase<Derived>> {
    static typename SE3TangentBase<Derived>::LieAlg run(const unsigned int i) {
        using LieAlg = typename SE3TangentBase<Derived>::LieAlg;
        using Scalar = typename SE3TangentBase<Derived>::Scalar;

        switch (i) {
            case 0: {
                static const LieAlg E0(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(1),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0))
                                .finished());
                return E0;
            }
            case 1: {
                static const LieAlg E1(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(1), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0))
                                .finished());
                return E1;
            }
            case 2: {
                static const LieAlg E2(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0))
                                .finished());
                return E2;
            }
            case 3: {
                static const LieAlg E3(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(-1), Scalar(0), Scalar(0),
                         Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0))
                                .finished());
                return E3;
            }
            case 4: {
                static const LieAlg E4(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(1), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(-1),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0))
                                .finished());
                return E4;
            }
            case 5: {
                static const LieAlg E5(
                        (LieAlg() << Scalar(0), Scalar(-1), Scalar(0),
                         Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0))
                                .finished());
                return E5;
            }
            default:
                HOLISTIC_MOTION_THROW("Index i must be in [0,5]!", invalid_argument);
                break;
        }

        return LieAlg{};
    }
};

//! @brief Random specialization for SE3TangentBase objects.
template <typename Derived>
struct RandomEvaluatorImpl<SE3TangentBase<Derived>> {
    static void run(SE3TangentBase<Derived>& m) {
        m.Coeffs().template head<3>().setRandom();
        // In ball of radius PI
        m.Coeffs().template tail<3>() =
                randPointInBall(HOLISTIC_MOTION_PI)
                        .template cast<typename Derived::Scalar>();
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE3_BASE_H_ */
