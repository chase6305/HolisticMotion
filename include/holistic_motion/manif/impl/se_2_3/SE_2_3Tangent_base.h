#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3TANGENT_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3TANGENT_BASE_H_

#include "holistic_motion/manif/impl/se3/SE3Tangent_map.h"
#include "holistic_motion/manif/impl/se_2_3/SE_2_3_properties.h"
#include "holistic_motion/manif/impl/so3/SO3Tangent_map.h"
#include "holistic_motion/manif/impl/tangent_base.h"

namespace holistic_motion {
namespace robotics {
//
// Tangent
//

/**
 * @brief The base class of the SE_2_3 tangent.
 */
template <typename _Derived>
struct SE_2_3TangentBase : TangentBase<_Derived> {
private:
    using Base = TangentBase<_Derived>;
    using Type = SE_2_3TangentBase<_Derived>;

public:
    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
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

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(SE_2_3TangentBase)

public:
    HOLISTIC_MOTION_TANGENT_ML_ASSIGN_OP(SE_2_3TangentBase)

    // Tangent common API

    /**
     * @brief Hat operator of SE_2_3.
     * @return An element of the Lie algebra se_2_3.
     * @note See Eq. (169).
     */
    LieAlg Hat() const;

    /**
     * @brief Get the SE_2_3 element.
     * @param[out] -optional- J_m_t Jacobian of the SE_2_3 element wrt this.
     * @return The SE_2_3 element.
     * @note This is the Exp() map with the argument in vector form.
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
     * @brief Get the right Jacobian of SE_2_3.
     */
    Jacobian Rjac() const;

    /**
     * @brief Get the inverse right Jacobian of SE_2_3.
     */
    Jacobian Rjacinv() const;

    /**
     * @brief Get the left Jacobian of SE_2_3.
     */
    Jacobian Ljac() const;

    /**
     * @brief Get the inverse left Jacobian of SE_2_3.
     */
    Jacobian Ljacinv() const;

    /**
     * @brief Get the small adjoint matrix ad() of SE_2_3
     * that maps isomorphic tangent vectors of SE_2_3
     * @return
     */
    Jacobian SmallAdj() const;

    // SE_2_3Tangent specific API

    //! @brief Get the linear velocity part.
    LinBlock lin();
    const ConstLinBlock lin() const;

    //! @brief Get the angular part.
    AngBlock ang();
    const ConstAngBlock ang() const;

    //! @brief Get the linear acceleration part
    LinBlock lin2();
    const ConstLinBlock lin2() const;

public:  /// @todo make protected
    const Eigen::Map<const SO3Tangent<Scalar>> AsSO3() const {
        return Eigen::Map<const SO3Tangent<Scalar>>(Coeffs().data() + 3);
    }

    Eigen::Map<SO3Tangent<Scalar>> AsSO3() {
        return Eigen::Map<SO3Tangent<Scalar>>(Coeffs().data() + 3);
    }
};

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::LieGroup SE_2_3TangentBase<_Derived>::Exp(
        OptJacobianRef J_m_t) const {
    if (J_m_t) {
        *J_m_t = Rjac();
    }

    const Eigen::Map<const SO3Tangent<Scalar>> so3 = AsSO3();
    const typename SO3<Scalar>::Jacobian so3_ljac = so3.Ljac();

    return LieGroup(so3_ljac * lin(), so3.Exp().GetQuat(), so3_ljac * lin2());
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::LieGroup
SE_2_3TangentBase<_Derived>::Retract(OptJacobianRef J_m_t) const {
    return Exp(J_m_t);
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::LieAlg SE_2_3TangentBase<_Derived>::Hat()
        const {
    return (LieAlg() << Scalar(0), Scalar(-Coeffs()(5)), Scalar(Coeffs()(4)),
            Scalar(Coeffs()(0)), Scalar(Coeffs()(6)), Scalar(Coeffs()(5)),
            Scalar(0), Scalar(-Coeffs()(3)), Scalar(Coeffs()(1)),
            Scalar(Coeffs()(7)), Scalar(-Coeffs()(4)), Scalar(Coeffs()(3)),
            Scalar(0), Scalar(Coeffs()(2)), Scalar(Coeffs()(8)), Scalar(0),
            Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
            Scalar(0), Scalar(0), Scalar(0))
            .finished();
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::Jacobian
SE_2_3TangentBase<_Derived>::Rjac() const {
    Jacobian Jr;
    Jr.template block<6, 3>(3, 0).setZero();
    Jr.template block<6, 3>(0, 6).setZero();
    Jr.template topLeftCorner<3, 3>() = AsSO3().Rjac();
    Jr.template block<3, 3>(3, 3) = Jr.template topLeftCorner<3, 3>();
    Jr.template bottomRightCorner<3, 3>() = Jr.template topLeftCorner<3, 3>();

    // fill Qv
    SE3Tangent<Scalar>::FillQ(Jr.template block<3, 3>(0, 3),
                              -Coeffs().template head<6>());

    // fill Qa
    Eigen::Matrix<Scalar, 6, 1> aw;
    aw << -Coeffs()(6), -Coeffs()(7), -Coeffs()(8), -Coeffs()(3), -Coeffs()(4),
            -Coeffs()(5);
    SE3Tangent<Scalar>::FillQ(Jr.template block<3, 3>(6, 3), aw);

    return Jr;
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::Jacobian
SE_2_3TangentBase<_Derived>::Rjacinv() const {
    Jacobian Jr_inv;
    Jr_inv.template block<3, 3>(3, 0).setZero();
    // Jr_inv.template block<3, 3>(6, 0).setZero(); // Serves as temp Q
    Jr_inv.template block<6, 3>(0, 6).setZero();
    Jr_inv.template topLeftCorner<3, 3>() = AsSO3().Rjacinv();
    Jr_inv.template block<3, 3>(3, 3) = Jr_inv.template topLeftCorner<3, 3>();
    Jr_inv.template bottomRightCorner<3, 3>() =
            Jr_inv.template topLeftCorner<3, 3>();

    // fill Qv
    SE3Tangent<Scalar>::FillQ(Jr_inv.template block<3, 3>(6, 0),
                              -Coeffs().template head<6>());
    Jr_inv.template block<3, 3>(0, 3).noalias() =
            -Jr_inv.template topLeftCorner<3, 3>() *
            Jr_inv.template block<3, 3>(6, 0) *
            Jr_inv.template topLeftCorner<3, 3>();

    // fill Qa
    Eigen::Matrix<Scalar, 6, 1> aw;
    aw << -Coeffs()(6), -Coeffs()(7), -Coeffs()(8), -Coeffs()(3), -Coeffs()(4),
            -Coeffs()(5);
    SE3Tangent<Scalar>::FillQ(Jr_inv.template block<3, 3>(6, 0), aw);
    Jr_inv.template block<3, 3>(6, 3).noalias() =
            -Jr_inv.template topLeftCorner<3, 3>() *
            Jr_inv.template block<3, 3>(6, 0) *
            Jr_inv.template topLeftCorner<3, 3>();

    Jr_inv.template block<3, 3>(6, 0).setZero();

    return Jr_inv;
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::Jacobian
SE_2_3TangentBase<_Derived>::Ljac() const {
    Jacobian Jl;
    Jl.template block<6, 3>(3, 0).setZero();
    Jl.template block<6, 3>(0, 6).setZero();
    Jl.template topLeftCorner<3, 3>() = AsSO3().Ljac();
    Jl.template block<3, 3>(3, 3) = Jl.template topLeftCorner<3, 3>();
    Jl.template bottomRightCorner<3, 3>() = Jl.template topLeftCorner<3, 3>();

    // fill Qv
    SE3Tangent<Scalar>::FillQ(Jl.template block<3, 3>(0, 3),
                              Coeffs().template head<6>());

    // fill Qa
    Eigen::Matrix<Scalar, 6, 1> aw;
    aw << Coeffs()(6), Coeffs()(7), Coeffs()(8), Coeffs()(3), Coeffs()(4),
            Coeffs()(5);
    SE3Tangent<Scalar>::FillQ(Jl.template block<3, 3>(6, 3), aw);

    return Jl;
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::Jacobian
SE_2_3TangentBase<_Derived>::Ljacinv() const {
    Jacobian Jlinv;
    Jlinv.template block<3, 3>(3, 0).setZero();
    // Jlinv.template block<3, 3>(6, 0).setZero(); // Serves as temp Q
    Jlinv.template block<6, 3>(0, 6).setZero();
    Jlinv.template topLeftCorner<3, 3>() = AsSO3().Ljacinv();
    Jlinv.template block<3, 3>(3, 3) = Jlinv.template topLeftCorner<3, 3>();
    Jlinv.template bottomRightCorner<3, 3>() =
            Jlinv.template topLeftCorner<3, 3>();

    // fill Qv
    SE3Tangent<Scalar>::FillQ(Jlinv.template block<3, 3>(6, 0),
                              Coeffs().template head<6>());
    Jlinv.template block<3, 3>(0, 3).noalias() =
            -Jlinv.template topLeftCorner<3, 3>() *
            Jlinv.template block<3, 3>(6, 0) *
            Jlinv.template topLeftCorner<3, 3>();

    // fill Qa
    Eigen::Matrix<Scalar, 6, 1> aw;
    aw << Coeffs()(6), Coeffs()(7), Coeffs()(8), Coeffs()(3), Coeffs()(4),
            Coeffs()(5);
    SE3Tangent<Scalar>::FillQ(Jlinv.template block<3, 3>(6, 0), aw);
    Jlinv.template block<3, 3>(6, 3).noalias() =
            -Jlinv.template topLeftCorner<3, 3>() *
            Jlinv.template block<3, 3>(6, 0) *
            Jlinv.template topLeftCorner<3, 3>();

    Jlinv.template block<3, 3>(6, 0).setZero();

    return Jlinv;
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::Jacobian
SE_2_3TangentBase<_Derived>::SmallAdj() const {
    /// this is
    ///       ad(g) = |  Omega  V      0|
    ///               |   0   Omega    0|
    ///               |   0     A  Omega|
    ///
    /// considering vee(log(g)) = (v;w; a)

    Jacobian smallAdj;
    smallAdj.template block<6, 3>(3, 0).setZero();
    smallAdj.template block<6, 3>(0, 6).setZero();
    smallAdj.template block<3, 3>(0, 3) = skew(lin());
    smallAdj.template topLeftCorner<3, 3>() = skew(ang());
    smallAdj.template block<3, 3>(3, 3) =
            smallAdj.template topLeftCorner<3, 3>();
    smallAdj.template bottomRightCorner<3, 3>() =
            smallAdj.template topLeftCorner<3, 3>();
    smallAdj.template block<3, 3>(6, 3) = skew(lin2());
    return smallAdj;
}

// SE_2_3Tangent specific API

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::LinBlock
SE_2_3TangentBase<_Derived>::lin() {
    return Coeffs().template head<3>();
}

template <typename _Derived>
const typename SE_2_3TangentBase<_Derived>::ConstLinBlock
SE_2_3TangentBase<_Derived>::lin() const {
    return Coeffs().template head<3>();
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::AngBlock
SE_2_3TangentBase<_Derived>::ang() {
    return Coeffs().template segment<3>(3);
}

template <typename _Derived>
const typename SE_2_3TangentBase<_Derived>::ConstAngBlock
SE_2_3TangentBase<_Derived>::ang() const {
    return Coeffs().template segment<3>(3);
}

template <typename _Derived>
typename SE_2_3TangentBase<_Derived>::LinBlock
SE_2_3TangentBase<_Derived>::lin2() {
    return Coeffs().template tail<3>();
}

template <typename _Derived>
const typename SE_2_3TangentBase<_Derived>::ConstLinBlock
SE_2_3TangentBase<_Derived>::lin2() const {
    return Coeffs().template tail<3>();
}

namespace internal {

//! @brief Generator specialization for SE_2_3TangentBase objects.
template <typename Derived>
struct GeneratorEvaluator<SE_2_3TangentBase<Derived>> {
    static typename SE_2_3TangentBase<Derived>::LieAlg run(
            const unsigned int i) {
        using LieAlg = typename SE_2_3TangentBase<Derived>::LieAlg;
        using Scalar = typename SE_2_3TangentBase<Derived>::Scalar;

        switch (i) {
            case 0: {
                static const LieAlg E0(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(1),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0))
                                .finished());
                return E0;
            }
            case 1: {
                static const LieAlg E1(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(1),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0))
                                .finished());
                return E1;
            }
            case 2: {
                static const LieAlg E2(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(1),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0))
                                .finished());
                return E2;
            }
            case 3: {
                static const LieAlg E3(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(-1), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0))
                                .finished());
                return E3;
            }
            case 4: {
                static const LieAlg E4(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(1), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(-1), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0))
                                .finished());
                return E4;
            }
            case 5: {
                static const LieAlg E5(
                        (LieAlg() << Scalar(0), Scalar(-1), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0))
                                .finished());
                return E5;
            }
            case 6: {
                static const LieAlg E6(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0))
                                .finished());
                return E6;
            }
            case 7: {
                static const LieAlg E7(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0))
                                .finished());
                return E7;
            }
            case 8: {
                static const LieAlg E8(
                        (LieAlg() << Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                         Scalar(0))
                                .finished());
                return E8;
            }
            default:
                HOLISTIC_MOTION_THROW("Index i must be in [0,8]!", invalid_argument);
                break;
        }

        return LieAlg{};
    }
};

//! @brief Random specialization for SE_2_3TangentBase objects.
template <typename Derived>
struct RandomEvaluatorImpl<SE_2_3TangentBase<Derived>> {
    static void run(SE_2_3TangentBase<Derived>& m) {
        // in [-1,1]
        m.Coeffs().setRandom();
        // In ball of radius PI
        m.Coeffs().template segment<3>(3) =
                randPointInBall(HOLISTIC_MOTION_PI)
                        .template cast<typename Derived::Scalar>();
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_SE_2_3_BASE_H_ */
