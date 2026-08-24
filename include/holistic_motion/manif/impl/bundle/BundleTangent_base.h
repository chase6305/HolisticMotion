#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLETANGENT_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLETANGENT_BASE_H_

#include "holistic_motion/manif/impl/tangent_base.h"
#include "holistic_motion/manif/impl/traits.h"

namespace holistic_motion {
namespace robotics {
/**
 * @brief The base class of the Bundle tangent.
 */
template <typename _Derived>
struct BundleTangentBase : TangentBase<_Derived> {
private:
    using Base = TangentBase<_Derived>;
    using Type = BundleTangentBase<_Derived>;

public:
    /**
     * @brief Number of elements in the BundleTangent
     */
    static constexpr std::size_t BundleSize =
            internal::traits<_Derived>::BundleSize;

    using Elements = typename internal::traits<_Derived>::Elements;

    template <int Idx>
    using Element = typename internal::traits<_Derived>::template Element<Idx>;

    template <int Idx>
    using MapElement =
            typename internal::traits<_Derived>::template MapElement<Idx>;

    template <int Idx>
    using MapConstElement =
            typename internal::traits<_Derived>::template MapConstElement<Idx>;

    HOLISTIC_MOTION_TANGENT_TYPEDEF
    HOLISTIC_MOTION_INHERIT_TANGENT_API
    HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR

    using Base::Coeffs;
    using Base::Data;

protected:
    using Base::Derived;

    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(BundleTangentBase)

public:
    HOLISTIC_MOTION_TANGENT_ML_ASSIGN_OP(BundleTangentBase)

    // Tangent common API

    /**
     * @brief Hat operator.
     * @return An element of the Lie algebra.
     */
    LieAlg Hat() const;

    /**
     * @brief Exponential operator.
     * @return An element of the Lie Group.
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
     * @brief Get the right Jacobian.
     */
    Jacobian Rjac() const;

    /**
     * @brief Get the left Jacobian.
     */
    Jacobian Ljac() const;

    /**
     * @brief Get the inverse of the right Jacobian.
     */
    Jacobian Rjacinv() const;

    /**
     * @brief Get the inverse of the left Jacobian.
     */
    Jacobian Ljacinv() const;

    /**
     * @brief
     */
    Jacobian SmallAdj() const;

    // BundleTangent specific API

    /**
     * @brief Access BundleTangent element as Map
     * @tparam _Idx element index
     */
    template <int _Idx>
    MapElement<_Idx> element();

    /**
     * @brief Access BundleTangent element as Map to const
     * @tparam _Idx element index
     */
    template <int _Idx>
    MapConstElement<_Idx> element() const;

protected:
    template <int... _Idx>
    LieAlg hat_impl(internal::intseq<_Idx...>) const;

    template <int... _Idx>
    LieGroup exp_impl(OptJacobianRef J_m_t, internal::intseq<_Idx...>) const;

    template <int... _Idx>
    Jacobian rjac_impl(internal::intseq<_Idx...>) const;

    template <int... _Idx>
    Jacobian ljac_impl(internal::intseq<_Idx...>) const;

    template <int... _Idx>
    Jacobian rjacinv_impl(internal::intseq<_Idx...>) const;

    template <int... _Idx>
    Jacobian ljacinv_impl(internal::intseq<_Idx...>) const;

    template <int... _Idx>
    Jacobian smallAdj_impl(internal::intseq<_Idx...>) const;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::LieAlg BundleTangentBase<_Derived>::Hat()
        const {
    return hat_impl(internal::make_intseq_t<BundleSize>{});
}

template <typename _Derived>
template <int... _Idx>
typename BundleTangentBase<_Derived>::LieAlg
BundleTangentBase<_Derived>::hat_impl(internal::intseq<_Idx...>) const {
    LieAlg ret = LieAlg::Zero();
    // c++11 "fold expression"
    auto l = {((ret.template block<Element<_Idx>::LieAlg::RowsAtCompileTime,
                                   Element<_Idx>::LieAlg::RowsAtCompileTime>(
                        std::get<_Idx>(internal::traits<_Derived>::AlgIdx),
                        std::get<_Idx>(internal::traits<_Derived>::AlgIdx)) =
                        element<_Idx>().Hat()),
               0)...};
    static_cast<void>(l);  // compiler warning
    return ret;
}

template <typename _Derived>
typename BundleTangentBase<_Derived>::LieGroup BundleTangentBase<_Derived>::Exp(
        OptJacobianRef J_m_t) const {
    if (J_m_t) {
        J_m_t->setZero();
    }
    return exp_impl(J_m_t, internal::make_intseq_t<BundleSize>{});
}

template <typename _Derived>
template <int... _Idx>
typename BundleTangentBase<_Derived>::LieGroup
BundleTangentBase<_Derived>::exp_impl(OptJacobianRef J_m_t,
                                      internal::intseq<_Idx...>) const {
    if (J_m_t) {
        return LieGroup(element<_Idx>().exp(
                J_m_t->template block<Element<_Idx>::DoF, Element<_Idx>::DoF>(
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx),
                        std::get<_Idx>(
                                internal::traits<_Derived>::DoFIdx)))...);
    }
    return LieGroup(element<_Idx>().Exp()...);
}

template <typename _Derived>
typename BundleTangentBase<_Derived>::LieGroup
BundleTangentBase<_Derived>::Retract(OptJacobianRef J_m_t) const {
    return Exp(J_m_t);
}

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::Rjac() const {
    return rjac_impl(internal::make_intseq_t<BundleSize>{});
}

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::Ljac() const {
    return ljac_impl(internal::make_intseq_t<BundleSize>{});
}

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::Rjacinv() const {
    return rjacinv_impl(internal::make_intseq_t<BundleSize>{});
}

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::Ljacinv() const {
    return ljacinv_impl(internal::make_intseq_t<BundleSize>{});
}

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::SmallAdj() const {
    return smallAdj_impl(internal::make_intseq_t<BundleSize>{});
}

template <typename _Derived>
template <int... _Idx>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::rjac_impl(internal::intseq<_Idx...>) const {
    Jacobian Jr = Jacobian::Zero();
    // c++11 "fold expression"
    auto l = {((Jr.template block<Element<_Idx>::DoF, Element<_Idx>::DoF>(
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx),
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx)) =
                        element<_Idx>().Rjac()),
               0)...};
    static_cast<void>(l);  // compiler warning
    return Jr;
}

template <typename _Derived>
template <int... _Idx>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::ljac_impl(internal::intseq<_Idx...>) const {
    Jacobian Jr = Jacobian::Zero();
    // c++11 "fold expression"
    auto l = {((Jr.template block<Element<_Idx>::DoF, Element<_Idx>::DoF>(
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx),
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx)) =
                        element<_Idx>().Ljac()),
               0)...};
    static_cast<void>(l);  // compiler warning
    return Jr;
}

template <typename _Derived>
template <int... _Idx>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::rjacinv_impl(internal::intseq<_Idx...>) const {
    Jacobian Jr = Jacobian::Zero();
    // c++11 "fold expression"
    auto l = {((Jr.template block<Element<_Idx>::DoF, Element<_Idx>::DoF>(
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx),
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx)) =
                        element<_Idx>().rjacinv()),
               0)...};
    static_cast<void>(l);  // compiler warning
    return Jr;
}

template <typename _Derived>
template <int... _Idx>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::ljacinv_impl(internal::intseq<_Idx...>) const {
    Jacobian Jr = Jacobian::Zero();
    // c++11 "fold expression"
    auto l = {((Jr.template block<Element<_Idx>::DoF, Element<_Idx>::DoF>(
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx),
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx)) =
                        element<_Idx>().ljacinv()),
               0)...};
    static_cast<void>(l);  // compiler warning
    return Jr;
}

template <typename _Derived>
template <int... _Idx>
typename BundleTangentBase<_Derived>::Jacobian
BundleTangentBase<_Derived>::smallAdj_impl(internal::intseq<_Idx...>) const {
    Jacobian Jr = Jacobian::Zero();
    // c++11 "fold expression"
    auto l = {((Jr.template block<Element<_Idx>::DoF, Element<_Idx>::DoF>(
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx),
                        std::get<_Idx>(internal::traits<_Derived>::DoFIdx)) =
                        element<_Idx>().smallAdj()),
               0)...};
    static_cast<void>(l);  // compiler warning
    return Jr;
}

template <typename _Derived>
template <int _Idx>
auto BundleTangentBase<_Derived>::element() -> MapElement<_Idx> {
    return MapElement<_Idx>(
            static_cast<_Derived &>(*this).Coeffs().data() +
            std::get<_Idx>(internal::traits<_Derived>::RepSizeIdx));
}

template <typename _Derived>
template <int _Idx>
auto BundleTangentBase<_Derived>::element() const -> MapConstElement<_Idx> {
    return MapConstElement<_Idx>(
            static_cast<const _Derived &>(*this).Coeffs().data() +
            std::get<_Idx>(internal::traits<_Derived>::RepSizeIdx));
}

namespace internal {

/**
 * @brief Generator specialization for BundleTangentBase objects.
 */
template <typename Derived>
struct GeneratorEvaluator<BundleTangentBase<Derived>> {
    static typename BundleTangentBase<Derived>::LieAlg run(
            const unsigned int i) {
        HOLISTIC_MOTION_CHECK(i < BundleTangentBase<Derived>::DoF,
                   "Index i must less than DoF!", invalid_argument);

        return run(i, make_intseq_t<Derived::BundleSize>{});
    }

    template <int... _Idx>
    static typename BundleTangentBase<Derived>::LieAlg run(const unsigned int i,
                                                           intseq<_Idx...>) {
        using LieAlg = typename BundleTangentBase<Derived>::LieAlg;
        LieAlg Ei = LieAlg::Zero();
        // c++11 "fold expression"
        auto l = {
                ((Ei.template block<Derived::template Element<
                                            _Idx>::LieAlg::RowsAtCompileTime,
                                    Derived::template Element<
                                            _Idx>::LieAlg::RowsAtCompileTime>(
                          std::get<_Idx>(internal::traits<Derived>::AlgIdx),
                          std::get<_Idx>(internal::traits<Derived>::AlgIdx)) =
                          (static_cast<int>(i) >=
                                   std::get<_Idx>(
                                           internal::traits<Derived>::DoFIdx) &&
                           static_cast<int>(i) <
                                   std::get<_Idx>(
                                           internal::traits<Derived>::DoFIdx) +
                                           Derived::template Element<_Idx>::DoF)
                                  ? Derived::template Element<_Idx>::Generator(
                                            static_cast<int>(i) -
                                            std::get<_Idx>(internal::traits<
                                                           Derived>::DoFIdx))
                                  : Derived::template Element<
                                            _Idx>::LieAlg::Zero()),
                 0)...};
        static_cast<void>(l);  // compiler warning
        return Ei;
    }
};

/**
 * @brief Random specialization for BundleTangent objects.
 */
template <typename Derived>
struct RandomEvaluatorImpl<BundleTangentBase<Derived>> {
    static void run(BundleTangentBase<Derived> &m) {
        run(m, make_intseq_t<Derived::BundleSize>{});
    }

    template <int... _Idx>
    static void run(BundleTangentBase<Derived> &m, intseq<_Idx...>) {
        m = typename BundleTangentBase<Derived>::Tangent(
                Derived::template Element<_Idx>::Random()...);
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_BUNDLETANGENT_BASE_H_
