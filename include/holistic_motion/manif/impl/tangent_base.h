#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_TANGENT_BASE_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_TANGENT_BASE_H_

#include "holistic_motion/manif/external/tl/tl/optional.hpp"
#include "holistic_motion/manif/constants.h"
#include "holistic_motion/manif/impl/eigen.h"
#include "holistic_motion/manif/impl/generator.h"
#include "holistic_motion/manif/impl/macro.h"
#include "holistic_motion/manif/impl/random.h"
#include "holistic_motion/manif/impl/traits.h"

namespace holistic_motion {
namespace robotics {
/**
 * @brief Base class for Lie groups' tangents.
 * Defines the minimum common API.
 * @see LieGroupBase.
 */
template <class _Derived>
struct TangentBase {
    static constexpr int Dim = internal::traits<_Derived>::Dim;
    static constexpr int RepSize = internal::traits<_Derived>::RepSize;
    static constexpr int DoF = internal::traits<_Derived>::DoF;

    using Scalar = typename internal::traits<_Derived>::Scalar;
    using LieGroup = typename internal::traits<_Derived>::LieGroup;
    using Tangent = typename internal::traits<_Derived>::Tangent;
    using DataType = typename internal::traits<_Derived>::DataType;
    using Jacobian = typename internal::traits<_Derived>::Jacobian;
    using LieAlg = typename internal::traits<_Derived>::LieAlg;

    using InnerWeightsMatrix = Jacobian;

    using OptJacobianRef = tl::optional<Eigen::Ref<Jacobian>>;

    template <typename _Scalar>
    using TangentTemplate =
            typename internal::traitscast<Tangent, _Scalar>::cast;

protected:
    HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(TangentBase)

public:
    /**
     * @brief Assignment operator.
     * @param[in] t An element of the same Tangent group.
     * @return A reference to this.
     */
    _Derived& operator=(const TangentBase& t);

    /**
     * @brief Assignment operator.
     * @param[in] t An element of the same Tangent group.
     * @return A reference to this.
     */
    template <typename _DerivedOther>
    _Derived& operator=(const TangentBase<_DerivedOther>& t);

    /**
     * @brief Assignment operator.
     * @param[in] t A DataType object.
     * @return A reference to this.
     * @see DataType.
     */
    template <typename _EigenDerived>
    _Derived& operator=(const Eigen::MatrixBase<_EigenDerived>& v);

    //! @brief Access the underlying data by reference
    DataType& Coeffs();
    //! @brief Access the underlying data by const reference
    const DataType& Coeffs() const;

    //! @brief Access the underlying data by pointer
    Scalar* Data();
    //! @brief Access the underlying data by const pointer
    const Scalar* Data() const;

    //! @brief Cast the Tangent object to a copy
    //! of a different scalar type
    template <class _NewScalar>
    TangentTemplate<_NewScalar> cast() const;

    // Common Tangent API

    /**
     * @brief Set the Tangent object this to Zero.
     * @return A reference to this.
     */
    _Derived& SetZero();

    /**
     * @brief Set the LieGroup object this to a random value.
     * @return A reference to this.
     */
    _Derived& SetRandom();

    // Minimum API
    // Those functions must be implemented in the Derived class !

    /**
     * @brief Get the ith basis element of the Lie Algebra.
     * @return the ith basis element of the Lie Algebra.
     */
    LieAlg Generator(const int i) const;

    /**
     * @brief Get the weight matrix of the Weighted Euclidean inner product,
     * relative to the space basis.
     * @return the weight matrix.
     * @see generator
     */
    InnerWeightsMatrix InnerWeights() const;

    /**
     * @brief Get inner product of this and another Tangent
     * weighted by W.
     * @return The inner product of this and t.
     * @note ip = v0' . W . v1
     * @see innerWeights()
     */
    template <typename _DerivedOther>
    Scalar Inner(const TangentBase<_DerivedOther>& t) const;

    /**
     * @brief Get the Euclidean weighted norm.
     * @return The Euclidean weighted norm.
     * @see innerWeights()
     * @see squaredWeightedNorm()
     */
    Scalar WeightedNorm() const;

    /**
     * @brief Get the squared Euclidean weighted norm.
     * @return The squared Euclidean weighted norm.
     * @see innerWeights()
     * @see WeightedNorm()
     */
    Scalar SquaredWeightedNorm() const;

    /**
     * @brief Hat operator of the Tangent element.
     * @return The isomorphic element in the Lie algebra.
     * @note See Eq. (10).
     */
    LieAlg Hat() const;

    /**
     * @brief Get the Lie group element
     * @param[out] -optional- J_m_t Jacobian of the Lie groupe element wrt this.
     * @return Associated Lie group element.
     * @note This is the Exp() map with the argument in vector form.
     * @note See Eq. (23).
     */
    LieGroup Exp(OptJacobianRef J_m_t = OptJacobianRef{}) const;

    /**
     * @brief This function is deprecated.
     * Please considere using
     * @ref exp instead.
     */
    HOLISTIC_MOTION_DEPRECATED
    LieGroup Retract(OptJacobianRef J_m_t = OptJacobianRef{}) const;

    /**
     * @brief Right oplus operation of the Lie group.
     * @param[in]  t An element of the tangent of the Lie group.
     * @param[out] -optional- J_mout_t Jacobian of the oplus operation wrt this.
     * @param[out] -optional- J_mout_m Jacobian of the oplus operation wrt group
     * element.
     * @return An element of the Lie group.
     * @note See Eq. (25).
     */
    LieGroup Rplus(const LieGroup& m,
                   OptJacobianRef J_mout_t = {},
                   OptJacobianRef J_mout_m = {}) const;

    /**
     * @brief Left oplus operation of the Lie group.
     * @param[in]  t An element of the tangent of the Lie group.
     * @param[out] -optional- J_mout_t Jacobian of the oplus operation wrt this.
     * @param[out] -optional- J_mout_m Jacobian of the oplus operation wrt the
     * group element.
     * @return An element of the Lie group.
     * @note See Eq. (27).
     */
    LieGroup Lplus(const LieGroup& m,
                   OptJacobianRef J_mout_t = {},
                   OptJacobianRef J_mout_m = {}) const;

    /**
     * @brief An alias for the right oplus operation.
     * @see rplus
     */
    LieGroup Plus(const LieGroup& m,
                  OptJacobianRef J_mout_t = {},
                  OptJacobianRef J_mout_m = {}) const;

    template <typename _DerivedOther>
    Tangent Plus(const TangentBase<_DerivedOther>& t,
                 OptJacobianRef J_mout_ta = {},
                 OptJacobianRef J_mout_tb = {}) const;

    template <typename _DerivedOther>
    Tangent Minus(const TangentBase<_DerivedOther>& t,
                  OptJacobianRef J_mout_ta = {},
                  OptJacobianRef J_mout_tb = {}) const;

    /**
     * @brief Get the right Jacobian.
     * @note this is the right Jacobian of @ref exp, what is commonly known as
     * "the right Jacobian".
     * @note See Eq. (41) for the right Jacobian of general functions.
     * @note See Eqs. (126,143,163,179,191) for implementations of the right
     * Jacobian of @ref exp.
     */
    Jacobian Rjac() const;

    /**
     * @brief Get the left Jacobian.
     * @note this is the left Jacobian of @ref exp, what is commonly known as
     * "the left Jacobian".
     * @note See Eq. (44) for the left Jacobian of general functions.
     * @note See Eqs. (126,145,164,179,191) for implementations of the left
     * Jacobian of @ref exp.
     */
    Jacobian Ljac() const;

    /// @note Calls Derived's 'overload'
    template <typename U = _Derived>
    typename std::enable_if<internal::has_rjacinv<U>::value,
                            typename TangentBase<U>::Jacobian>::type
    Rjacinv() const;

    /// @note Calls Base default impl
    template <typename U = _Derived>
    typename std::enable_if<!internal::has_rjacinv<U>::value,
                            typename TangentBase<U>::Jacobian>::type
    Rjacinv() const;

    /// @note Calls Derived's 'overload'
    template <typename U = _Derived>
    typename std::enable_if<internal::has_ljacinv<U>::value,
                            typename TangentBase<U>::Jacobian>::type
    Ljacinv() const;

    /// @note Calls Base default impl
    template <typename U = _Derived>
    typename std::enable_if<!internal::has_ljacinv<U>::value,
                            typename TangentBase<U>::Jacobian>::type
    Ljacinv() const;

    /**
     * @brief
     * @return [description]
     */
    Jacobian SmallAdj() const;

    /**
     * @brief Evaluate whether this and v are 'close'.
     * @details This evaluation is performed element-wise.
     * @param[in] v A vector.
     * @param[in] eps Threshold for equality copmarison.
     * @return true if the Tangent element t is 'close' to this,
     * false otherwise.
     */
    template <typename _EigenDerived>
    bool IsApprox(const Eigen::MatrixBase<_EigenDerived>& v,
                  const Scalar eps = Constants<Scalar>::eps) const;

    /**
     * @brief Evaluate whether this and t are 'close'.
     * @details This evaluation is performed element-wise.
     * @param[in] t An element of the same Tangent group.
     * @param[in] eps Threshold for equality copmarison.
     * @return true if the Tangent element t is 'close' to this,
     * false otherwise.
     */
    template <typename _DerivedOther>
    bool IsApprox(const TangentBase<_DerivedOther>& t,
                  const Scalar eps = Constants<Scalar>::eps) const;

    // Some operators

    // Copy assignment

    template <typename T>
    auto operator<<(T&& v) -> decltype(
            std::declval<DataType>().operator<<(std::forward<T>(v)));

    // Math

    //! @brief Equivalent to v * -1.
    Tangent operator-() const;

    /**
     * @brief Left oplus operator.
     * @see lplus.
     */
    LieGroup operator+(const LieGroup& m) const;

    //! @brief In-place plus operator, simple vector in-place plus operation.
    template <typename _DerivedOther>
    _Derived& operator+=(const TangentBase<_DerivedOther>& t);

    //! @brief In-place minus operator, simple vector in-place minus operation.
    template <typename _DerivedOther>
    _Derived& operator-=(const TangentBase<_DerivedOther>& t);

    //! @brief In-place plus operator, simple vector in-place plus operation.
    template <typename _EigenDerived>
    _Derived& operator+=(const Eigen::MatrixBase<_EigenDerived>& v);

    //! @brief In-place minus operator, simple vector in-place minus operation.
    template <typename _EigenDerived>
    _Derived& operator-=(const Eigen::MatrixBase<_EigenDerived>& v);

    //! @brief Multiply the underlying vector with a scalar.
    Tangent operator*=(const Scalar scalar);

    //! @brief Divide the underlying vector with a scalar.
    Tangent operator/=(const Scalar scalar);

    //! Access the ith coeffs
    auto operator[](const unsigned int i) const -> decltype(Coeffs()[i]) {
        return Coeffs()[i];
    }

    //! Access the ith coeffs
    auto operator[](const unsigned int i) -> decltype(Coeffs()[i]) {
        return Coeffs()[i];
    }

    //! @brief The size of the underlying vector
    constexpr unsigned int size() const { return RepSize; }

    // static helpers

    //! Static helper the create a Tangent object set to Zero.
    static Tangent ZeroHelper();
    //! Static helper the create a random Tangent object.
    static Tangent RandomHelper();
    //! Static helper to get a Basis of the Lie group.
    static LieAlg GeneratorHelper(const int i);
    //! Static helper to get a Basis of the Lie group.
    static InnerWeightsMatrix InnerWeightsHelper();

protected:
    inline _Derived& Derived() & noexcept {
        return *static_cast<_Derived*>(this);
    }
    inline const _Derived& Derived() const& noexcept {
        return *static_cast<const _Derived*>(this);
    }
};

template <typename _Derived>
constexpr int TangentBase<_Derived>::Dim;
template <typename _Derived>
constexpr int TangentBase<_Derived>::DoF;
template <typename _Derived>
constexpr int TangentBase<_Derived>::RepSize;

// Copy

template <typename _Derived>
_Derived& TangentBase<_Derived>::operator=(const TangentBase& t) {
    Coeffs() = t.Coeffs();
    return Derived();
}

template <typename _Derived>
template <typename _DerivedOther>
_Derived& TangentBase<_Derived>::operator=(
        const TangentBase<_DerivedOther>& t) {
    Coeffs() = t.Coeffs();
    return Derived();
}

template <typename _Derived>
template <typename _EigenDerived>
_Derived& TangentBase<_Derived>::operator=(
        const Eigen::MatrixBase<_EigenDerived>& v) {
    Coeffs() = v;
    return Derived();
}

template <typename _Derived>
typename TangentBase<_Derived>::DataType& TangentBase<_Derived>::Coeffs() {
    return Derived().Coeffs();
}

template <typename _Derived>
const typename TangentBase<_Derived>::DataType& TangentBase<_Derived>::Coeffs()
        const {
    return Derived().Coeffs();
}

template <class _Derived>
typename TangentBase<_Derived>::Scalar* TangentBase<_Derived>::Data() {
    return Derived().Coeffs().data();
}

template <class _Derived>
const typename TangentBase<_Derived>::Scalar* TangentBase<_Derived>::Data()
        const {
    return Derived().Coeffs().data();
}

template <typename _Derived>
template <class _NewScalar>
typename TangentBase<_Derived>::template TangentTemplate<_NewScalar>
TangentBase<_Derived>::cast() const {
    return TangentTemplate<_NewScalar>(Coeffs().template cast<_NewScalar>());
}

template <class _Derived>
_Derived& TangentBase<_Derived>::SetZero() {
    Coeffs().setZero();
    return Derived();
}

template <class _Derived>
_Derived& TangentBase<_Derived>::SetRandom() {
    internal::RandomEvaluator<typename internal::traits<_Derived>::Base>(
            Derived())
            .run();

    return Derived();
}

template <class _Derived>
typename TangentBase<_Derived>::LieGroup TangentBase<_Derived>::Exp(
        OptJacobianRef J_m_t) const {
    return Derived().Exp(J_m_t);
}

template <class _Derived>
typename TangentBase<_Derived>::LieGroup TangentBase<_Derived>::Retract(
        OptJacobianRef J_m_t) const {
    return Derived().Exp(J_m_t);
}

template <typename _Derived>
typename TangentBase<_Derived>::LieAlg TangentBase<_Derived>::Generator(
        const int i) const {
    return GeneratorHelper(i);
}

template <typename _Derived>
typename TangentBase<_Derived>::InnerWeightsMatrix
TangentBase<_Derived>::InnerWeights() const {
    return InnerWeightsHelper();
}

template <typename _Derived>
template <typename _DerivedOther>
typename TangentBase<_Derived>::Scalar TangentBase<_Derived>::Inner(
        const TangentBase<_DerivedOther>& t) const {
    return (Coeffs().transpose() * InnerWeightsHelper() * t.Coeffs())(0);
}

template <class _Derived>
typename TangentBase<_Derived>::Scalar TangentBase<_Derived>::WeightedNorm()
        const {
    using std::sqrt;
    return sqrt(SquaredWeightedNorm());
}

template <class _Derived>
typename TangentBase<_Derived>::Scalar
TangentBase<_Derived>::SquaredWeightedNorm() const {
    return (Coeffs().transpose() * InnerWeightsHelper() * Coeffs())(0);
}

template <class _Derived>
typename TangentBase<_Derived>::LieAlg TangentBase<_Derived>::Hat() const {
    return Derived().Hat();
}

template <class _Derived>
typename TangentBase<_Derived>::LieGroup TangentBase<_Derived>::Rplus(
        const LieGroup& m,
        OptJacobianRef J_mout_t,
        OptJacobianRef J_mout_m) const {
    return m.Rplus(Derived(), J_mout_m, J_mout_t);
}

template <class _Derived>
typename TangentBase<_Derived>::LieGroup TangentBase<_Derived>::Lplus(
        const LieGroup& m,
        OptJacobianRef J_mout_t,
        OptJacobianRef J_mout_m) const {
    return m.Lplus(Derived(), J_mout_m, J_mout_t);
}

template <class _Derived>
typename TangentBase<_Derived>::LieGroup TangentBase<_Derived>::Plus(
        const LieGroup& m,
        OptJacobianRef J_mout_t,
        OptJacobianRef J_mout_m) const {
    return m.Lplus(Derived(), J_mout_m, J_mout_t);
}

template <class _Derived>
template <typename _DerivedOther>
typename TangentBase<_Derived>::Tangent TangentBase<_Derived>::Plus(
        const TangentBase<_DerivedOther>& t,
        OptJacobianRef J_mout_ta,
        OptJacobianRef J_mout_tb) const {
    if (J_mout_ta) J_mout_ta->setIdentity();

    if (J_mout_tb) J_mout_tb->setIdentity();

    return *this + t;
}

template <class _Derived>
template <typename _DerivedOther>
typename TangentBase<_Derived>::Tangent TangentBase<_Derived>::Minus(
        const TangentBase<_DerivedOther>& t,
        OptJacobianRef J_mout_ta,
        OptJacobianRef J_mout_tb) const {
    if (J_mout_ta) J_mout_ta->setIdentity();

    if (J_mout_tb) {
        J_mout_tb->setIdentity();
        (*J_mout_tb) *= Scalar(-1);
    }

    return *this - t;
}

template <class _Derived>
typename TangentBase<_Derived>::Jacobian TangentBase<_Derived>::Rjac() const {
    return Derived().Rjac();
}

template <class _Derived>
typename TangentBase<_Derived>::Jacobian TangentBase<_Derived>::Ljac() const {
    return Derived().Ljac();
}

template <class _Derived>
template <typename U>
typename std::enable_if<internal::has_rjacinv<U>::value,
                        typename TangentBase<U>::Jacobian>::type
TangentBase<_Derived>::Rjacinv() const {
    return Derived().Rjacinv();
}

template <class _Derived>
template <typename U>
typename std::enable_if<!internal::has_rjacinv<U>::value,
                        typename TangentBase<U>::Jacobian>::type
TangentBase<_Derived>::Rjacinv() const {
    return Derived().Rjac().Inverse();
}

template <class _Derived>
template <typename U>
typename std::enable_if<internal::has_ljacinv<U>::value,
                        typename TangentBase<U>::Jacobian>::type
TangentBase<_Derived>::Ljacinv() const {
    return Derived().Ljacinv();
}

template <class _Derived>
template <typename U>
typename std::enable_if<!internal::has_ljacinv<U>::value,
                        typename TangentBase<U>::Jacobian>::type
TangentBase<_Derived>::Ljacinv() const {
    return Derived().Ljac().Inverse();
}

template <class _Derived>
typename TangentBase<_Derived>::Jacobian TangentBase<_Derived>::SmallAdj()
        const {
    return Derived().SmallAdj();
}

template <typename _Derived>
template <typename _EigenDerived>
bool TangentBase<_Derived>::IsApprox(const Eigen::MatrixBase<_EigenDerived>& t,
                                     const Scalar eps) const {
    using std::min;
    bool result = false;

    if (min(Coeffs().norm(), t.norm()) < eps) {
        result = ((Coeffs() - t).isZero(eps));
    } else {
        result = (Coeffs().isApprox(t, eps));
    }

    return result;
}

template <typename _Derived>
template <typename _DerivedOther>
bool TangentBase<_Derived>::IsApprox(const TangentBase<_DerivedOther>& t,
                                     const Scalar eps) const {
    return IsApprox(t.Coeffs(), eps);
}

// Operators

template <typename _Derived>
template <typename T>
auto TangentBase<_Derived>::operator<<(T&& v)
        -> decltype(std::declval<DataType>().operator<<(std::forward<T>(v))) {
    return Coeffs().operator<<(std::forward<T>(v));
}

// Static helper

template <class _Derived>
typename TangentBase<_Derived>::Tangent TangentBase<_Derived>::ZeroHelper() {
    static const Tangent t(DataType::Zero());
    return t;
}

template <class _Derived>
typename TangentBase<_Derived>::Tangent TangentBase<_Derived>::RandomHelper() {
    return Tangent().SetRandom();
}

template <typename _Derived>
typename TangentBase<_Derived>::LieAlg TangentBase<_Derived>::GeneratorHelper(
        const int i) {
    return internal::GeneratorEvaluator<
            typename internal::traits<_Derived>::Base>::run(i);
}

template <typename _Derived>
typename TangentBase<_Derived>::InnerWeightsMatrix
TangentBase<_Derived>::InnerWeightsHelper() {
    return internal::InnerWeightsEvaluator<
            typename internal::traits<_Derived>::Base>::run();
}

// Math

template <typename _Derived>
typename TangentBase<_Derived>::Tangent TangentBase<_Derived>::operator-()
        const {
    return Tangent(-Coeffs());
}

template <typename _Derived>
typename TangentBase<_Derived>::LieGroup TangentBase<_Derived>::operator+(
        const LieGroup& m) const {
    return m.Lplus(Derived());
}

template <typename _Derived>
template <typename _DerivedOther>
_Derived& TangentBase<_Derived>::operator+=(
        const TangentBase<_DerivedOther>& t) {
    Coeffs() += t.Coeffs();
    return Derived();
}

template <typename _Derived>
template <typename _DerivedOther>
_Derived& TangentBase<_Derived>::operator-=(
        const TangentBase<_DerivedOther>& t) {
    Coeffs() -= t.Coeffs();
    return Derived();
}

template <typename _Derived, typename _DerivedOther>
typename TangentBase<_Derived>::Tangent operator+(
        const TangentBase<_Derived>& ta, const TangentBase<_DerivedOther>& tb) {
    typename TangentBase<_Derived>::Tangent tc(ta);
    return tc += tb;
}

template <typename _Derived, typename _DerivedOther>
typename TangentBase<_Derived>::Tangent operator-(
        const TangentBase<_Derived>& ta, const TangentBase<_DerivedOther>& tb) {
    typename TangentBase<_Derived>::Tangent tc(ta);
    return tc -= tb;
}

template <typename _Derived>
template <typename _EigenDerived>
_Derived& TangentBase<_Derived>::operator+=(
        const Eigen::MatrixBase<_EigenDerived>& v) {
    Coeffs() += v;
    return Derived();
}

template <typename _Derived>
template <typename _EigenDerived>
_Derived& TangentBase<_Derived>::operator-=(
        const Eigen::MatrixBase<_EigenDerived>& v) {
    Coeffs() -= v;
    return Derived();
}

template <typename _Derived, typename _EigenDerived>
typename TangentBase<_Derived>::Tangent operator+(
        const TangentBase<_Derived>& t,
        const Eigen::MatrixBase<_EigenDerived>& v) {
    typename TangentBase<_Derived>::Tangent ret(t);
    return ret += v;
}

template <typename _Derived, typename _EigenDerived>
typename TangentBase<_Derived>::Tangent operator-(
        const TangentBase<_Derived>& t,
        const Eigen::MatrixBase<_EigenDerived>& v) {
    typename TangentBase<_Derived>::Tangent ret(t);
    return ret -= v;
}

template <typename _EigenDerived, typename _Derived>
auto operator+(const Eigen::MatrixBase<_EigenDerived>& v,
               const TangentBase<_Derived>& t) -> decltype(v + t.Coeffs()) {
    return v + t.Coeffs();
}

template <typename _EigenDerived, typename _Derived>
auto operator-(const Eigen::MatrixBase<_EigenDerived>& v,
               const TangentBase<_Derived>& t) -> decltype(v - t.Coeffs()) {
    return v - t.Coeffs();
}

template <typename _Derived>
typename TangentBase<_Derived>::Tangent TangentBase<_Derived>::operator*=(
        const Scalar scalar) {
    Coeffs() *= scalar;
    return Derived();
}

template <typename _Derived>
typename TangentBase<_Derived>::Tangent TangentBase<_Derived>::operator/=(
        const Scalar scalar) {
    Coeffs() /= scalar;
    return Derived();
}

template <typename _Derived>
typename TangentBase<_Derived>::Tangent operator*(
        const TangentBase<_Derived>& t,
        const typename _Derived::Scalar scalar) {
    typename TangentBase<_Derived>::Tangent ret(t);
    return ret *= scalar;
}

template <typename _Derived>
typename TangentBase<_Derived>::Tangent operator*(
        const typename _Derived::Scalar scalar,
        const TangentBase<_Derived>& t) {
    return t * scalar;
}

template <typename _Derived>
typename TangentBase<_Derived>::Tangent operator/(
        const TangentBase<_Derived>& t,
        const typename _Derived::Scalar scalar) {
    typename TangentBase<_Derived>::Tangent ret(t);
    return ret /= scalar;
}

template <class _DerivedOther>
typename TangentBase<_DerivedOther>::Tangent operator*(
        const typename TangentBase<_DerivedOther>::Jacobian& J,
        const TangentBase<_DerivedOther>& t) {
    return typename TangentBase<_DerivedOther>::Tangent(
            typename TangentBase<_DerivedOther>::DataType(J * t.Coeffs()));
}

template <typename _Derived, typename _DerivedOther>
bool operator==(const TangentBase<_Derived>& ta,
                const TangentBase<_DerivedOther>& tb) {
    return ta.IsApprox(tb);
}

template <typename _Derived, typename _EigenDerived>
bool operator==(const TangentBase<_Derived>& t,
                const Eigen::MatrixBase<_EigenDerived>& v) {
    return t.IsApprox(v);
}

// Utils

template <typename _Stream, typename _Derived>
_Stream& operator<<(_Stream& s,
                    const holistic_motion::robotics::TangentBase<_Derived>& m) {
    s << m.Coeffs().transpose();
    return s;
}

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_TANGENT_BASE_H_ */
