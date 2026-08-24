#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_FWD_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_FWD_H_

#include <stdexcept>  // for std::runtime_error
#include <utility>    // for std::forward

#ifdef NDEBUG
#ifndef HOLISTIC_MOTION_NO_DEBUG
#define HOLISTIC_MOTION_NO_DEBUG
#endif
#endif

namespace holistic_motion {
namespace robotics {

struct runtime_error : std::runtime_error {
    using std::runtime_error::runtime_error;
    using std::runtime_error::what;
};

struct invalid_argument : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
    using std::invalid_argument::what;
};

namespace detail {

template <typename E, typename... Args>
void
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((noinline, cold, noreturn))
#elif defined(_MSC_VER)
        __declspec(noinline, noreturn)
#else
// nothing
#endif
        raise(Args&&... args) {
    throw E(std::forward<Args>(args)...);
}
}  // namespace detail
}  // namespace robotics
}  // namespace holistic_motion

#define HOLISTIC_MOTION_UNUSED_VARIABLE(x) EIGEN_UNUSED_VARIABLE(x)

// gcc expands __VA_ARGS___ before passing it into the macro.
// Visual Studio expands __VA_ARGS__ after passing it.
// This macro is a workaround to support both
#define __HOLISTIC_MOTION_EXPAND(x) x

#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define __HOLISTIC_MOTION_HAVE_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define __HOLISTIC_MOTION_HAVE_CPP_ATTRIBUTE(x) 0
#endif

#define __HOLISTIC_MOTION_THROW_EXCEPT(msg, except) \
    holistic_motion::robotics::detail::raise<except>(msg);
#define __HOLISTIC_MOTION_THROW(msg) \
    __HOLISTIC_MOTION_THROW_EXCEPT(msg, holistic_motion::robotics::runtime_error)

#define __HOLISTIC_MOTION_GET_MACRO_2(_1, _2, NAME, ...) NAME

#define HOLISTIC_MOTION_THROW(...)                                                \
    __HOLISTIC_MOTION_EXPAND(__HOLISTIC_MOTION_GET_MACRO_2(__VA_ARGS__, __HOLISTIC_MOTION_THROW_EXCEPT, \
                                     __HOLISTIC_MOTION_THROW)(__VA_ARGS__))

#define __HOLISTIC_MOTION_CHECK_MSG_EXCEPT(cond, msg, except) \
    if (!(cond)) {                                 \
        HOLISTIC_MOTION_THROW(msg, except);                   \
    }
#define __HOLISTIC_MOTION_CHECK_MSG(cond, msg) \
    __HOLISTIC_MOTION_CHECK_MSG_EXCEPT(cond, msg, holistic_motion::robotics::runtime_error)
#define __HOLISTIC_MOTION_CHECK(cond)                                          \
    __HOLISTIC_MOTION_CHECK_MSG_EXCEPT(cond, "Condition: '" #cond "' failed!", \
                            holistic_motion::robotics::runtime_error)

#define __HOLISTIC_MOTION_GET_MACRO_3(_1, _2, _3, NAME, ...) NAME

#define HOLISTIC_MOTION_CHECK(...)                                                    \
    __HOLISTIC_MOTION_EXPAND(__HOLISTIC_MOTION_GET_MACRO_3(__VA_ARGS__, __HOLISTIC_MOTION_CHECK_MSG_EXCEPT, \
                                     __HOLISTIC_MOTION_CHECK_MSG,                     \
                                     __HOLISTIC_MOTION_CHECK)(__VA_ARGS__))

// Assertions cost run time and can be turned off.
// You can suppress HOLISTIC_MOTION_ASSERT by defining
// HOLISTIC_MOTION_NO_DEBUG before including manif headers.
// HOLISTIC_MOTION_NO_DEBUG is undefined by default unless NDEBUG is defined.
#ifndef HOLISTIC_MOTION_NO_DEBUG
#define HOLISTIC_MOTION_ASSERT(...)                                                   \
    __HOLISTIC_MOTION_EXPAND(__HOLISTIC_MOTION_GET_MACRO_3(__VA_ARGS__, __HOLISTIC_MOTION_CHECK_MSG_EXCEPT, \
                                     __HOLISTIC_MOTION_CHECK_MSG,                     \
                                     __HOLISTIC_MOTION_CHECK)(__VA_ARGS__))
#else
#define HOLISTIC_MOTION_ASSERT(...) ((void)0)
#endif

#define HOLISTIC_MOTION_NOT_IMPLEMENTED_YET HOLISTIC_MOTION_THROW("Not implemented yet !");

#if defined(__cplusplus) && (__cplusplus >= 201402L) && \
        __HOLISTIC_MOTION_HAVE_CPP_ATTRIBUTE(deprecated)
#define HOLISTIC_MOTION_DEPRECATED [[deprecated]]
#elif defined(__GNUC__) || defined(__clang__)
#define HOLISTIC_MOTION_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define HOLISTIC_MOTION_DEPRECATED __declspec(deprecated)
#else
#pragma message(                            \
        "WARNING: Deprecation is disabled " \
        "-- the compiler is not supported.")
#define HOLISTIC_MOTION_DEPRECATED
#endif

// Common macros

#define HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND \
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(     \
            (Eigen::internal::traits<typename Base::DataType>::Alignment > 0))
#define HOLISTIC_MOTION_MAKE_ALIGNED_OPERATOR_NEW_COND_TYPE(X) \
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(             \
            (Eigen::internal::traits<typename X::DataType>::Alignment > 0))

#define HOLISTIC_MOTION_MOVE_NOEXCEPT \
    noexcept(std::is_nothrow_move_constructible<Scalar>::value)

#define HOLISTIC_MOTION_DEFAULT_CONSTRUCTOR(X) \
    X() = default;                  \
    ~X() = default;                 \
    X(const X&) = default;          \
    X(X&&) = default;

#define HOLISTIC_MOTION_GROUP_ML_ASSIGN_OP(X)                                            \
    _Derived& operator=(const X& o) {                                         \
        Coeffs() = o.Coeffs();                                                \
        return Derived();                                                     \
    }                                                                         \
    template <typename _DerivedOther>                                         \
    _Derived& operator=(const LieGroupBase<_DerivedOther>& o) {               \
        Coeffs() = o.Coeffs();                                                \
        return Derived();                                                     \
    }                                                                         \
    template <typename _EigenDerived>                                         \
    _Derived& operator=(const Eigen::MatrixBase<_EigenDerived>& o) {          \
        Coeffs() = o;                                                         \
        return Derived();                                                     \
    }                                                                         \
    _Derived& operator=(X&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {                           \
        Coeffs() = std::move(o.Coeffs());                                     \
        return Derived();                                                     \
    }                                                                         \
    template <typename _DerivedOther>                                         \
    _Derived& operator=(LieGroupBase<_DerivedOther>&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT { \
        Coeffs() = std::move(o.Coeffs());                                     \
        return Derived();                                                     \
    }                                                                         \
    template <typename _EigenDerived>                                         \
    _Derived& operator=(Eigen::MatrixBase<_EigenDerived>&& o) {               \
        Coeffs() = std::move(o);                                              \
        return Derived();                                                     \
    }

#define HOLISTIC_MOTION_GROUP_ASSIGN_OP(X)                                        \
    X& operator=(const X& o) {                                         \
        Coeffs() = o.Coeffs();                                         \
        return Derived();                                              \
    }                                                                  \
    template <typename _DerivedOther>                                  \
    X& operator=(const X##Base<_DerivedOther>& o) {                    \
        Coeffs() = o.Coeffs();                                         \
        return Derived();                                              \
    }                                                                  \
    template <typename _DerivedOther>                                  \
    X& operator=(const LieGroupBase<_DerivedOther>& o) {               \
        Coeffs() = o.Coeffs();                                         \
        return Derived();                                              \
    }                                                                  \
    template <typename _EigenDerived>                                  \
    X& operator=(const Eigen::MatrixBase<_EigenDerived>& o) {          \
        Coeffs() = o;                                                  \
        return Derived();                                              \
    }                                                                  \
    X& operator=(X&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {                           \
        Coeffs() = std::move(o.Coeffs());                              \
        return Derived();                                              \
    }                                                                  \
    template <typename _DerivedOther>                                  \
    X& operator=(X##Base<_DerivedOther>&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {      \
        Coeffs() = std::move(o.Coeffs());                              \
        return Derived();                                              \
    }                                                                  \
    template <typename _DerivedOther>                                  \
    X& operator=(LieGroupBase<_DerivedOther>&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT { \
        Coeffs() = std::move(o.Coeffs());                              \
        return Derived();                                              \
    }                                                                  \
    template <typename _EigenDerived>                                  \
    X& operator=(Eigen::MatrixBase<_EigenDerived>&& o) {               \
        Coeffs() = std::move(o);                                       \
        return Derived();                                              \
    }

#define HOLISTIC_MOTION_GROUP_MAP_ASSIGN_OP(X)                                        \
    Map(const Map& o) : Base(), data_(o.Coeffs()) {}                       \
    Map& operator=(const Map& o) {                                         \
        Coeffs() = o.Coeffs();                                             \
        return *this;                                                      \
    }                                                                      \
    template <typename _DerivedOther>                                      \
    Map& operator=(const holistic_motion::robotics::X##Base<_DerivedOther>& o) {      \
        Coeffs() = o.Coeffs();                                             \
        return *this;                                                      \
    }                                                                      \
    template <typename _DerivedOther>                                      \
    Map& operator=(const holistic_motion::robotics::LieGroupBase<_DerivedOther>& o) { \
        Coeffs() = o.Coeffs();                                             \
        return *this;                                                      \
    }                                                                      \
    template <typename _EigenDerived>                                      \
    Map& operator=(const Eigen::MatrixBase<_EigenDerived>& o) {            \
        Coeffs() = o;                                                      \
        return *this;                                                      \
    }                                                                      \
    Map& operator=(Map&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {                           \
        Coeffs() = std::move(o.Coeffs());                                  \
        return *this;                                                      \
    }                                                                      \
    template <typename _DerivedOther>                                      \
    Map& operator=(holistic_motion::robotics::X##Base<_DerivedOther>&& o)             \
            HOLISTIC_MOTION_MOVE_NOEXCEPT {                                           \
        Coeffs() = std::move(o.Coeffs());                                  \
        return *this;                                                      \
    }                                                                      \
    template <typename _DerivedOther>                                      \
    Map& operator=(holistic_motion::robotics::LieGroupBase<_DerivedOther>&& o)        \
            HOLISTIC_MOTION_MOVE_NOEXCEPT {                                           \
        Coeffs() = std::move(o.Coeffs());                                  \
        return *this;                                                      \
    }                                                                      \
    template <typename _EigenDerived>                                      \
    Map& operator=(Eigen::MatrixBase<_EigenDerived>&& o) {                 \
        Coeffs() = std::move(o);                                           \
        return *this;                                                      \
    }

#define HOLISTIC_MOTION_TANGENT_ML_ASSIGN_OP(X)                                         \
    _Derived& operator=(const X& o) {                                        \
        Coeffs() = o.Coeffs();                                               \
        return Derived();                                                    \
    }                                                                        \
    template <typename _DerivedOther>                                        \
    _Derived& operator=(const TangentBase<_DerivedOther>& o) {               \
        Coeffs() = o.Coeffs();                                               \
        return Derived();                                                    \
    }                                                                        \
    template <typename _EigenDerived>                                        \
    _Derived& operator=(const Eigen::MatrixBase<_EigenDerived>& o) {         \
        Coeffs() = o;                                                        \
        return Derived();                                                    \
    }                                                                        \
    _Derived& operator=(X&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {                          \
        Coeffs() = std::move(o.Coeffs());                                    \
        return Derived();                                                    \
    }                                                                        \
    template <typename _DerivedOther>                                        \
    _Derived& operator=(TangentBase<_DerivedOther>&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT { \
        Coeffs() = std::move(o.Coeffs());                                    \
        return Derived();                                                    \
    }                                                                        \
    template <typename _EigenDerived>                                        \
    _Derived& operator=(Eigen::MatrixBase<_EigenDerived>&& o)                \
            HOLISTIC_MOTION_MOVE_NOEXCEPT {                                             \
        Coeffs() = std::move(o);                                             \
        return Derived();                                                    \
    }

#define HOLISTIC_MOTION_TANGENT_ASSIGN_OP(X)                                           \
    X& operator=(const X& o) {                                              \
        Coeffs() = o.Coeffs();                                              \
        return Derived();                                                   \
    }                                                                       \
    template <typename _DerivedOther>                                       \
    X& operator=(const X##Base<_DerivedOther>& o) {                         \
        Coeffs() = o.Coeffs();                                              \
        return Derived();                                                   \
    }                                                                       \
    template <typename _DerivedOther>                                       \
    X& operator=(const TangentBase<_DerivedOther>& o) {                     \
        Coeffs() = o.Coeffs();                                              \
        return Derived();                                                   \
    }                                                                       \
    template <typename _EigenDerived>                                       \
    X& operator=(const Eigen::MatrixBase<_EigenDerived>& o) {               \
        Coeffs() = o;                                                       \
        return Derived();                                                   \
    }                                                                       \
    X& operator=(X&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {                                \
        Coeffs() = std::move(o.Coeffs());                                   \
        return Derived();                                                   \
    }                                                                       \
    template <typename _DerivedOther>                                       \
    X& operator=(X##Base<_DerivedOther>&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {           \
        Coeffs() = std::move(o.Coeffs());                                   \
        return Derived();                                                   \
    }                                                                       \
    template <typename _DerivedOther>                                       \
    X& operator=(TangentBase<_DerivedOther>&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {       \
        Coeffs() = std::move(o.Coeffs());                                   \
        return Derived();                                                   \
    }                                                                       \
    template <typename _EigenDerived>                                       \
    X& operator=(Eigen::MatrixBase<_EigenDerived>&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT { \
        Coeffs() = std::move(o);                                            \
        return Derived();                                                   \
    }

#define HOLISTIC_MOTION_TANGENT_MAP_ASSIGN_OP(X)                                         \
    Map(const Map& o) : Base(), data_(o.Coeffs()) {}                          \
    Map& operator=(const Map& o) {                                            \
        Coeffs() = o.Coeffs();                                                \
        return *this;                                                         \
    }                                                                         \
    template <typename _DerivedOther>                                         \
    Map& operator=(const holistic_motion::robotics::X##Base<_DerivedOther>& o) {         \
        Coeffs() = o.Coeffs();                                                \
        return *this;                                                         \
    }                                                                         \
    template <typename _DerivedOther>                                         \
    Map& operator=(const holistic_motion::robotics::TangentBase<_DerivedOther>& o) {     \
        Coeffs() = o.Coeffs();                                                \
        return *this;                                                         \
    }                                                                         \
    template <typename _EigenDerived>                                         \
    Map& operator=(const Eigen::MatrixBase<_EigenDerived>& o) {               \
        Coeffs() = o;                                                         \
        return *this;                                                         \
    }                                                                         \
    Map& operator=(Map&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT {                              \
        Coeffs() = std::move(o.Coeffs());                                     \
        return *this;                                                         \
    }                                                                         \
    template <typename _DerivedOther>                                         \
    Map& operator=(holistic_motion::robotics::X##Base<_DerivedOther>&& o)                \
            HOLISTIC_MOTION_MOVE_NOEXCEPT {                                              \
        Coeffs() = std::move(o.Coeffs());                                     \
        return *this;                                                         \
    }                                                                         \
    template <typename _DerivedOther>                                         \
    Map& operator=(holistic_motion::robotics::TangentBase<_DerivedOther>&& o)            \
            HOLISTIC_MOTION_MOVE_NOEXCEPT {                                              \
        Coeffs() = std::move(o.Coeffs());                                     \
        return *this;                                                         \
    }                                                                         \
    template <typename _EigenDerived>                                         \
    Map& operator=(Eigen::MatrixBase<_EigenDerived>&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT { \
        Coeffs() = std::move(o);                                              \
        return *this;                                                         \
    }

/**
 * @brief Automatically define:
 * - copy constructor
 * - copy constructor given Base object
 * - copy constructor given Eigen object
 */
#define HOLISTIC_MOTION_COPY_CONSTRUCTOR(X)                                          \
    X(const X& o) : Base(), data_(o.Coeffs()) {}                          \
    X(const Base& o) : Base(), data_(o.Coeffs()) {}                       \
    template <typename D>                                                 \
    X(const Eigen::MatrixBase<D>& o) : Base(), data_(o) {                 \
        holistic_motion::robotics::internal::AssignmentEvaluator<Base>().run(data_); \
    }

#define HOLISTIC_MOTION_MOVE_CONSTRUCTOR(X)                                             \
    X(X&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT : Base(), data_(std::move(o.Coeffs())) {}    \
    X(Base&& o) HOLISTIC_MOTION_MOVE_NOEXCEPT : Base(), data_(std::move(o.Coeffs())) {} \
    template <typename D>                                                    \
    X(Eigen::MatrixBase<D>&& o) : Base(), data_(std::move(o)) {              \
        holistic_motion::robotics::internal::AssignmentEvaluator<Base>().run(data_);    \
    }

#define HOLISTIC_MOTION_COEFFS_FUNCTIONS()           \
    DataType& Coeffs()& { return data_; } \
    const DataType& Coeffs() const& { return data_; }

// LieGroup - related macros

#define HOLISTIC_MOTION_INHERIT_GROUP_AUTO_API using Base::SetRandom;
   // using Base::Rplus;
   // using Base::Lplus;
   // using Base::Rminus;
   // using Base::Lminus;
   // using Base::Between; # todo: unuse, In the MSVC compiler, the template
   // derived class using declares the function of the template base class but
   // does not overload it, which will cause the problem of not being able to
   // find the corresponding overloaded function at the pybind level;

#define HOLISTIC_MOTION_INHERIT_GROUP_API  \
    HOLISTIC_MOTION_INHERIT_GROUP_AUTO_API \
    using Base::SetIdentity;    \
    using Base::Inverse;        \
    using Base::Lift;           \
    using Base::Log;            \
    using Base::Adj;

#define HOLISTIC_MOTION_INHERIT_GROUP_OPERATOR \
    using Base::operator+;          \
    using Base::operator+=;         \
    using Base::operator-;          \
    using Base::operator*;          \
    using Base::operator*=;         \
    using Base::operator=;

#define HOLISTIC_MOTION_GROUP_PROPERTIES \
    using Base::Dim;          \
    using Base::DoF;

#define HOLISTIC_MOTION_GROUP_TYPEDEF                    \
    HOLISTIC_MOTION_GROUP_PROPERTIES                     \
    using Scalar = typename Base::Scalar;     \
    using LieGroup = typename Base::LieGroup; \
    using Tangent = typename Base::Tangent;   \
    using Jacobian = typename Base::Jacobian; \
    using DataType = typename Base::DataType; \
    using Vector = typename Base::Vector;     \
    using OptJacobianRef = typename Base::OptJacobianRef;

#define HOLISTIC_MOTION_COMPLETE_GROUP_TYPEDEF \
    HOLISTIC_MOTION_GROUP_TYPEDEF              \
    HOLISTIC_MOTION_INHERIT_GROUP_OPERATOR

#define HOLISTIC_MOTION_EXTRA_GROUP_TYPEDEF(group) \
    using group##f = group<float>;      \
    using group##d = group<double>;

// Tangent - related macros

#define HOLISTIC_MOTION_INHERIT_TANGENT_API \
    using Base::SetZero;         \
    using Base::SetRandom;       \
    using Base::Retract;         \
    using Base::Exp;             \
    using Base::Hat;             \
    using Base::Rjac;            \
    using Base::Ljac;            \
    using Base::SmallAdj;

#define HOLISTIC_MOTION_INHERIT_TANGENT_OPERATOR \
    using Base::operator+=;           \
    using Base::operator-=;           \
    using Base::operator*=;           \
    using Base::operator/=;           \
    using Base::operator=;            \
    using Base::operator<<;

#define HOLISTIC_MOTION_TANGENT_PROPERTIES \
    using Base::Dim;            \
    using Base::DoF;

#define HOLISTIC_MOTION_TANGENT_TYPEDEF                  \
    HOLISTIC_MOTION_TANGENT_PROPERTIES                   \
    using Scalar = typename Base::Scalar;     \
    using LieGroup = typename Base::LieGroup; \
    using Tangent = typename Base::Tangent;   \
    using Jacobian = typename Base::Jacobian; \
    using DataType = typename Base::DataType; \
    using LieAlg = typename Base::LieAlg;     \
    using OptJacobianRef = typename Base::OptJacobianRef;

#define HOLISTIC_MOTION_EXTRA_TANGENT_TYPEDEF(tangent) \
    using tangent##f = tangent<float>;      \
    using tangent##d = tangent<double>;

#define HOLISTIC_MOTION_GROUP_INSTANTIATIONS(group) \
    template class group<R6d>;           \
    template class group<R7d>;           \
    template class group<SE3d>;

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_FWD_H_ */
