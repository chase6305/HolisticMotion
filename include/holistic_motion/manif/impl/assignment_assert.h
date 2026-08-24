#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_IMPL_ASSIGNMENT_ASSERT_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_IMPL_ASSIGNMENT_ASSERT_H_

namespace holistic_motion {
namespace robotics {
namespace internal {

template <typename Derived>
struct AssignmentEvaluatorImpl {
    template <typename T>
    static void run_impl(const T&) {}
};

template <typename Derived>
struct AssignmentEvaluator : AssignmentEvaluatorImpl<Derived> {
    using Base = AssignmentEvaluatorImpl<Derived>;

    template <typename T>
    void run(T&& t) {
        Base::run_impl(std::forward<T>(t));
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif  // _HOLISTIC_MOTION_HOLISTIC_MOTION_IMPL_ASSIGNMENT_ASSERT_H_
