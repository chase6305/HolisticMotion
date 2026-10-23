// Synthetic immutable-space copies and complete short planning queries.
// Allocation counting runs separately from timing. Linux builds also wrap
// malloc so Eigen's direct allocations are included; other platforms count
// ordinary C++ operator new. No robot asset or additional dependency is needed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"

using namespace holistic_motion::robotics::planning;

namespace {
bool count_allocations = false;
std::size_t allocations = 0;
#ifdef HOLISTICMOTION_COUNT_MALLOC
constexpr const char *kAllocationKind = "malloc";
#else
constexpr const char *kAllocationKind = "cpp_new";
#endif
} // namespace
#ifdef HOLISTICMOTION_COUNT_MALLOC
extern "C" void *__real_malloc(std::size_t);
extern "C" void *__wrap_malloc(std::size_t n) {
    if (count_allocations)
        ++allocations;
    return __real_malloc(n);
}
#endif
void *operator new(std::size_t n) {
#ifndef HOLISTICMOTION_COUNT_MALLOC
    if (count_allocations)
        ++allocations;
#endif
    if (void *p = std::malloc(std::max(n, std::size_t{1})))
        return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
JointSpace
CopySpace(const JointSpace &space) {
    return space;
}

template <class F> void Measure(const char *operation, int dof, F call) {
    double checksum = 0;
    constexpr int calls = 32768;
    count_allocations = true;
    allocations = 0;
    for (int i = 0; i < 128; ++i)
        checksum += call();
    const double alloc_per_call = double(allocations) / 128;
    count_allocations = false;
    std::vector<double> elapsed;
    for (int sample = 0; sample < 7; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < calls; ++i)
            checksum += call();
        elapsed.push_back(std::chrono::duration<double, std::nano>(
                              std::chrono::steady_clock::now() - start)
                              .count() /
                          calls);
    }
    std::sort(elapsed.begin(), elapsed.end());
    std::cout << operation << ',' << dof << ',' << elapsed[3] << ',' << alloc_per_call
              << ',' << kAllocationKind << ',' << checksum << '\n';
}
int main() {
    std::cout << std::setprecision(12)
              << "operation,dof,median_ns,allocation_calls,allocation_kind,checksum\n";
    for (int dof : {2, 7, 64, 256}) {
        std::vector<std::string> names;
        for (int i = 0; i < dof; ++i)
            names.push_back("robot_chain_joint_number_" + std::to_string(i));
        const JointSpace space(Eigen::VectorXd::Constant(dof, -4),
                               Eigen::VectorXd::Constant(dof, 4), names);
        Measure("copy", dof, [&] {
            const auto copied = CopySpace(space);
            return copied.GetJointWeights()[0] + copied.GetJointNames().size();
        });
        const SamplingPlanner planner(space);
        const Eigen::VectorXd start = Eigen::VectorXd::Constant(dof, -.5),
                              goal = -start;
        PlanningOptions options;
        options.edge_resolution = .2;
        options.simplify_path = false;
        Measure("direct_plan", dof, [&] {
            auto result = planner.Plan(start, goal, options);
            if (!result.Success() || result.path.size() != 2)
                std::abort();
            return result.statistics.final_path_length + result.joint_space->GetDOF();
        });
        const PathOptimizer optimizer(space);
        const std::vector<Eigen::VectorXd> path{start, goal};
        PathOptimizationOptions optimization;
        Measure("unchanged_optimize", dof, [&] {
            auto result = optimizer.Optimize(path, optimization);
            if (!result.Success())
                std::abort();
            return result.statistics.final_objective + result.joint_space->GetDOF();
        });
    }
}
