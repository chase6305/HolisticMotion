#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/kinematics/fep/FEPKinematics.h"
#include "holistic_motion/kinematics/srs/SRSKinematics.h"

using namespace holistic_motion::robotics;

namespace {
std::atomic<std::size_t> allocations{0};
}

// Count ordinary C++ allocations in this executable only. Aligned allocations
// and direct malloc calls are outside this diagnostic counter.
void *operator new(std::size_t size) {
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void *memory = std::malloc(std::max(size, std::size_t{1})))
        return memory;
    throw std::bad_alloc();
}
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

namespace {
std::vector<JointNode> MakeModel(int dof, bool mixed) {
    std::vector<JointNode> nodes;
    for (int i = 0; i < dof; ++i)
        nodes.emplace_back(
            SE3d(Eigen::Vector3d(0.02 * i, -0.01, 0.03), SO3d(0.01 * i, -0.02, 0.03)),
            Eigen::Vector3d(1 + i % 3, 2 + i % 2, 3).normalized() * (i % 2 ? 1.5 : 0.5),
            mixed && i % 3 == 2 ? JointType::PRISMATIC : JointType::REVOLUTE, -100.0,
            100.0);
    nodes.emplace_back(
        SE3d(Eigen::Vector3d(0.03, -0.01, 0.04), SO3d(0.02, 0.03, -0.01)),
        Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0);
    return nodes;
}

template <class Solver> bool Measure(const char *name, int dof, bool mixed) {
    Solver solver(MakeModel(dof, mixed));
    solver.SetTCP(SE3d(Eigen::Vector3d(0.1, 0.2, -0.3), SO3d(0.2, -0.3, 0.4)));
    std::vector<Eigen::VectorXd> states(64, Eigen::VectorXd(dof));
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < dof; ++j)
            states[i][j] = 0.3 * std::sin(0.2 * i + 0.3 * j);
        SE3d pose;
        std::vector<SE3d> trace;
        if (!solver.GetFK(states[i], pose) || !solver.GetAllFK(states[i], trace) ||
            !pose.GetTransform().isApprox(trace.back().GetTransform(), 1e-14))
            return false;
    }
    constexpr int calls = 65536;
    SE3d pose;
    std::vector<double> elapsed, counts;
    double checksum = 0.0;
    for (int sample = 0; sample < 7; ++sample) {
        const auto before = allocations.load(std::memory_order_relaxed);
        const auto started = std::chrono::steady_clock::now();
        double sum = 0.0;
        for (int call = 0; call < calls; ++call) {
            if (!solver.GetFK(states[call % states.size()], pose))
                return false;
            sum += pose.GetTranslation().x();
        }
        const auto finished = std::chrono::steady_clock::now();
        const auto allocated = allocations.load(std::memory_order_relaxed) - before;
        elapsed.push_back(
            std::chrono::duration<double, std::nano>(finished - started).count() /
            calls);
        counts.push_back(static_cast<double>(allocated) / calls);
        checksum = sum;
    }
    std::sort(elapsed.begin(), elapsed.end());
    std::sort(counts.begin(), counts.end());
    std::cout << name << ',' << dof << ',' << calls << ',' << elapsed[3] << ','
              << counts[3] << ',' << checksum << '\n';
    return true;
}
} // namespace

int main() {
    std::cout << std::setprecision(12)
              << "solver,dof,calls_per_sample,median_ns,new_calls_per_query,checksum\n";
    for (int dof : {0, 1, 3, 7, 14, 64})
        if (!Measure<NumericalKinematics>("numerical", dof, true))
            return 1;
    if (!Measure<SRSKinematics>("srs", 7, false) ||
        !Measure<FEPKinematics>("fep", 7, false))
        return 1;
}
