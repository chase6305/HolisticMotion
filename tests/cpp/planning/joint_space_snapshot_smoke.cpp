#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <utility>
#include <vector>

#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"

using namespace holistic_motion::robotics::planning;

int main() {
    const double pi = std::acos(-1.0);
    JointSpace original(Eigen::Vector2d(-pi, -1.0), Eigen::Vector2d(pi, 1.0),
                        {"turn", "slide"});
    original = original.WithContinuousJoints({0});
    JointSpace copied = original;
    JointSpace moved = std::move(copied);
    // Both immutable handles remain useful; a move must not leave a null
    // snapshot behind inside a moved-from planner or optional result.
    if (copied.GetDOF() != 2 || copied.GetJointNames() != moved.GetJointNames() ||
        copied.GetContinuousJoints() != std::vector<std::size_t>{0})
        return 1;
    auto &alias = copied;
    copied = std::move(alias);
    if (copied.GetDOF() != 2)
        return 2;

    const auto changed =
        copied.WithJointWeights(Eigen::Vector2d(4.0, 0.25)).WithContinuousJoints({});
    if (!changed.GetContinuousJoints().empty() ||
        original.GetContinuousJoints() != std::vector<std::size_t>{0} ||
        !original.GetJointWeights().isApprox(moved.GetJointWeights()) ||
        changed.GetJointWeights().isApprox(original.GetJointWeights()))
        return 3;

    const std::vector<Eigen::VectorXd> path{Eigen::Vector2d(3.0, 0.0),
                                            Eigen::Vector2d(-3.0, 0.5)};
    SamplingPlanner planner(original);
    PathOptimizer optimizer(planner.GetJointSpace());
    auto planned = planner.Plan(path.front(), path.back());
    auto optimized = optimizer.Optimize(planned.path);
    planner.SetContinuousJoints({});
    planner.SetJointWeights(Eigen::Vector2d(9.0, 9.0));
    if (!planned.Success() || !optimized.Success() ||
        planned.joint_space->GetContinuousJoints() != std::vector<std::size_t>{0} ||
        optimized.joint_space->GetContinuousJoints() != std::vector<std::size_t>{0})
        return 4;
    const auto lifted = optimized.joint_space->UnwrapPath(path);
    if (std::abs(lifted.back()[0] - (2.0 * pi - 3.0)) > 1e-12)
        return 5;

    std::array<std::future<bool>, 4> workers;
    for (std::size_t worker = 0; worker < workers.size(); ++worker)
        workers[worker] = std::async(std::launch::async, [&, worker] {
            for (int request = 0; request < 64; ++request) {
                const JointSpace snapshot = original;
                const auto local = snapshot.WithJointWeights(
                    Eigen::Vector2d(worker + 1.0, request + 1.0));
                if (local.GetJointWeights()[0] != worker + 1.0 ||
                    local.GetJointWeights()[1] != request + 1.0 ||
                    !snapshot.GetJointWeights().isApprox(moved.GetJointWeights()) ||
                    !local.UnwrapPath(path).back().isApprox(lifted.back()))
                    return false;
            }
            return true;
        });
    for (auto &worker : workers)
        if (!worker.get()) {
            std::cerr << "joint-space snapshots shared mutable state\n";
            return 6;
        }
}
