#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"

using namespace holistic_motion::robotics;

class CountedKinematics : public NumericalKinematics {
public:
    using NumericalKinematics::NumericalKinematics;
    bool GetAllFK(const Eigen::VectorXd& joints,
                  std::vector<SE3d>& poses) const override {
        ++fk_evaluations;
        return NumericalKinematics::GetAllFK(joints, poses);
    }
    mutable std::size_t fk_evaluations{0};
};

int main() {
    std::cout
        << "dof,scenario,median_us,success,fk_evaluations,residual,checksum\n"
        << std::setprecision(12);
    for (int dof : {1, 3, 6, 7, 14}) {
        std::vector<JointNode> nodes;
        for (int joint = 0; joint < dof; ++joint) {
            nodes.emplace_back(SE3d(Eigen::Vector3d(0.02, -0.01 * joint, 0.1),
                                    SO3d(0.01 * joint, -0.02, 0.03)),
                               Eigen::Vector3d::Unit(joint % 3),
                               JointType::REVOLUTE, -2.0, 2.0);
        }
        nodes.emplace_back(SE3d(Eigen::Vector3d(0.03, -0.02, 0.08), SO3d()),
                           Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0);
        CountedKinematics solver(nodes);
        solver.SetMaxIterNum(60);
        solver.SetTCP(
            SE3d(Eigen::Vector3d(0.04, 0.02, 0.03), SO3d(0.2, -0.1, 0.3)));
        Eigen::VectorXd expected(dof);
        for (int joint = 0; joint < dof; ++joint)
            expected[joint] = (joint % 2 ? -1 : 1) * (0.15 + 0.015 * joint);
        SE3d reachable;
        if (!solver.GetFK(expected, reachable)) return 1;
        for (int scenario = 0; scenario < 3; ++scenario) {
            const char* name = scenario == 0   ? "exact_seed"
                               : scenario == 1 ? "nearby"
                                               : "unreachable";
            SE3d target = reachable;
            if (scenario == 2)
                target = SE3d(
                    reachable.GetTranslation() + Eigen::Vector3d(10, 10, 10),
                    reachable.GetRotation());
            const Eigen::VectorXd initial =
                scenario == 0 ? expected : Eigen::VectorXd::Zero(dof).eval();
            IkRtn solutions;
            std::vector<double> distances;
            Eigen::VectorXd seed = initial;
            auto solve = [&]() {
                seed = initial;
                solver.fk_evaluations = 0;
                return solver.GetIK(target, solutions, seed, distances);
            };
            const bool success = solve();
            const std::size_t evaluations = solver.fk_evaluations;
            double checksum = 0.0;
            double residual = 0.0;
            if (success) {
                checksum = solutions.ik_joints.front().sum();
                SE3d actual;
                if (!solver.GetFK(solutions.ik_joints.front(), actual))
                    return 1;
                residual =
                    (actual.GetTransform() - target.GetTransform()).norm();
            }
            if (success != (scenario != 2)) return 1;
            std::vector<double> timings;
            const int calls = scenario == 0   ? 16384
                              : scenario == 1 ? 1024
                                              : 128;
            // Keep sub-microsecond exact-seed measurements above timer and
            // scheduling noise, and warm the same workload before recording.
            for (int sample = -1; sample < 7; ++sample) {
                const auto started = std::chrono::steady_clock::now();
                for (int call = 0; call < calls; ++call) {
                    if (solve() != success ||
                        solver.fk_evaluations != evaluations)
                        return 1;
                }
                if (sample >= 0)
                    timings.push_back(
                        std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - started)
                            .count() /
                        calls);
            }
            std::sort(timings.begin(), timings.end());
            std::cout << dof << ',' << name << ',' << timings[3] << ','
                      << success << ',' << evaluations << ',' << residual << ','
                      << checksum << '\n';
        }
    }
}
