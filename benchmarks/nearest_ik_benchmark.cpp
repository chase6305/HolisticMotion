#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"

using namespace holistic_motion::robotics;

int main() {
    holistic_motion::utility::SetVerbosityLevel(
        holistic_motion::utility::VerbosityLevel::Error);
    std::cout << "dof,scenario,median_us,success,solutions,distance,checksum\n"
              << std::setprecision(17);
    for (int dof : {1, 7, 14, 17, 32}) {
        for (int scenario = 0; scenario < 3; ++scenario) {
            const bool wide = scenario != 0;
            std::vector<JointNode> nodes(
                dof,
                JointNode(SE3d(), Eigen::Vector3d::UnitZ(), JointType::REVOLUTE,
                          wide ? 0.0 : -1.0, wide ? 2.0 * M_PI : 1.0));
            nodes.emplace_back();
            nodes.back().joint_type = JointType::FIXED;
            NumericalKinematics solver(nodes);
            if (scenario == 2)
                solver.SetMaxIterNum(1);
            const double joint = scenario == 0 ? 0.2 : 0.0;
            const Eigen::VectorXd expected =
                Eigen::VectorXd::Constant(dof, joint);
            const Eigen::VectorXd initial = Eigen::VectorXd::Constant(
                dof, scenario == 2 ? 2.0 * M_PI - 0.1 : joint);
            SE3d target;
            if (!solver.GetFK(expected, target))
                return 1;
            IkRtn solutions;
            Eigen::VectorXd seed;
            double distance = -1.0;
            const auto solve = [&]() {
                seed = initial;
                distance = -1.0;
                return solver.GetNearestIK(target, solutions, seed, distance);
            };
            const bool success = solve();
            const int count = solutions.ik_number;
            const double reference_distance = distance;
            double checksum = 0.0;
            if (success) {
                SE3d actual;
                if (count != 1 || !seed.isApprox(initial) ||
                    !solver.GetFK(solutions.ik_joints.front(), actual) ||
                    (actual.GetTransform() - target.GetTransform()).norm() >
                        1e-10)
                    return 1;
                const double nearest = scenario == 2 ? 2.0 * M_PI : joint;
                for (int i = 0; i < dof; ++i) {
                    if (std::abs(solutions.ik_joints.front()[i] - nearest) >
                        1e-12)
                        return 1;
                    checksum += (i + 1) * solutions.ik_joints.front()[i];
                }
                if (std::abs(distance - (initial - solutions.ik_joints.front())
                                            .cwiseAbs()
                                            .sum()) > 1e-12)
                    return 1;
            } else if (count != 0 || !solutions.ik_joints.empty()) {
                return 1;
            }
            // Unsupported baseline cases remain visible as failures in CSV.
            const int calls = wide && dof == 14 ? 32 : 16384;
            std::vector<double> samples;
            for (int repeat = -1; repeat < 7; ++repeat) {
                const auto begin = std::chrono::steady_clock::now();
                for (int call = 0; call < calls; ++call) {
                    if (solve() != success || solutions.ik_number != count ||
                        distance != reference_distance)
                        throw std::runtime_error("unstable nearest IK result");
                }
                if (repeat >= 0)
                    samples.push_back(
                        std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - begin)
                            .count() /
                        calls);
            }
            std::sort(samples.begin(), samples.end());
            const char *name = scenario == 0   ? "bounded_exact"
                               : scenario == 1 ? "multi_turn_exact"
                                               : "zero_seed_retry";
            std::cout << dof << ',' << name << ',' << samples[3] << ','
                      << success << ',' << count << ',' << distance << ','
                      << checksum << '\n';
        }
    }
}
