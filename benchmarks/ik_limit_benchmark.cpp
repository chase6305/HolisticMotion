#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "holistic_motion/kinematics/utility.h"

using namespace holistic_motion::robotics;

int main() {
    std::cout << "dof,wide,median_us,solutions,checksum\n"
              << std::setprecision(12);
    for (int dof : {1, 7, 14}) {
        for (bool wide : {false, true}) {
            const double bound = wide ? 2.0 * M_PI + 0.1 : 1.0;
            const std::vector<JointNode> nodes(
                dof, JointNode(SE3d(), Eigen::Vector3d::UnitZ(),
                               JointType::REVOLUTE, -bound, bound));
            const Eigen::VectorXd seed = Eigen::VectorXd::Constant(dof, 0.3);
            const int expected_count = wide ? (1 << dof) : 1;
            IkRtn result;
            const auto filter = [&]() {
                result.Clear();
                result.PushBack(seed);
                return result.GetLimitsIK(nodes) &&
                       result.ik_number == expected_count;
            };
            if (!filter()) return 1;
            double checksum = 0.0;
            for (const auto& solution : result.ik_joints) {
                checksum += solution.sum();
                for (int joint = 0; joint < dof; ++joint)
                    if (solution[joint] < -bound || solution[joint] > bound ||
                        std::abs(std::sin(solution[joint]) - std::sin(0.3)) >
                            1e-10)
                        return 1;
            }
            const int calls = !wide ? 8192 : dof == 14 ? 8 : 1024;
            std::vector<double> timings;
            for (int sample = -1; sample < 7; ++sample) {
                const auto start = std::chrono::steady_clock::now();
                for (int call = 0; call < calls; ++call)
                    if (!filter()) return 1;
                if (sample >= 0)
                    timings.push_back(
                        std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - start)
                            .count() /
                        calls);
            }
            std::sort(timings.begin(), timings.end());
            std::cout << dof << ',' << wide << ',' << timings[3] << ','
                      << expected_count << ',' << checksum << '\n';
        }
    }
}
