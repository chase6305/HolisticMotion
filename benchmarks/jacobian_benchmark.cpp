#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "AllocationCounter.h"
#include "holistic_motion/kinematics/NumericalKinematics.h"

using namespace holistic_motion::robotics;

int main() {
    std::cout << "dof,median_us,allocations,allocation_kind,checksum\n"
              << std::setprecision(17);
    for (int dof : {0, 1, 3, 7, 14, 64, 256}) {
        std::vector<JointNode> nodes;
        for (int joint = 0; joint < dof; ++joint)
            nodes.emplace_back(
                SE3d(Eigen::Vector3d(0.02, -0.01, 0.1), SO3d(0.01, -0.02, 0.03)),
                (joint % 2 ? 0.5 : 1.5) * Eigen::Vector3d::Unit(joint % 3),
                joint % 4 == 0 ? JointType::PRISMATIC : JointType::REVOLUTE, -2., 2.);
        nodes.emplace_back(SE3d(Eigen::Vector3d(0.03, -0.02, 0.08), SO3d()),
                           Eigen::Vector3d::Zero(), JointType::FIXED, 0., 0.);
        NumericalKinematics solver(nodes);
        solver.SetTCP(SE3d(Eigen::Vector3d(0.04, 0.02, 0.03), SO3d(0.2, -0.1, 0.3)));
        const Eigen::VectorXd joints = Eigen::VectorXd::Constant(dof, 0.2);
        Eigen::MatrixXd jacobian;
        if (!solver.GetJacobian(joints, jacobian) || !jacobian.allFinite())
            return 1;
        const double expected = jacobian.sum();
        hm_benchmark::StartCounting();
        for (int query = 0; query < 128; ++query)
            if (!solver.GetJacobian(joints, jacobian))
                return 1;
        const double allocations = double(hm_benchmark::StopCounting()) / 128;
        constexpr int queries = 8192;
        std::vector<double> times;
        double checksum = 0;
        for (int repeat = 0; repeat < 8; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            double sum = 0;
            for (int query = 0; query < queries; ++query) {
                if (!solver.GetJacobian(joints, jacobian))
                    return 1;
                sum += jacobian.sum();
            }
            const double elapsed = std::chrono::duration<double, std::micro>(
                                       std::chrono::steady_clock::now() - begin)
                                       .count() /
                                   queries;
            if (std::abs(sum - queries * expected) >
                1e-8 * std::max(1.0, std::abs(sum)))
                return 1;
            if (repeat > 0)
                times.push_back(elapsed);
            checksum = sum;
        }
        std::sort(times.begin(), times.end());
        std::cout << dof << ',' << times[times.size() / 2] << ',' << allocations << ','
                  << hm_benchmark::allocation_kind << ',' << checksum << '\n';
    }
}
