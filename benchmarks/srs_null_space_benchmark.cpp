#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"

using namespace holistic_motion::robotics;

int main() {
    const std::vector<Eigen::Vector3d> axes{
        Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitX()};
    std::vector<JointNode> nodes;
    for (const auto &axis : axes)
        nodes.emplace_back(SE3d(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0), axis,
                           JointType::REVOLUTE, -3.0, 3.0);
    nodes.emplace_back();
    nodes.back().joint_type = JointType::FIXED;
    SRSKinematics solver(nodes);
    Eigen::VectorXd nominal(7);
    nominal << 0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2;
    std::vector<Eigen::VectorXd> directions;
    for (int sample = 0; sample < 1024; ++sample) {
        Eigen::VectorXd direction(7);
        for (int joint = 0; joint < 7; ++joint)
            direction[joint] = std::sin(0.17 * (sample + 1) * (joint + 1));
        directions.push_back(direction);
    }
    std::cout << "joint_scale,median_us,checksum,max_task_residual\n"
              << std::setprecision(17);
    for (double scale : {1.0, 0.0, 1e-10}) {
        const Eigen::VectorXd q = scale * nominal;
        Eigen::MatrixXd jacobian;
        if (!solver.GetJacobian(q, jacobian))
            return 1;
        Eigen::VectorXd velocity;
        double maximum_residual = 0.0;
        for (const auto &preferred : directions) {
            if (!solver.GetNullSpaceVelocity(q, preferred, velocity))
                return 1;
            maximum_residual = std::max(maximum_residual, (jacobian * velocity).norm());
        }
        if (maximum_residual > 1e-7)
            return 1;
        std::vector<double> times;
        double checksum = 0.0;
        constexpr int queries = 20000;
        for (int repeat = 0; repeat < 8; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            double sum = 0.0;
            for (int query = 0; query < queries; ++query) {
                if (!solver.GetNullSpaceVelocity(
                        q, directions[query % directions.size()], velocity))
                    return 1;
                sum += velocity.sum();
            }
            const double elapsed = std::chrono::duration<double, std::micro>(
                                       std::chrono::steady_clock::now() - begin)
                                       .count() /
                                   queries;
            if (repeat > 0)
                times.push_back(elapsed);
            if (repeat > 0 && sum != checksum)
                return 1;
            checksum = sum;
        }
        std::sort(times.begin(), times.end());
        std::cout << scale << ',' << times[times.size() / 2] << ',' << checksum << ','
                  << maximum_residual << '\n';
    }
}
