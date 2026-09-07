#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "holistic_motion/kinematics/fep/FEPKinematics.h"

using namespace holistic_motion::robotics;

namespace {

Eigen::Matrix4d ReferenceFK(const std::vector<JointNode>& nodes,
                            const Eigen::VectorXd& joints, const SE3d& tcp) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    for (int joint = 0; joint < 7; ++joint) {
        Eigen::Matrix4d motion = Eigen::Matrix4d::Identity();
        motion.topLeftCorner<3, 3>() =
            Eigen::AngleAxisd(joints[joint], nodes[joint].axis)
                .toRotationMatrix();
        pose = (pose * nodes[joint].origin_pose.GetTransform() * motion).eval();
    }
    return pose * nodes.back().origin_pose.GetTransform() * tcp.GetTransform();
}

bool Check(FEPKinematics& solver, const Eigen::MatrixXd& joints) {
    std::vector<Eigen::Matrix4d> poses;
    if (!solver.ForwardBatch(joints, FEPBackend::CPU, poses) ||
        poses.size() != static_cast<std::size_t>(joints.rows()))
        return false;
    const auto nodes = solver.GetJointNode();
    for (Eigen::Index row = 0; row < joints.rows(); ++row) {
        const Eigen::VectorXd state = joints.row(row).transpose();
        const Eigen::Matrix4d expected =
            ReferenceFK(nodes, state, solver.GetTCP());
        SE3d scalar;
        if (!solver.GetFK(state, scalar) ||
            (scalar.GetTransform() - expected).norm() > 1e-12 ||
            (poses[row] - expected).norm() > 1e-12) {
            std::cerr
                << "oblique-axis FK differs from independent matrix chain\n";
            return false;
        }
        if (row != 0) continue;
        Eigen::MatrixXd jacobian;
        if (!solver.GetJacobian(state, jacobian)) return false;
        for (int joint = 0; joint < 7; ++joint) {
            Eigen::VectorXd positive = state, negative = state;
            constexpr double step = 1e-6;
            positive[joint] += step;
            negative[joint] -= step;
            const Eigen::Matrix4d derivative =
                (ReferenceFK(nodes, positive, solver.GetTCP()) -
                 ReferenceFK(nodes, negative, solver.GetTCP())) /
                (2 * step);
            const Eigen::Matrix3d angular =
                derivative.topLeftCorner<3, 3>() *
                expected.topLeftCorner<3, 3>().transpose();
            if ((jacobian.block<3, 1>(0, joint) -
                 derivative.topRightCorner<3, 1>())
                        .norm() > 1e-8 ||
                (jacobian.block<3, 1>(3, joint) -
                 Eigen::Vector3d(angular(2, 1), angular(0, 2), angular(1, 0)))
                        .norm() > 1e-8) {
                std::cerr << "continuous/oblique-axis Jacobian differs from "
                             "finite differences\n";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    std::vector<JointNode> nodes;
    for (int joint = 0; joint < 7; ++joint) {
        nodes.emplace_back(
            SE3d(Eigen::Vector3d(0.04, -0.02 * joint, 0.12),
                 SO3d(0.03 * joint, -0.1, 0.2)),
            Eigen::Vector3d(1.0, joint + 1.0, -0.5).normalized(),
            joint % 2 ? JointType::CONTINUOUS : JointType::REVOLUTE, -2.0, 2.0);
    }
    nodes.emplace_back(
        SE3d(Eigen::Vector3d(0.1, -0.03, 0.2), SO3d(0.1, 0.2, -0.3)),
        Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0);
    FEPKinematics solver(nodes);
    std::mt19937 generator(17);
    std::uniform_real_distribution<double> distribution(-1.5, 1.5);
    Eigen::MatrixXd joints(65, 7);
    for (Eigen::Index row = 0; row < joints.rows(); ++row)
        for (Eigen::Index col = 0; col < joints.cols(); ++col)
            joints(row, col) = distribution(generator);
    joints.row(1).setZero();
    joints.row(2).setConstant(1e-12);
    if (!Check(solver, joints)) return 1;
    solver.SetTCP(SE3d(Eigen::Vector3d(0.1, -0.2, 0.05), SO3d(0.2, -0.4, 0.3)));
    if (!Check(solver, joints)) return 1;
    nodes[2].axis = Eigen::Vector3d(-0.3, 0.7, 0.2).normalized();
    nodes[4].origin_pose =
        SE3d(Eigen::Vector3d(0.2, 0.1, -0.3), SO3d(0.3, 0.1, 0.2));
    nodes.back().origin_pose = SE3d(Eigen::Vector3d(-0.1, 0.2, 0.3), SO3d());
    if (!solver.SetJointNode(nodes) || !Check(solver, joints)) return 1;
    solver.ClearTCP();
    if (!Check(solver, joints)) return 1;
    std::vector<Eigen::Matrix4d> poses(1);
    if (!solver.ForwardBatch(Eigen::MatrixXd(0, 7), FEPBackend::CPU, poses) ||
        !poses.empty())
        return 1;
    joints(0, 0) = 3.0;
    poses.resize(1);
    if (solver.ForwardBatch(joints, FEPBackend::CPU, poses) || !poses.empty())
        return 1;
    // An eighth actuated node cannot index the seven-column input.
    nodes.back() = nodes.front();
    if (solver.SetJointNode(nodes) || !solver.IsCompatible()) return 1;
    joints(0, 0) = 0.0;
    if (!Check(solver, joints)) return 1;
    return 0;
}
