#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/kinematics/fep/FEPKinematics.h"
#include "holistic_motion/kinematics/srs/SRSKinematics.h"

using namespace holistic_motion::robotics;

namespace {
std::vector<JointNode> MakeModel(int dof, bool mixed) {
    std::vector<JointNode> nodes;
    for (int i = 0; i < dof; ++i) {
        const Eigen::Vector3d axis =
            Eigen::Vector3d(1 + i % 3, 2 + i % 2, 3).normalized() * (i % 2 ? 1.5 : 0.5);
        nodes.emplace_back(
            SE3d(Eigen::Vector3d(0.02 * i, -0.01, 0.03), SO3d(0.01 * i, -0.02, 0.03)),
            axis, mixed && i % 3 == 2 ? JointType::PRISMATIC : JointType::REVOLUTE,
            -100.0, 100.0);
    }
    nodes.emplace_back(
        SE3d(Eigen::Vector3d(0.03, -0.01, 0.04), SO3d(0.02, 0.03, -0.01)),
        Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0);
    return nodes;
}

template <class Solver> bool CheckModel(int dof, bool mixed) {
    Solver solver(MakeModel(dof, mixed));
    const SE3d tool(Eigen::Vector3d(0.1, 0.2, -0.3), SO3d(0.2, -0.3, 0.4));
    for (const auto &tcp : {SE3d(), tool, SE3d()}) {
        if (!solver.SetTCP(tcp))
            return false;
        for (int sample = 0; sample < 32; ++sample) {
            Eigen::VectorXd joints(dof);
            for (int j = 0; j < dof; ++j)
                joints[j] = sample == 0 ? 0.0 : 0.3 * std::sin(0.2 * sample + 0.3 * j);
            Eigen::Matrix4d reference = Eigen::Matrix4d::Identity();
            for (int j = 0; j <= dof; ++j) {
                const auto &node = solver.GetJointNodesView()[j];
                reference = reference * node.origin_pose.GetTransform();
                Eigen::Matrix4d motion = Eigen::Matrix4d::Identity();
                if (j < dof) {
                    if (node.joint_type == JointType::PRISMATIC)
                        motion.block<3, 1>(0, 3) = node.axis * joints[j];
                    else
                        motion.block<3, 3>(0, 0) =
                            Eigen::AngleAxisd(node.axis.norm() * joints[j],
                                              node.axis.normalized())
                                .toRotationMatrix();
                }
                reference = reference * motion;
            }
            reference = reference * tcp.GetTransform();
            SE3d pose;
            std::vector<SE3d> trace;
            if (!solver.GetFK(joints, pose) || !solver.GetAllFK(joints, trace) ||
                trace.size() != static_cast<std::size_t>(dof + 2) ||
                !pose.GetTransform().isApprox(reference, 1e-12) ||
                !pose.GetTransform().isApprox(trace.back().GetTransform(), 1e-14))
                return false;
        }
    }
    // Invalid inputs leave the caller's terminal output intact and discard
    // incomplete frame traces, including when no joint coordinates exist.
    const SE3d sentinel(Eigen::Vector3d(4, 5, 6), SO3d(0.1, 0.2, 0.3));
    for (int kind = 0; kind < (dof == 0 ? 1 : 3); ++kind) {
        Eigen::VectorXd invalid = Eigen::VectorXd::Zero(kind == 0 ? dof + 1 : dof);
        if (kind == 1)
            invalid[0] = std::numeric_limits<double>::quiet_NaN();
        if (kind == 2)
            invalid[0] = std::numeric_limits<double>::infinity();
        SE3d pose = sentinel;
        std::vector<SE3d> trace{sentinel};
        if (solver.GetFK(invalid, pose) || !pose.IsApprox(sentinel) ||
            solver.GetAllFK(invalid, trace) || !trace.empty())
            return false;
    }
    return true;
}

class CustomSolver : public NumericalKinematics {
  public:
    using NumericalKinematics::NumericalKinematics;
    mutable int calls = 0;
    bool GetAllFK(const Eigen::VectorXd &, std::vector<SE3d> &poses) const override {
        ++calls;
        poses = {SE3d(1, 2, 3, 0, 0, 0, 1)};
        return true;
    }
};
} // namespace

int main() {
    for (int dof : {0, 1, 3, 7, 14, 64})
        if (!CheckModel<NumericalKinematics>(dof, true)) {
            std::cerr << "numerical FK mismatch for DOF " << dof << '\n';
            return 1;
        }
    if (!CheckModel<SRSKinematics>(7, false) || !CheckModel<FEPKinematics>(7, false))
        return 2;
    CustomSolver custom(MakeModel(7, false));
    SE3d pose;
    if (!custom.GetFK(Eigen::VectorXd::Zero(7), pose) || custom.calls != 1 ||
        !pose.GetTranslation().isApprox(Eigen::Vector3d(1, 2, 3)))
        return 3;
}
