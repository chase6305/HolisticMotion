#include <vector>

#include "holistic_motion/kinematics/fep/FEPKinematics.h"

int main() {
    using namespace holistic_motion::robotics;
    std::vector<JointNode> nodes;
    for (int index = 0; index < 7; ++index)
        nodes.emplace_back(SE3d(), Eigen::Vector3d::UnitZ(),
                           JointType::REVOLUTE, -1.0, 1.0);
    nodes.emplace_back();
    nodes.back().joint_type = JointType::FIXED;
    FEPKinematics solver(nodes);
    const KinematicsBase &base = solver;
    const auto capabilities = base.GetCapabilities();
    if (capabilities.family != KinematicsFamily::FEP ||
        capabilities.default_ik_algorithm != IKAlgorithm::NUMERICAL ||
        !capabilities.supports_batch_fk || capabilities.supports_closed_form_ik ||
        capabilities.cuda_compiled != FEPKinematics::IsCudaCompiled())
        return 1;
    const Eigen::MatrixXd joints = Eigen::MatrixXd::Zero(2, 7);
    std::vector<Eigen::Matrix4d> poses;
    if (!solver.ForwardBatch(joints, FEPBackend::CPU, poses) || poses.size() != 2)
        return 1;
    return poses[0].isIdentity(1e-12) && poses[1].isIdentity(1e-12) ? 0 : 1;
}
