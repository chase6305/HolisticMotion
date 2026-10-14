#include <cmath>
#include <iostream>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/kinematics/OPWKinematics.h"
#include "holistic_motion/kinematics/URKinematics.h"

using namespace holistic_motion::robotics;

namespace {

bool CheckAliases(KinematicsBase &solver, const Eigen::VectorXd &joints) {
    SE3d reachable;
    if (!solver.GetFK(joints, reachable))
        return false;
    const SE3d unreachable(reachable.GetTranslation() +
                               Eigen::Vector3d(100, 0, 0),
                           reachable.GetRotation());
    for (bool nearest : {false, true}) {
        for (bool last : {false, true}) {
            for (bool failure : {false, true}) {
                IkRtn solutions;
                const Eigen::VectorXd other = joints.array() + 0.1;
                solutions.PushBack(last ? other : joints);
                solutions.PushBack(last ? joints : other);
                auto &seed = last ? solutions.ik_joints.back()
                                  : solutions.ik_joints.front();
                std::vector<double> distances;
                double distance = -1.0;
                const auto &target = failure ? unreachable : reachable;
                const bool success =
                    nearest
                        ? solver.GetNearestIK(target, solutions, seed, distance)
                        : solver.GetIK(target, solutions, seed, distances);
                // The output call invalidates its old element references.
                // Examine only the newly returned storage from this point.
                if (success == failure)
                    return false;
                if (failure) {
                    if (!solutions.ik_joints.empty() ||
                        solutions.ik_number != 0 || !distances.empty())
                        return false;
                    continue;
                }
                if (solutions.ik_number < 1 ||
                    !solutions.ik_joints.front().isApprox(joints, 1e-8))
                    return false;
                if (nearest) {
                    if (solutions.ik_number != 1 || std::abs(distance) > 1e-8)
                        return false;
                } else if (distances.size() != solutions.ik_joints.size() ||
                           std::abs(distances.front()) > 1e-8) {
                    return false;
                }
                for (const auto &solution : solutions.ik_joints) {
                    SE3d actual;
                    if (!solver.GetFK(solution, actual) ||
                        !actual.IsApprox(target))
                        return false;
                }
            }
        }
    }
    return true;
}

} // namespace

int main() {
    std::vector<JointNode> numerical_nodes{
        JointNode(SE3d(), Eigen::Vector3d::UnitX(), JointType::PRISMATIC, -2,
                  2),
        JointNode()};
    numerical_nodes.back().joint_type = JointType::FIXED;
    NumericalKinematics numerical(numerical_nodes);
    if (!CheckAliases(numerical, Eigen::VectorXd::Constant(1, 0.3)))
        return 1;

    std::vector<JointNode> fixed_nodes(1);
    fixed_nodes.front().joint_type = JointType::FIXED;
    NumericalKinematics fixed(fixed_nodes);
    if (!CheckAliases(fixed, Eigen::VectorXd(0)))
        return 1;

    std::vector<JointNode> turning_nodes{
        JointNode(SE3d(), Eigen::Vector3d::UnitZ(), JointType::REVOLUTE, 0, 7),
        JointNode()};
    turning_nodes.back().joint_type = JointType::FIXED;
    NumericalKinematics turning(turning_nodes);
    turning.SetMaxIterNum(1);
    IkRtn retry;
    retry.PushBack(Eigen::VectorXd::Constant(1, 2 * M_PI - 0.1));
    double distance = -1;
    if (!turning.GetNearestIK(SE3d(), retry, retry.ik_joints.front(),
                              distance) ||
        retry.ik_number != 1 ||
        std::abs(retry.ik_joints.front()[0] - 2 * M_PI) > 1e-12 ||
        std::abs(distance - 0.1) > 1e-12)
        return 1;

    std::vector<JointNode> nodes(6, JointNode(SE3d(), Eigen::Vector3d::UnitZ(),
                                              JointType::REVOLUTE, -2 * M_PI,
                                              2 * M_PI));
    OPWParameters opw;
    opw.a1 = 0.15;
    opw.a2 = -0.10;
    opw.b = 0.0;
    opw.c1 = 0.45;
    opw.c2 = 0.60;
    opw.c3 = 0.10;
    opw.c4 = 0.10;
    opw.rotation_directions.fill(1);
    OPWKinematics opw_solver(opw, nodes);
    opw_solver.DisableRobotConfig();
    URParameters ur;
    ur.d1 = 0.089159;
    ur.a2 = -0.425;
    ur.a3 = -0.39225;
    ur.d4 = 0.10915;
    ur.d5 = 0.09465;
    ur.d6 = 0.0823;
    ur.rotation_directions.fill(1);
    URKinematics ur_solver(ur, nodes);
    ur_solver.DisableRobotConfig();
    Eigen::VectorXd joints(6);
    joints << 0.25, -0.70, 0.85, -0.45, 0.65, -0.20;
    if (!CheckAliases(opw_solver, joints) || !CheckAliases(ur_solver, joints)) {
        std::cerr << "aliased analytic IK seed failed\n";
        return 1;
    }
    return 0;
}
