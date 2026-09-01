#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "holistic_motion/kinematics/OPWKinematics.h"
#include "holistic_motion/kinematics/URKinematics.h"

using holistic_motion::robotics::IkRtn;
using holistic_motion::robotics::JointNode;
using holistic_motion::robotics::JointType;
using holistic_motion::robotics::OPWKinematics;
using holistic_motion::robotics::OPWParameters;
using holistic_motion::robotics::SE3d;
using holistic_motion::robotics::URKinematics;
using holistic_motion::robotics::URParameters;

namespace {

std::vector<JointNode> MakeJoints() {
    std::vector<JointNode> joints;
    joints.reserve(6);
    for (int index = 0; index < 6; ++index) {
        joints.emplace_back(SE3d(), Eigen::Vector3d::UnitZ(),
                            JointType::REVOLUTE, -2.0 * M_PI, 2.0 * M_PI);
    }
    return joints;
}

template <typename Solver>
bool CheckRoundTrip(Solver& solver, const Eigen::VectorXd& joints) {
    SE3d target;
    if (!solver.GetFK(joints, target)) return false;
    Eigen::VectorXd seed = joints;
    IkRtn solutions;
    std::vector<double> distances;
    if (!solver.GetIK(target, solutions, seed, distances)) return false;
    if (solutions.ik_number < 1 || distances.size() != solutions.ik_joints.size()) {
        return false;
    }
    for (const auto& solution : solutions.ik_joints) {
        SE3d recovered;
        if (!solver.GetFK(solution, recovered) || !target.IsApprox(recovered)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const auto joints = MakeJoints();

    OPWParameters opw;
    opw.a1 = 0.15;
    opw.a2 = -0.10;
    opw.b = 0.0;
    opw.c1 = 0.45;
    opw.c2 = 0.60;
    opw.c3 = 0.10;
    opw.c4 = 0.10;
    opw.rotation_directions.fill(1);
    OPWKinematics opw_solver(opw, joints);
    opw_solver.DisableRobotConfig();

    URParameters ur;
    ur.d1 = 0.089159;
    ur.a2 = -0.425;
    ur.a3 = -0.39225;
    ur.d4 = 0.10915;
    ur.d5 = 0.09465;
    ur.d6 = 0.0823;
    ur.rotation_directions.fill(1);
    URKinematics ur_solver(ur, joints);
    ur_solver.DisableRobotConfig();

    Eigen::VectorXd q(6);
    q << 0.25, -0.70, 0.85, -0.45, 0.65, -0.20;
    if (!CheckRoundTrip(opw_solver, q) || !CheckRoundTrip(ur_solver, q)) {
        std::cerr << "analytic FK/IK round trip failed\n";
        return 1;
    }

    Eigen::VectorXd wrist_singular(6);
    wrist_singular << 0.20, -0.55, 0.75, 0.90, 0.0, -0.35;
    if (!CheckRoundTrip(opw_solver, wrist_singular) ||
        !CheckRoundTrip(ur_solver, wrist_singular)) {
        std::cerr << "analytic wrist-singularity round trip failed\n";
        return 1;
    }

    Eigen::Matrix4d unreachable_matrix = Eigen::Matrix4d::Identity();
    unreachable_matrix(0, 3) = 100.0;
    const SE3d unreachable(unreachable_matrix);
    for (auto* solver : std::vector<holistic_motion::robotics::KinematicsBase*>{
             &opw_solver, &ur_solver}) {
        Eigen::VectorXd seed = q;
        IkRtn solutions;
        std::vector<double> distances;
        if (solver->GetIK(unreachable, solutions, seed, distances) ||
            solutions.ik_number != 0 || !distances.empty()) {
            std::cerr << "unreachable target produced an IK solution\n";
            return 1;
        }
    }

    SE3d unused;
    Eigen::VectorXd malformed = Eigen::VectorXd::Zero(5);
    Eigen::VectorXd nonfinite = Eigen::VectorXd::Zero(6);
    nonfinite[2] = std::numeric_limits<double>::quiet_NaN();
    if (opw_solver.GetFK(malformed, unused) || ur_solver.GetFK(malformed, unused) ||
        opw_solver.GetFK(nonfinite, unused) || ur_solver.GetFK(nonfinite, unused)) {
        std::cerr << "invalid FK input was accepted\n";
        return 1;
    }

    Eigen::VectorXd original_upper, original_lower;
    opw_solver.GetJointLimits(original_upper, original_lower);
    Eigen::VectorXd invalid_upper = original_upper;
    invalid_upper[2] = std::numeric_limits<double>::quiet_NaN();
    if (opw_solver.SetJointLimits(invalid_upper, original_lower)) {
        std::cerr << "non-finite joint limits were accepted\n";
        return 1;
    }
    Eigen::VectorXd actual_upper, actual_lower;
    opw_solver.GetJointLimits(actual_upper, actual_lower);
    if (!(actual_upper.array() == original_upper.array()).all() ||
        !(actual_lower.array() == original_lower.array()).all()) {
        std::cerr << "rejected joint limits changed solver state\n";
        return 1;
    }
    Eigen::VectorXd invalid_weight = Eigen::VectorXd::Ones(6);
    invalid_weight[0] = -1.0;
    if (opw_solver.SetIkNearstWeight(invalid_weight)) {
        std::cerr << "negative IK weight was accepted\n";
        return 1;
    }
    return 0;
}
