#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"

using namespace holistic_motion::robotics;

int main() {
    {
        std::vector<JointNode> nodes{JointNode(SE3d(), Eigen::Vector3d::UnitX(),
                                               JointType::PRISMATIC, -1.0, 1.0),
                                     JointNode()};
        nodes.back().joint_type = JointType::FIXED;
        NumericalKinematics solver(nodes);
        using Setter = bool (NumericalKinematics::*)(const double&);
        using Getter = bool (NumericalKinematics::*)(double&) const;
        const std::array<std::pair<Setter, Getter>, 4> settings{
            {{&NumericalKinematics::SetTransErrTh,
              &NumericalKinematics::GetTransErrTh},
             {&NumericalKinematics::SetAngleErrTh,
              &NumericalKinematics::GetAngleErrTh},
             {&NumericalKinematics::SetStepSize,
              &NumericalKinematics::GetStepSize},
             {&NumericalKinematics::SetDamp, &NumericalKinematics::GetDamp}}};
        for (const auto& setting : settings) {
            double original = 0.0;
            (solver.*setting.second)(original);
            for (double invalid :
                 {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                  std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()}) {
                double current = 0.0;
                if ((solver.*setting.first)(invalid) ||
                    !(solver.*setting.second)(current) || current != original) {
                    std::cerr
                        << "invalid IK setting changed solver configuration\n";
                    return 1;
                }
            }
        }
        SE3d target;
        solver.GetFK(Eigen::VectorXd::Constant(1, 0.3), target);
        Eigen::VectorXd seed = Eigen::VectorXd::Zero(1);
        IkRtn solutions;
        std::vector<double> distances;
        if (!solver.GetIK(target, solutions, seed, distances) ||
            distances.empty())
            return 1;
        seed[0] = std::numeric_limits<double>::quiet_NaN();
        if (solver.GetIK(target, solutions, seed, distances) ||
            solutions.ik_number != 0 || !distances.empty()) {
            std::cerr << "failed IK retained stale result distances\n";
            return 1;
        }
        seed.setZero();
        if (!solver.GetIK(target, solutions, seed, distances) ||
            distances.size() != 1)
            return 1;
    }
    // Exercise actual IK iterations on both sides of the six-dimensional task
    // space. An exact FK seed would bypass the SVD and hide dimension errors.
    for (int dof : {1, 3, 6, 7}) {
        std::vector<JointNode> nodes;
        for (int i = 0; i < dof; ++i) {
            nodes.emplace_back(SE3d(), Eigen::Vector3d::UnitX(),
                               JointType::PRISMATIC, -1.0, 1.0);
        }
        nodes.emplace_back();
        nodes.back().joint_type = JointType::FIXED;
        NumericalKinematics solver(nodes);
        const Eigen::VectorXd expected =
                Eigen::VectorXd::Constant(dof, 0.3 / dof);
        SE3d target;
        if (!solver.GetFK(expected, target)) return 1;
        Eigen::VectorXd seed = Eigen::VectorXd::Zero(dof);
        IkRtn solutions;
        std::vector<double> distances;
        if (!solver.GetIK(target, solutions, seed, distances)) {
            std::cerr << "numerical IK failed for DOF " << dof << '\n';
            return 1;
        }
        for (const auto& solution : solutions.ik_joints) {
            SE3d recovered;
            if (!solution.allFinite() || !solver.GetFK(solution, recovered) ||
                (target.GetTransform() - recovered.GetTransform()).norm() >
                        5e-4) {
                std::cerr << "numerical IK round trip failed\n";
                return 1;
            }
        }
    }

    for (const Eigen::Vector3d& offset : std::vector<Eigen::Vector3d>{
                 Eigen::Vector3d::Zero(), Eigen::Vector3d(100.0, -50.0, 3.0)}) {
        std::vector<JointNode> nodes{
                JointNode(SE3d(offset, SO3d(0.2, -0.3, 0.4)),
                          Eigen::Vector3d::UnitZ(), JointType::REVOLUTE, -1.0,
                          1.0),
                JointNode()};
        nodes.back().joint_type = JointType::FIXED;
        nodes.back().origin_pose = SE3d(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        NumericalKinematics solver(nodes);
        for (double angle : {0.001, 0.3}) {
            SE3d target;
            solver.GetFK(Eigen::VectorXd::Constant(1, angle), target);
            Eigen::VectorXd seed = Eigen::VectorXd::Zero(1);
            IkRtn solutions;
            std::vector<double> distances;
            if (!solver.GetIK(target, solutions, seed, distances)) {
                std::cerr << "IK with a translated base or TCP offset failed\n";
                return 1;
            }
            for (const auto& solution : solutions.ik_joints) {
                SE3d recovered;
                if (!solver.GetFK(solution, recovered) ||
                    (recovered.GetTranslation() - target.GetTranslation())
                                    .norm() >= 5e-4) {
                    std::cerr << "IK converged outside the TCP position "
                                 "tolerance\n";
                    return 1;
                }
            }
        }
    }

    const std::vector<JointNode> slide{
            JointNode(SE3d(), Eigen::Vector3d::UnitX(), JointType::PRISMATIC,
                      -10.0, 10.0)};
    for (bool wrap_near : {false, true}) {
        IkRtn solutions;
        solutions.PushBack(Eigen::VectorXd::Constant(1, 0.5));
        const Eigen::VectorXd reference = Eigen::VectorXd::Constant(1, 8.0);
        const bool valid =
                wrap_near ? solutions.WrapToLimitsNear(slide, reference)
                          : solutions.GetLimitsIK(slide);
        if (!valid || solutions.ik_number != 1 ||
            std::abs(solutions.ik_joints.front()[0] - 0.5) > 1e-12) {
            std::cerr << "prismatic joint was wrapped like an angle\n";
            return 1;
        }
        solutions.Clear();
        solutions.PushBack(Eigen::VectorXd::Constant(1, 12.0));
        if (wrap_near ? solutions.WrapToLimitsNear(slide, reference)
                      : solutions.GetLimitsIK(slide)) {
            std::cerr << "out-of-bounds prismatic joint was accepted\n";
            return 1;
        }
    }
    return 0;
}
