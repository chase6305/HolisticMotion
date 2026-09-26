#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"
#include "holistic_motion/planning/NullSpacePlanner.h"

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
    if (!solver.IsCompatible())
        return 1;
    Eigen::VectorXd nominal(7);
    nominal << 0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2;
    auto shared_solver = std::make_shared<SRSKinematics>(nodes);
    planning::NullSpacePlanner planner(shared_solver);
    const Eigen::VectorXd direction = Eigen::VectorXd::Unit(7, 2);
    std::vector<Eigen::VectorXd> normal_path, scaled_path;
    if (!planner.Plan(nominal, direction, 2, 0.01, normal_path) ||
        !planner.Plan(nominal, 1e300 * direction, 2, 0.01, scaled_path) ||
        normal_path.size() != scaled_path.size())
        return 6;
    for (std::size_t i = 0; i < normal_path.size(); ++i)
        if (!normal_path[i].isApprox(scaled_path[i], 1e-11))
            return 7;
    Eigen::VectorXd outside = nominal;
    outside[0] += 2.0 * std::acos(-1.0);
    if (planner.Plan(outside, direction, 2, 0.01, scaled_path) || !scaled_path.empty())
        return 8;
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(7, 3.0);
    const Eigen::VectorXd lower = -upper;
    upper[0] = nominal[0] - 0.01;
    if (!shared_solver->SetUserJointLimits(upper, lower) ||
        planner.Plan(nominal, direction, 2, 0.01, scaled_path) || !scaled_path.empty())
        return 9;
    for (const Eigen::VectorXd &q : std::vector<Eigen::VectorXd>{
             nominal, Eigen::VectorXd::Zero(7), 1e-10 * nominal}) {
        Eigen::MatrixXd jacobian;
        if (!solver.GetJacobian(q, jacobian))
            return 2;
        Eigen::MatrixXd projection(7, 7);
        for (Eigen::Index axis = 0; axis < 7; ++axis) {
            Eigen::VectorXd preferred = Eigen::VectorXd::Unit(7, axis), velocity;
            // In particular, exercise the 6-by-7 thin SVD with Eigen assertions
            // enabled in the sanitizer build of the library itself.
            if (!solver.GetNullSpaceVelocity(q, preferred, velocity))
                return 3;
            projection.col(axis) = velocity;
            if (!solver.GetNullSpaceVelocity(q, preferred, preferred) ||
                !preferred.isApprox(velocity, 1e-12)) {
                std::cerr << "in-place null-space projection changed the result\n";
                return 4;
            }
        }
        if (!projection.allFinite() ||
            (projection - projection.transpose()).norm() > 1e-12 ||
            (projection * projection - projection).norm() > 1e-12 ||
            (jacobian * projection).norm() > 1e-8 * jacobian.norm() ||
            projection.trace() < 1.0 - 1e-12 ||
            std::abs(projection.trace() - std::round(projection.trace())) > 1e-12) {
            std::cerr << "SRS result is not an orthogonal null-space projection\n";
            return 5;
        }
    }
    return 0;
}
