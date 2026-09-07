#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/kinematics/OPWKinematics.h"
#include "holistic_motion/kinematics/URKinematics.h"
#include "holistic_motion/kinematics/srs/SRSKinematics.h"

using namespace holistic_motion::robotics;

namespace {
int Fail(int line) {
    std::cerr << "kinematic model regression failed at line " << line << "\n";
    return 1;
}

template <class Function>
bool Rejects(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// Exercise the runtime guards too: user-defined subclasses can bypass the
// supported mutation API through protected fields or a GetAllFK override.
class CorruptibleSolver : public NumericalKinematics {
public:
    using NumericalKinematics::NumericalKinematics;
    void ClearModel() { joint_nodes_.clear(); }
    void ActivateTerminal() { joint_nodes_.back() = joint_nodes_.front(); }
    void CorruptDOF() { dof_ = 100; }
    bool empty_output{false};
    bool GetAllFK(const Eigen::VectorXd& joints,
                  std::vector<SE3d>& poses) const override {
        if (!empty_output) return KinematicsBase::GetAllFK(joints, poses);
        poses.clear();
        return true;
    }
};
}  // namespace

int main() {
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<JointNode> nodes{JointNode(SE3d(), Eigen::Vector3d::UnitX(),
                                           JointType::PRISMATIC, -1.0, 1.0),
                                 JointNode()};
    nodes.back().joint_type = JointType::FIXED;
    NumericalKinematics solver(nodes);
    const Eigen::VectorXd q = Eigen::VectorXd::Constant(1, 0.3);
    SE3d original;
    if (!solver.GetFK(q, original)) return Fail(__LINE__);

    auto invalid = nodes;
    invalid.back() = nodes.front();
    if (solver.SetJointNode(invalid) ||
        !Rejects([&] { NumericalKinematics invalid_solver(invalid); }) ||
        !Rejects([] { NumericalKinematics empty({}); })) {
        std::cerr << "invalid numerical model layout was accepted\n";
        return Fail(__LINE__);
    }
    for (int kind = 0; kind < 7; ++kind) {
        invalid = nodes;
        if (kind == 0) invalid[0].axis.setZero();
        if (kind == 1) invalid[0].axis[0] = infinity;
        if (kind == 2) invalid[0].axis[0] = 1e200;
        if (kind == 3) invalid[0].joint_type = JointType::PLANAR;
        if (kind == 4) invalid[0].joint_type = JointType::FLOATING;
        if (kind == 5) invalid[0].origin_pose.Coeffs()[0] = nan;
        if (kind == 6) invalid[0].origin_pose.Coeffs().tail<4>().setZero();
        if (solver.SetJointNode(invalid) ||
            !Rejects([&] { NumericalKinematics invalid_solver(invalid); })) {
            std::cerr << "invalid joint model fields were accepted\n";
            return Fail(__LINE__);
        }
    }
    SE3d actual;
    if (!solver.GetFK(q, actual) || !actual.IsApprox(original))
        return Fail(__LINE__);

    const Eigen::VectorXd home = Eigen::VectorXd::Constant(1, 0.2);
    const Eigen::VectorXd weight = Eigen::VectorXd::Constant(1, 2.0);
    solver.SetHomeJoints(home);
    solver.SetIkNearstWeight(weight);
    solver.SetDOF(1);
    for (int invalid_dof : {-1, 0, 2, 100})
        if (!Rejects([&] { solver.SetDOF(invalid_dof); }))
            return Fail(__LINE__);
    Eigen::VectorXd actual_weight;
    if (solver.GetDOF() != 1 || !solver.GetIkNearstWeight(actual_weight) ||
        !actual_weight.isApprox(weight) ||
        !solver.GetHomeJoints().isApprox(home)) {
        std::cerr << "rejected DOF changed model-sized state\n";
        return Fail(__LINE__);
    }

    // Hardware and user limits must retain a nonempty intersection in either
    // update direction; a rejected update leaves both intervals usable.
    const Eigen::VectorXd user_lower = Eigen::VectorXd::Constant(1, 0.2);
    const Eigen::VectorXd user_upper = Eigen::VectorXd::Constant(1, 0.4);
    if (!solver.SetUserJointLimits(user_upper, user_lower))
        return Fail(__LINE__);
    if (solver.SetJointLimits(Eigen::VectorXd::Constant(1, -0.5),
                              Eigen::VectorXd::Constant(1, -1.0)))
        return Fail(__LINE__);
    Eigen::VectorXd upper, lower;
    solver.GetJointLimits(upper, lower);
    if (upper[0] != 1.0 || lower[0] != -1.0) return Fail(__LINE__);
    Eigen::VectorXd seed = q;
    IkRtn solutions;
    std::vector<double> distances;
    if (!solver.GetIK(original, solutions, seed, distances))
        return Fail(__LINE__);

    for (double invalid_value : {nan, infinity, -infinity}) {
        SE3d frame;
        frame.Coeffs()[0] = invalid_value;
        if (solver.SetTCP(frame) || solver.SetUserFrame(frame))
            return Fail(__LINE__);
    }
    SE3d bad_rotation;
    bad_rotation.Coeffs().tail<4>().setZero();
    if (solver.SetTCP(bad_rotation) || solver.SetUserFrame(bad_rotation) ||
        !solver.GetTCP().IsApprox(SE3d()) ||
        !solver.GetUserFrame().IsApprox(SE3d()))
        return Fail(__LINE__);

    // A fixed-only model can match its sole pose but cannot move toward another
    // target. It must never enter a zero-column SVD.
    const std::vector<JointNode> fixed{
        JointNode(SE3d(0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0),
                  Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0)};
    NumericalKinematics stationary(fixed);
    Eigen::VectorXd empty(0);
    Eigen::MatrixXd jacobian;
    SE3d fixed_pose;
    if (stationary.GetDOF() != 0 || !stationary.GetFK(empty, fixed_pose) ||
        !fixed_pose.IsApprox(fixed.front().origin_pose) ||
        !stationary.GetJacobian(empty, jacobian) || jacobian.rows() != 6 ||
        jacobian.cols() != 0 ||
        !stationary.GetIK(fixed_pose, solutions, empty, distances) ||
        solutions.ik_number != 1 || solutions.ik_joints.front().size() != 0)
        return Fail(__LINE__);
    if (stationary.GetIK(SE3d(), solutions, empty, distances) ||
        solutions.ik_number != 0 || !distances.empty())
        return Fail(__LINE__);
    if (!solutions.PushBack(empty) || solutions.PushBack(q) ||
        !solutions.PushBack(empty) || !solutions.RemoveRepeatedIK() ||
        !solutions.GetLimitsIK(fixed) || solutions.ik_number != 1 ||
        solutions.dof != 0)
        return Fail(__LINE__);
    SRSKinematics incompatible(std::vector<JointNode>(1));
    if (incompatible.IsCompatible()) return Fail(__LINE__);

    for (int count : {0, 1, 5}) {
        const std::vector<JointNode> short_nodes(count, nodes.front());
        if (!Rejects(
                [&] { OPWKinematics opw(OPWParameters{}, short_nodes); }) ||
            !Rejects([&] { URKinematics ur(URParameters{}, short_nodes); }))
            return Fail(__LINE__);
    }
    std::vector<JointNode> analytic_nodes(6, nodes.front());
    for (auto& node : analytic_nodes) node.joint_type = JointType::REVOLUTE;
    analytic_nodes.push_back(nodes.back());
    OPWKinematics opw(OPWParameters{}, analytic_nodes);
    URKinematics ur(URParameters{}, analytic_nodes);
    std::vector<SE3d> poses;
    if (!opw.GetAllFK(Eigen::VectorXd::Zero(6), poses) || poses.size() != 8 ||
        !ur.GetAllFK(Eigen::VectorXd::Zero(6), poses) || poses.size() != 8)
        return Fail(__LINE__);
    analytic_nodes.back() = analytic_nodes.front();
    if (opw.SetJointNode(analytic_nodes) || ur.SetJointNode(analytic_nodes))
        return Fail(__LINE__);

    for (int mode = 0; mode < 4; ++mode) {
        CorruptibleSolver broken(nodes);
        if (mode == 0) broken.ClearModel();
        if (mode == 1) broken.ActivateTerminal();
        if (mode == 2) broken.CorruptDOF();
        if (mode == 3) broken.empty_output = true;
        const Eigen::VectorXd input =
            mode == 2 ? Eigen::VectorXd::Zero(100).eval() : q;
        if (broken.GetFK(input, actual) || broken.GetJacobian(input, jacobian))
            return Fail(__LINE__);
        seed = input;
        if (broken.GetIK(original, solutions, seed, distances))
            return Fail(__LINE__);
    }
    // Finite joint values can still overflow FK arithmetic.
    auto huge_nodes = nodes;
    huge_nodes[0].axis[0] = 2.0;
    NumericalKinematics huge_slide(huge_nodes);
    poses.resize(1);
    if (huge_slide.GetAllFK(Eigen::VectorXd::Constant(1, 1e308), poses) ||
        !poses.empty())
        return Fail(__LINE__);
    huge_nodes[0].joint_type = JointType::REVOLUTE;
    NumericalKinematics huge_rotation(huge_nodes);
    if (huge_rotation.GetAllFK(Eigen::VectorXd::Constant(1, 1e200), poses) ||
        !poses.empty())
        return Fail(__LINE__);
    return 0;
}
