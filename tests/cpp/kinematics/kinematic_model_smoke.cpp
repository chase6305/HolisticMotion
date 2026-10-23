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

    // Representable tiny translation errors must not disappear when their
    // squares underflow before comparison with a caller-selected tolerance.
    NumericalKinematics tiny_slide(nodes);
    for (double target_translation : {1e-160, 1e-180}) {
        const double tolerance = target_translation * 1e-10;
        tiny_slide.SetTransErrTh(tolerance);
        seed = Eigen::VectorXd::Zero(1);
        const SE3d target(Eigen::Vector3d(target_translation, 0.0, 0.0),
                          SO3d());
        if (!tiny_slide.GetIK(target, solutions, seed, distances) ||
            !tiny_slide.GetFK(solutions.ik_joints.front(), actual) ||
            std::abs(actual.GetTranslation().x() / target_translation - 1.0) >=
                1e-10)
            return Fail(__LINE__);
    }

    auto rotation_nodes = nodes;
    rotation_nodes[0].joint_type = JointType::REVOLUTE;
    rotation_nodes[0].axis = Eigen::Vector3d::UnitZ();
    NumericalKinematics tiny_rotation(rotation_nodes);
    for (double angle : {1e-8, 1e-12, 1e-180}) {
        const Eigen::Matrix3d rotation =
            Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ())
                .toRotationMatrix();
        if (std::abs(GetRAngle(rotation) / angle - 1.0) > 1e-14)
            return Fail(__LINE__);
        tiny_rotation.SetAngleErrTh(angle * 1e-10);
        seed = Eigen::VectorXd::Zero(1);
        if (!tiny_rotation.GetIK(SE3d(Eigen::Vector3d::Zero(), SO3d(rotation)),
                                 solutions, seed, distances) ||
            std::abs(solutions.ik_joints.front()[0] / angle - 1.0) >= 1e-10)
            return Fail(__LINE__);
    }

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

    // A zero-seed retry must still choose the multi-turn representation
    // nearest to the caller's seed, and report that movement's distance.
    std::vector<JointNode> turning{JointNode(SE3d(), Eigen::Vector3d::UnitZ(),
                                             JointType::REVOLUTE, -7.0, 7.0),
                                   JointNode()};
    turning.back().joint_type = JointType::FIXED;
    NumericalKinematics nearest(turning);
    nearest.SetMaxIterNum(1);
    nearest.SetIkNearstWeight(Eigen::VectorXd::Constant(1, 2.0));
    seed = Eigen::VectorXd::Constant(1, 2.0 * M_PI - 0.1);
    double nearest_distance = -1.0;
    if (!nearest.GetNearestIK(SE3d(), solutions, seed, nearest_distance) ||
        solutions.ik_number != 1 ||
        std::abs(solutions.ik_joints.front()[0] - 2.0 * M_PI) > 1e-12 ||
        std::abs(nearest_distance - 0.2) > 1e-12 || seed[0] != 2.0 * M_PI - 0.1)
        return Fail(__LINE__);
    seed[0] = 0.2;
    SE3d nearest_target;
    if (!nearest.GetFK(seed, nearest_target) ||
        !nearest.GetNearestIK(nearest_target, solutions, seed,
                              nearest_distance) ||
        solutions.ik_number != 1 || nearest_distance != 0.0 ||
        solutions.ik_joints.front()[0] != seed[0])
        return Fail(__LINE__);
    for (double invalid_seed : {nan, infinity, -infinity}) {
        seed[0] = invalid_seed;
        if (nearest.GetNearestIK(SE3d(), solutions, seed, nearest_distance) ||
            solutions.ik_number != 0)
            return Fail(__LINE__);
    }

    // Nearest-only queries must not enumerate a Cartesian product of
    // equivalent turns or inherit the all-solutions expansion budget.
    for (int dof : {17, 32, 65}) {
        std::vector<JointNode> periodic_nodes(
            dof, JointNode(SE3d(), Eigen::Vector3d::UnitZ(),
                           JointType::REVOLUTE, 0.0, 2.0 * M_PI));
        periodic_nodes.emplace_back();
        periodic_nodes.back().joint_type = JointType::FIXED;
        NumericalKinematics periodic(periodic_nodes);
        seed = Eigen::VectorXd::Zero(dof);
        if (periodic.GetIK(SE3d(), solutions, seed, distances) ||
            solutions.ik_number != 0 || !distances.empty())
            return Fail(__LINE__);
        if (!periodic.GetNearestIK(SE3d(), solutions, seed, nearest_distance) ||
            solutions.ik_number != 1 || nearest_distance != 0.0 ||
            !solutions.ik_joints.front().isZero(0.0))
            return Fail(__LINE__);
    }
    for (double axis_scale : {0.5, -0.5, 1.5, -1.5, 2.0}) {
        auto scaled_nodes = turning;
        scaled_nodes[0].axis = axis_scale * Eigen::Vector3d::UnitZ();
        NumericalKinematics scaled(scaled_nodes);
        seed = Eigen::VectorXd::Constant(1, 2.0 * M_PI / std::abs(axis_scale) *
                                                2.0);
        if (!scaled.GetIK(SE3d(), solutions, seed, distances))
            return Fail(__LINE__);
        for (const auto &joints : solutions.ik_joints) {
            if (!scaled.GetFK(joints, actual) ||
                (actual.GetTransform() - Eigen::Matrix4d::Identity()).norm() >
                    1e-10)
                return Fail(__LINE__);
        }
        if (!scaled.GetNearestIK(SE3d(), solutions, seed, nearest_distance) ||
            !scaled.GetFK(solutions.ik_joints.front(), actual) ||
            (actual.GetTransform() - Eigen::Matrix4d::Identity()).norm() >
                1e-10)
            return Fail(__LINE__);
    }
    solutions.Clear();
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
    if (!huge_rotation.GetAllFK(Eigen::VectorXd::Constant(1, 1e200), poses) ||
        !poses.back().GetRotation().isApprox(
            Eigen::AngleAxisd(2e200, Eigen::Vector3d::UnitX()).toRotationMatrix(),
            1e-12))
        return Fail(__LINE__);
    // The angle itself, rather than its squared norm, must overflow here.
    if (huge_rotation.GetAllFK(Eigen::VectorXd::Constant(1, 1e308), poses) ||
        !poses.empty())
        return Fail(__LINE__);

    // Configuration filtering must use the same oblique-axis motion as FK.
    std::vector<JointNode> oblique_nodes(
        6, JointNode(SE3d(0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0),
                     Eigen::Vector3d::Ones().normalized(),
                     JointType::REVOLUTE, -3.0, 3.0));
    oblique_nodes.emplace_back();
    oblique_nodes.back().joint_type = JointType::FIXED;
    Eigen::VectorXd oblique_q(6);
    oblique_q << 1.711518684233507, 1.7151561451572688, 2.8148161051371146,
        1.4883586374573916, 0.9333048557466572, 2.6333072650031477;
    NumericalKinematics oblique(oblique_nodes);
    if (!oblique.GetAllFK(oblique_q, poses) ||
        poses[2].GetTranslation().z() <= poses[1].GetTranslation().z())
        return Fail(__LINE__);
    RobotConfigManager elbow_up;
    elbow_up.SetElbowUp(true);
    IkRtn oblique_solutions;
    if (!oblique_solutions.PushBack(oblique_q) ||
        !oblique_solutions.LimitRobotConfig(oblique_nodes, elbow_up) ||
        oblique_solutions.ik_number != 1)
        return Fail(__LINE__);
    for (int count : {0, 1, 3}) {
        oblique_solutions.Clear();
        oblique_solutions.PushBack(oblique_q);
        if (oblique_solutions.LimitRobotConfig(
                std::vector<JointNode>(oblique_nodes.begin(),
                                       oblique_nodes.begin() + count), elbow_up) ||
            !oblique_solutions.ik_joints.empty())
            return Fail(__LINE__);
    }
    return 0;
}
