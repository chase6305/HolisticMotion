#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <vector>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"

using namespace holistic_motion::robotics;

namespace {

std::vector<JointNode> MakeModel() {
    const std::vector<Eigen::Vector3d> axes{
        Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitX()};
    std::vector<JointNode> nodes;
    for (const auto &axis : axes)
        nodes.emplace_back(SE3d(), axis, JointType::REVOLUTE, -3.0, 3.0);
    nodes[3].origin_pose = SE3d(Eigen::Vector3d(0, 0, 0.4), SO3d());
    nodes[4].origin_pose = SE3d(Eigen::Vector3d(0, 0, 0.3), SO3d());
    nodes.emplace_back();
    nodes.back().joint_type = JointType::FIXED;
    return nodes;
}

bool SameGeometry(const SRSGeometryAnalysis &lhs, const SRSGeometryAnalysis &rhs) {
    return lhs.structurally_compatible == rhs.structurally_compatible &&
           lhs.closed_form_compatible == rhs.closed_form_compatible &&
           lhs.shoulder_center.isApprox(rhs.shoulder_center, 1e-13) &&
           lhs.wrist_center.isApprox(rhs.wrist_center, 1e-13) &&
           lhs.upper_arm_length == rhs.upper_arm_length &&
           lhs.forearm_length == rhs.forearm_length &&
           lhs.shoulder_axis_residual == rhs.shoulder_axis_residual &&
           lhs.wrist_axis_residual == rhs.wrist_axis_residual &&
           lhs.shoulder_orthogonality_residual == rhs.shoulder_orthogonality_residual &&
           lhs.wrist_orthogonality_residual == rhs.wrist_orthogonality_residual &&
           lhs.elbow_angle_offset == rhs.elbow_angle_offset &&
           lhs.elbow_angle_direction == rhs.elbow_angle_direction;
}

} // namespace

int main() {
    auto nodes = MakeModel();
    SRSKinematics solver(nodes);
    const auto original = solver.AnalyzeGeometry();
    if (!original.closed_form_compatible ||
        !solver.GetCapabilities().supports_closed_form_ik ||
        std::abs(original.upper_arm_length - 0.4) > 1e-12 ||
        std::abs(original.forearm_length - 0.3) > 1e-12)
        return 1;

    // The tolerance changes classification only. It must not trigger a
    // shared write or change the geometry seen by another request.
    const std::vector<double> tolerances{1e-8,
                                         1e-4,
                                         0.1,
                                         1.0,
                                         0.0,
                                         -1.0,
                                         std::numeric_limits<double>::infinity(),
                                         std::numeric_limits<double>::quiet_NaN()};
    std::vector<SRSGeometryAnalysis> expected;
    for (double tolerance : tolerances)
        expected.push_back(solver.AnalyzeGeometry(tolerance));
    if (!expected[0].closed_form_compatible || expected[3].closed_form_compatible)
        return 2;

    Eigen::VectorXd joints(7);
    joints << 0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2;
    SE3d target;
    if (!solver.GetFK(joints, target))
        return 3;
    const auto configuration = solver.GetConfiguration(joints);
    Eigen::VectorXd analytic;
    const bool analytic_available =
        solver.GetAnalyticSolution(target, configuration, joints, analytic);

    std::promise<void> start;
    const auto ready = start.get_future().share();
    std::vector<std::future<bool>> workers;
    for (int worker = 0; worker < 4; ++worker)
        workers.push_back(std::async(std::launch::async, [&, worker, ready] {
            ready.wait();
            for (std::size_t request = 0; request < 100; ++request) {
                const auto index = (request + worker) % tolerances.size();
                if (!SameGeometry(solver.AnalyzeGeometry(tolerances[index]),
                                  expected[index]))
                    return false;
                const auto actual = solver.GetConfiguration(joints);
                if (actual.shoulder != configuration.shoulder ||
                    actual.elbow != configuration.elbow ||
                    actual.wrist != configuration.wrist ||
                    actual.redundancy != configuration.redundancy)
                    return false;
                Eigen::VectorXd result;
                if (solver.GetAnalyticSolution(target, configuration, joints, result) !=
                        analytic_available ||
                    (analytic_available && !result.isApprox(analytic, 1e-12)))
                    return false;
            }
            return true;
        }));
    start.set_value();
    for (auto &worker : workers)
        if (!worker.get()) {
            std::cerr << "concurrent SRS queries changed geometry or solutions\n";
            return 4;
        }

    // A base translation moves both centers while preserving link lengths.
    const Eigen::Vector3d offset(0.1, -0.2, 0.3);
    nodes[0].origin_pose = SE3d(offset, SO3d());
    if (!solver.SetJointNode(nodes))
        return 5;
    const auto translated = solver.AnalyzeGeometry();
    if (!(translated.shoulder_center - original.shoulder_center)
             .isApprox(offset, 1e-12) ||
        !(translated.wrist_center - original.wrist_center).isApprox(offset, 1e-12) ||
        std::abs(translated.upper_arm_length - original.upper_arm_length) > 1e-12 ||
        std::abs(translated.forearm_length - original.forearm_length) > 1e-12)
        return 6;

    // New geometry must be ready before SetJointNode returns. A refused
    // update must leave that geometry intact.
    nodes[5].origin_pose = SE3d(Eigen::Vector3d(0, 0.03, 0), SO3d());
    if (!solver.SetJointNode(nodes) ||
        solver.AnalyzeGeometry().closed_form_compatible ||
        solver.GetCapabilities().supports_closed_form_ik ||
        !solver.AnalyzeGeometry(0.1).closed_form_compatible)
        return 7;
    const auto changed = solver.AnalyzeGeometry();
    nodes[0].axis.setZero();
    if (solver.SetJointNode(nodes) || !SameGeometry(solver.AnalyzeGeometry(), changed))
        return 8;

    // The SRS facade still accepts other numerical-chain models, but must
    // report their missing SRS geometry without touching shared state.
    nodes = MakeModel();
    nodes[0].joint_type = JointType::PRISMATIC;
    SRSKinematics incompatible(nodes);
    if (incompatible.AnalyzeGeometry().structurally_compatible ||
        incompatible.AnalyzeGeometry().closed_form_compatible)
        return 9;
    return 0;
}
