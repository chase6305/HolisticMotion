#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <vector>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"

using namespace holistic_motion::robotics;

namespace {

std::vector<JointNode> MakeModel() {
    const double pi = std::acos(-1.0), half = pi / 2.0;
    const std::array<Eigen::Vector3d, 7> translations{{{0, 0, 0},
                                                       {0, 0, 0.1025},
                                                       {0, 0.260, 0},
                                                       {0, 0, 0},
                                                       {0, 0.166, 0},
                                                       {0, 0, 0.098},
                                                       {0, 0, 0}}};
    const std::array<Eigen::Vector3d, 7> rotations{{{0, 0, 0},
                                                    {half, 0, pi},
                                                    {half, -half, pi},
                                                    {half, 0, 0},
                                                    {-half, pi, 0},
                                                    {half, 0, 0},
                                                    {-half, pi, half}}};
    std::vector<JointNode> nodes;
    for (std::size_t i = 0; i < translations.size(); ++i)
        nodes.emplace_back(SE3d(translations[i], SO3d(rotations[i])),
                           (i == 5 ? -1.0 : 1.0) * Eigen::Vector3d::UnitZ(),
                           JointType::REVOLUTE, -3.0, 3.0);
    nodes.emplace_back(SE3d(Eigen::Vector3d(-0.066, 0, 0), SO3d(0, -half, 0)),
                       Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0);
    return nodes;
}

bool RoundTrip(const SRSKinematics &solver, const Eigen::VectorXd &seed) {
    SE3d target, actual;
    Eigen::VectorXd solution;
    if (!solver.AnalyzeGeometry().closed_form_compatible ||
        !solver.GetFK(seed, target) ||
        !solver.GetAnalyticSolution(target, solver.GetConfiguration(seed), seed,
                                    solution) ||
        !solver.GetFK(solution, actual))
        return false;
    return actual.GetTransform().isApprox(target.GetTransform(), 1e-8);
}

bool CheckAxisDirectionsAndTurns() {
    const double period = 2.0 * std::acos(-1.0);
    Eigen::VectorXd physical(7);
    physical << 0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2;
    // Reverse every subset of axes, keeping the same physical joint motion.
    // A reversed first axis must not reverse the canonical link displacement.
    for (int mask = 0; mask < 128; ++mask) {
        auto nodes = MakeModel();
        Eigen::VectorXd scales(7);
        for (int joint = 0; joint < 7; ++joint) {
            scales[joint] =
                (mask & (1 << joint) ? -1.0 : 1.0) * (0.5 + 0.25 * (joint % 5));
            nodes[joint].axis *= scales[joint];
            nodes[joint].lower_limit = nodes[joint].lower_limit_set = -100.0;
            nodes[joint].upper_limit = nodes[joint].upper_limit_set = 100.0;
        }
        nodes[0].origin_pose =
            SE3d(Eigen::Vector3d(0.1, -0.03, 0.04), SO3d(0.02 * mask, -0.2, 0.3)) *
            nodes[0].origin_pose;
        const SRSKinematics solver(nodes);
        for (int sample = 0; sample < 4; ++sample) {
            Eigen::VectorXd joints = physical;
            for (int joint = 0; joint < 7; ++joint)
                joints[joint] =
                    (joints[joint] + 0.1 * std::sin(sample + joint)) / scales[joint];
            if (!RoundTrip(solver, joints))
                return false;
            const auto original = solver.GetConfiguration(joints);
            SE3d target;
            if (!solver.GetFK(joints, target))
                return false;
            for (int joint : {1, 3, 5}) {
                auto lifted = joints.eval();
                lifted[joint] +=
                    (sample % 2 ? -3.0 : 2.0) * period / std::abs(scales[joint]);
                const auto configuration = solver.GetConfiguration(lifted);
                if (configuration.shoulder != original.shoulder ||
                    configuration.elbow != original.elbow ||
                    configuration.wrist != original.wrist ||
                    std::abs(std::remainder(
                        configuration.redundancy - original.redundancy, period)) > 1e-8)
                    return false;
                Eigen::VectorXd solution;
                if (!solver.GetAnalyticSolution(target, configuration, lifted,
                                                solution) ||
                    !solution.isApprox(lifted, 1e-8))
                    return false;
            }
        }
    }
    return true;
}

bool CheckScaledCandidateSet() {
    const double pi = std::acos(-1.0);
    Eigen::VectorXd physical(7);
    physical << 0.2, -pi / 2, 0.3, pi / 2, 0.25, pi / 2, -0.2;
    for (int joint = 0; joint < 7; ++joint)
        if (joint != 1 && joint != 3 && joint != 5)
            physical[joint] += 0.1 * std::sin(6 + joint);
    std::vector<Eigen::VectorXd> reference;
    for (double scale : {1.0, 0.5, 1.5, 2.0, 3.0}) {
        auto nodes = MakeModel();
        for (int joint = 0; joint < 7; ++joint) {
            nodes[joint].axis *= scale;
            nodes[joint].lower_limit = nodes[joint].lower_limit_set = -pi / scale;
            nodes[joint].upper_limit = nodes[joint].upper_limit_set = pi / scale;
        }
        SRSKinematics solver(nodes);
        SE3d target;
        if (!solver.GetFK(physical / scale, target))
            return false;
        std::vector<Eigen::VectorXd> candidates;
        if (!solver.Solve(target, physical / scale, SRSSolveMethod::ALL_CONFIGURATIONS,
                          candidates))
            return false;
        for (auto &candidate : candidates)
            candidate *= scale;
        if (scale == 1.0) {
            reference = std::move(candidates);
            if (reference.size() < 2)
                return false;
            continue;
        }
        if (candidates.size() != reference.size())
            return false;
        for (const auto &expected : reference) {
            bool found = false;
            for (const auto &actual : candidates) {
                double squared = 0.0;
                for (int joint = 0; joint < 7; ++joint) {
                    const double phase =
                        std::atan2(std::sin(actual[joint] - expected[joint]),
                                   std::cos(actual[joint] - expected[joint]));
                    squared += phase * phase;
                }
                found = found || squared < 1e-8;
            }
            if (!found)
                return false;
        }
    }
    return true;
}

bool CheckScaledElbowPreparation() {
    Eigen::VectorXd physical(7);
    physical << 0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2;
    for (double scale : {1e-10, 1.0, 1e6, 1e10, 1e154}) {
        for (double offset : {0.4, -1.2, 2.8}) {
            auto nodes = MakeModel();
            for (int joint = 0; joint < 7; ++joint) {
                nodes[joint].axis *= scale;
                nodes[joint].lower_limit = nodes[joint].lower_limit_set = -10.0 / scale;
                nodes[joint].upper_limit = nodes[joint].upper_limit_set = 10.0 / scale;
            }
            nodes[3].origin_pose =
                nodes[3].origin_pose *
                SE3d(Eigen::Vector3d::Zero(),
                     SO3d(Eigen::AngleAxisd(offset, Eigen::Vector3d::UnitZ())));
            SRSKinematics solver(nodes);
            const auto geometry = solver.AnalyzeGeometry();
            if (!geometry.closed_form_compatible ||
                geometry.elbow_angle_direction != 1.0 ||
                !RoundTrip(solver, physical / scale))
                return false;
        }
    }
    return true;
}

} // namespace

int main() {
    auto nodes = MakeModel();
    SRSKinematics solver(nodes);
    Eigen::VectorXd seed(7);
    seed << 0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2;
    const SE3d tool(Eigen::Vector3d(0.04, -0.02, 0.03), SO3d(0.2, -0.1, 0.3));
    for (const SE3d &tcp : {SE3d(), tool, SE3d()})
        if (!solver.SetTCP(tcp) || !RoundTrip(solver, seed))
            return 1;

    // Rebuilding geometry while a tool is installed must not bake that tool
    // into the prepared model; later tool changes remain effective.
    solver.SetTCP(tool);
    if (!solver.SetJointNode(nodes) || !RoundTrip(solver, seed))
        return 2;
    solver.ClearTCP();
    if (!RoundTrip(solver, seed))
        return 3;

    SRSKinematics copy = solver;
    SE3d copy_target, changed_target;
    copy.GetFK(seed, copy_target);
    nodes[0].origin_pose = SE3d(Eigen::Vector3d(0.1, -0.2, 0.3), SO3d(0.1, 0.2, -0.1));
    if (!solver.SetJointNode(nodes) || !RoundTrip(solver, seed) ||
        !RoundTrip(copy, seed) || !solver.GetFK(seed, changed_target) ||
        changed_target.GetTransform().isApprox(copy_target.GetTransform(), 1e-6))
        return 4;

    // Losing and then restoring closed-form geometry cannot retain the old
    // prepared model or leave the restored instance on a numerical fallback.
    nodes[5].origin_pose =
        SE3d(Eigen::Vector3d(0.0, 0.04, 0.098), SO3d(std::acos(-1.0) / 2, 0, 0));
    if (!solver.SetJointNode(nodes) || solver.AnalyzeGeometry().closed_form_compatible)
        return 5;
    Eigen::VectorXd solution;
    if (solver.GetAnalyticSolution(copy_target, copy.GetConfiguration(seed), seed,
                                   solution))
        return 6;
    if (!solver.SetJointNode(MakeModel()) || !RoundTrip(solver, seed))
        return 7;

    std::vector<std::future<bool>> workers;
    for (int worker = 0; worker < 4; ++worker)
        workers.push_back(std::async(std::launch::async, [&, worker] {
            for (int request = 0; request < 32; ++request) {
                Eigen::VectorXd joints = seed;
                joints[0] += 0.01 * std::sin(worker + request);
                if (!RoundTrip(solver, joints))
                    return false;
            }
            return true;
        }));
    for (auto &worker : workers)
        if (!worker.get()) {
            std::cerr << "prepared SRS model changed concurrent query results\n";
            return 8;
        }

    // Direct JointNode construction retains axis scale. Analytic coordinates
    // must describe the same physical angles as scalar FK, without relying
    // on numerical refinement to repair an incorrect closed-form seed.
    for (double scale : {0.5, 1.5, 3.0}) {
        auto scaled_nodes = MakeModel();
        for (std::size_t i = 0; i < 7; ++i)
            scaled_nodes[i].axis *= scale;
        const SRSKinematics scaled(scaled_nodes);
        const Eigen::VectorXd scaled_seed = seed / scale;
        if (!RoundTrip(scaled, scaled_seed))
            return 9;
        Eigen::VectorXd elbow_seed;
        if (!scaled.GetAnalyticElbowSeed(copy_target,
                                         scaled.GetConfiguration(scaled_seed),
                                         scaled_seed, elbow_seed) ||
            std::abs(elbow_seed[3] - scaled_seed[3]) > 1e-8)
            return 14;
        SE3d actual;
        if (!scaled.GetFK(scaled_seed, actual) ||
            !actual.GetTransform().isApprox(copy_target.GetTransform(), 1e-12))
            return 10;
    }

    Eigen::VectorXd lower = Eigen::VectorXd::Constant(7, -3.0);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(7, 3.0);
    lower[0] = 1.0;
    upper[0] = 1.1;
    if (!solver.SetUserJointLimits(upper, lower) ||
        solver.GetAnalyticSolution(copy_target, copy.GetConfiguration(seed), seed,
                                   solution))
        return 11;
    lower[0] = seed[0] - 0.05;
    upper[0] = seed[0] + 0.05;
    if (!solver.SetUserJointLimits(upper, lower) || !RoundTrip(solver, seed))
        return 12;
    const auto report =
        solver.SolveDetailed(copy_target, seed, SRSSolveMethod::CONFIGURATION);
    if (report.status != SRSSolveStatus::SUCCESS || report.solutions.size() != 1 ||
        std::abs(report.minimum_joint_limit_margins[0] - 0.05) > 1e-8 ||
        std::abs(report.joint_limit_margins[0][0] - 0.05) > 1e-8)
        return 13;
    if (!CheckAxisDirectionsAndTurns()) {
        std::cerr << "SRS axis reversal or whole turns changed the physical branch\n";
        return 15;
    }
    if (!CheckScaledCandidateSet()) {
        std::cerr
            << "SRS candidate deduplication changed a rescaled model's branches\n";
        return 16;
    }
    if (!CheckScaledElbowPreparation()) {
        std::cerr << "SRS elbow preparation depends on coordinate scale\n";
        return 17;
    }
    return 0;
}
