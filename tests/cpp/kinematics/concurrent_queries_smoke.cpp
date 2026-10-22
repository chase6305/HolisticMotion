#include <cmath>
#include <future>
#include <iostream>
#include <vector>

#include "holistic_motion/kinematics/fep/FEPKinematics.h"
#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"
#include "holistic_motion/robot/JointPositionMap.h"

using namespace holistic_motion::robotics;
using namespace holistic_motion::robotics::planning;

namespace {

bool SamePath(const std::vector<Eigen::VectorXd> &lhs,
              const std::vector<Eigen::VectorXd> &rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t i = 0; i < lhs.size(); ++i)
        if (!lhs[i].isApprox(rhs[i], 1e-14)) return false;
    return true;
}

Eigen::Matrix4d ReferenceFK(const std::vector<JointNode> &nodes,
                            const Eigen::VectorXd &q) {
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 7; ++i) {
        Eigen::Matrix4d rotation = Eigen::Matrix4d::Identity();
        rotation.topLeftCorner<3, 3>() =
            Eigen::AngleAxisd(q[i], nodes[i].axis).toRotationMatrix();
        result = (result * nodes[i].origin_pose.GetTransform() * rotation).eval();
    }
    return result * nodes.back().origin_pose.GetTransform();
}

}  // namespace

int main() {
    std::vector<JointNode> nodes;
    for (int joint = 0; joint < 7; ++joint)
        nodes.emplace_back(
            SE3d(Eigen::Vector3d(0.04, 0.02 * joint, 0.1), SO3d(0.1, -0.2, 0.3)),
            Eigen::Vector3d::Unit(joint % 3), JointType::REVOLUTE, -2.0, 2.0);
    nodes.emplace_back(SE3d(Eigen::Vector3d(0.1, -0.2, 0.05), SO3d()),
                       Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0);
    FEPKinematics solver(nodes);
    solver.SetTransErrTh(1e-7);
    solver.SetAngleErrTh(1e-7);
    solver.SetMaxIterNum(200);
    // Same scalar chain embedded after a preserved free-flyer block. Two of
    // its coordinates use continuous pairs, so neither q nor v indices match
    // the chain indices. The mapping and reference are shared read-only.
    const JointPositionMap mapping(16, 13,
                                   {{"j0", 7, 1, 6, 1},
                                    {"j1", 8, 2, 7, 1},
                                    {"j2", 10, 1, 8, 1},
                                    {"j3", 11, 1, 9, 1},
                                    {"j4", 12, 2, 10, 1},
                                    {"j5", 14, 1, 11, 1},
                                    {"j6", 15, 1, 12, 1}});
    Eigen::VectorXd reference = Eigen::VectorXd::Zero(16);
    reference[0] = 0.3;
    reference[6] = 1.0;
    const Eigen::VectorXd original_reference = reference;

    const SamplingPlanner planner(
        Eigen::Vector2d(-1, -1), Eigen::Vector2d(1, 1), [](const Eigen::VectorXd &q) {
            return !(std::abs(q[0]) < 0.2 && std::abs(q[1]) < 0.7);
        });
    const PathOptimizer optimizer(Eigen::Vector2d(-1, -1), Eigen::Vector2d(1, 1));
    PlanningOptions options;
    options.random_seed = 71;
    options.max_iterations = 2000;
    options.timeout_seconds = 5.0;
    options.extension_range = 0.15;
    options.edge_resolution = 0.02;
    const Eigen::Vector2d start(-0.8, 0.0), goal(0.8, 0.0);
    const auto expected_plan = planner.Plan(start, goal, options);
    if (!expected_plan.Success() || expected_plan.statistics.iterations == 0) return 1;
    const std::vector<Eigen::VectorXd> rough{start, Eigen::Vector2d(-0.4, 0.5),
                                             Eigen::Vector2d(0, -0.5),
                                             Eigen::Vector2d(0.4, 0.5), goal};
    const auto expected_optimization = optimizer.Optimize(rough);
    if (!expected_optimization.Success()) return 1;
    const bool cuda_available = FEPKinematics::HasCudaBackend();

    std::promise<void> start_workers;
    const auto ready = start_workers.get_future().share();
    std::vector<std::future<bool>> workers;
    for (int worker = 0; worker < 4; ++worker) {
        workers.push_back(std::async(std::launch::async, [&, worker, ready] {
            ready.wait();
            for (int request = 0; request < 32; ++request) {
                Eigen::VectorXd q(7);
                for (int joint = 0; joint < 7; ++joint)
                    q[joint] = 0.7 * std::sin(0.1 * request + worker + 0.3 * joint);
                const auto full = mapping.ConfigurationWithJointPositions(reference, q);
                if (!full.head(7).isApprox(reference.head(7), 0.0) ||
                    !mapping.JointPositions(full).isApprox(q, 1e-14))
                    return false;
                const auto gradient =
                    Eigen::VectorXd::LinSpaced(13, worker, worker + 12);
                if (!mapping.JointVelocities(gradient).isApprox(gradient.tail(7), 0.0))
                    return false;

                const auto expected_pose = ReferenceFK(nodes, q);
                SE3d pose;
                Eigen::MatrixXd jacobian;
                if (!solver.GetFK(q, pose) ||
                    !pose.GetTransform().isApprox(expected_pose, 1e-13) ||
                    !solver.GetJacobian(q, jacobian) || jacobian.cols() != 7 ||
                    !jacobian.allFinite())
                    return false;
                Eigen::VectorXd seed = q;
                seed[0] += 0.002;
                IkRtn solutions;
                std::vector<double> distances;
                if (!solver.GetIK(pose, solutions, seed, distances) ||
                    solutions.ik_joints.empty() ||
                    !solver.GetFK(solutions.ik_joints.front(), pose) ||
                    !pose.GetTransform().isApprox(expected_pose, 1e-6))
                    return false;

                if (request % 8 == 0) {
                    Eigen::MatrixXd states(129, 7);
                    states.rowwise() = q.transpose();
                    std::vector<Eigen::Matrix4d> poses;
                    if (!solver.ForwardBatch(
                            states, cuda_available ? FEPBackend::CUDA : FEPBackend::CPU,
                            poses) ||
                        poses.size() != 129)
                        return false;
                    for (const auto &batch_pose : poses)
                        if (!batch_pose.isApprox(expected_pose, 1e-12)) return false;
                    const auto plan = planner.Plan(start, goal, options);
                    const auto optimized = optimizer.Optimize(rough);
                    if (!plan.Success() || !SamePath(plan.path, expected_plan.path) ||
                        !optimized.Success() ||
                        !SamePath(optimized.path, expected_optimization.path))
                        return false;
                }
            }
            return true;
        }));
    }
    start_workers.set_value();
    bool success = true;
    for (auto &worker : workers) {
        try {
            success = worker.get() && success;
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            success = false;
        }
    }
    if (!success || !reference.isApprox(original_reference, 0.0)) {
        std::cerr << "concurrent read-only queries changed independent results\n";
        return 1;
    }
    return 0;
}
