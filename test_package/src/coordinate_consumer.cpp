#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"
#include "holistic_motion/robot/JointPositionMap.h"
#include "holistic_motion/robot/Robot.h"

#include <chrono>
#include <filesystem>
#include <fstream>

int main() {
    using holistic_motion::robotics::JointPositionMap;
    const JointPositionMap mapping(
        4, 3, {{"slide", 3, 1, 2, 1}, {"turn", 1, 2, 1, 1}});
    Eigen::VectorXd reference(4), positions(2);
    reference << 0.6, 1.0, 0.0, 0.0;
    positions << 0.4, 0.3;
    const auto full =
        mapping.ConfigurationWithJointPositions(reference, positions);
    const auto recovered = mapping.JointPositions(full);
    if (full[0] != reference[0] || !recovered.isApprox(positions, 1e-12))
        return 1;

    using namespace holistic_motion::robotics::planning;
    const JointSpace space(Eigen::VectorXd::Constant(1, -1),
                           Eigen::VectorXd::Constant(1, 1), {"slide"});
    const SamplingPlanner planner(space);
    const PathOptimizer optimizer(planner.GetJointSpace());
    const auto planned =
        planner.Plan(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Constant(1, 0.3));
    if (!planned.Success() || !optimizer.Optimize(planned.path).Success() ||
        planner.GetJointSpace().GetJointNames() != std::vector<std::string>{"slide"} ||
        space.UnwrapPath(planned.path).size() != planned.path.size())
        return 1;

    // Exercise the exported Robot class layout and parsing/solver symbols,
    // rather than validating only a header-only map against the installed SDK.
    const auto path =
        std::filesystem::temp_directory_path() /
        ("holistic-motion-coordinate-consumer-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".urdf");
    {
        std::ofstream output(path);
        output
            << "<robot name='consumer'><link name='base'><visual><geometry>"
               "<sphere radius='0.1'/></geometry></visual></link>"
               "<link name='tip'/><joint name='slide' type='prismatic'>"
               "<parent link='base'/><child link='tip'/><axis xyz='1.5 0 0'/>"
               "<limit lower='-1' upper='1' velocity='1' effort='1'/>"
               "</joint></robot>";
    }
    holistic_motion::robotics::Robot robot(path.string());
    std::filesystem::remove(path);
    holistic_motion::robotics::SE3d pose;
    const auto report = robot.GetKinematics()->SolveIK(
        holistic_motion::robotics::SE3d(Eigen::Vector3d(0.3, 0, 0),
                                        holistic_motion::robotics::SO3d()),
        Eigen::VectorXd::Constant(1, 0.3));
    return robot.GetModelDoF() == 1 && robot.LoadVisuals() && report.Success() &&
                   robot.GetLink("base")->visuals.front().radius == 0.1 &&
                   robot.GetKinematics()->GetFK(Eigen::VectorXd::Constant(1, 0.3),
                                                pose) &&
                   pose.GetTranslation().isApprox(Eigen::Vector3d(0.3, 0, 0), 1e-14)
               ? 0
               : 1;
}
