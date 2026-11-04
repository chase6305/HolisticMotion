#include "holistic_motion/robot/RobotModel.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace holistic_motion::robotics;

int main() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("holistic-motion-model-data-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    const auto path = directory / "model.urdf";
    std::ofstream(path) << R"(<robot name="metadata">
      <link name="base"><visual><geometry><sphere radius="0.1"/></geometry></visual></link>
      <link name="arm"/><link name="tip"/><link name="free"/><link name="copy"/>
      <joint name="joint" type="revolute"><parent link="base"/><child link="arm"/>
        <axis xyz="0 0 2"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="terminal" type="fixed"><parent link="arm"/><child link="tip"/></joint>
      <joint name="floating" type="floating"><parent link="base"/><child link="free"/></joint>
      <joint name="mimic" type="revolute"><parent link="base"/><child link="copy"/>
        <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
        <mimic joint="joint" multiplier="1" offset="0.2"/></joint>
    </robot>)";
    RobotModel model(path.string());
    if (model.GetModelDoF() != 1 || model.GetLinks().size() != 5 ||
        model.GetChainJoints("base", "tip").size() != 2 ||
        model.GetChainJoints("base", "free").front()->joint_type !=
            JointType::FLOATING ||
        model.GetChainJoints("base", "copy").front()->mimic_joint != "joint" ||
        !model.GetJoint("joint")->axis.isApprox(Eigen::Vector3d::UnitZ(),
                                                1e-15))
        return 1;
    const auto mapping = model.CreateJointPositionMap({"joint"});
    if (mapping.GetJoints().size() != 1 ||
        model.GetJointCoordinates().size() != 1)
        return 1;
    const auto chain = model.GetChainJoints("base", "tip");
    auto copy = model;
    if (model.LoadURDF((directory / "missing.urdf").string()) ||
        model.GetName() != "metadata" || model.HasVisualsLoaded())
        return 1;
    std::filesystem::remove(path);
    if (!model.LoadVisuals() || !copy.LoadVisuals() ||
        model.GetLink("base")->visuals.size() != 1 ||
        copy.GetLink("base")->visuals.front().radius != 0.1)
        return 1;
    model.GetLink("arm")->parent_joint = "terminal";
    if (!model.GetChainJoints("base", "tip").empty())
        return 1;
    if (chain.front()->name != "joint")
        return 1;
    std::filesystem::remove(directory);
}
