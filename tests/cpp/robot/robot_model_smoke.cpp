#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "holistic_motion/robot/Robot.h"

using namespace holistic_motion::robotics;

int main() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("holistic-motion-robot-test-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(directory)) return 1;
    const auto path = directory / "model.urdf";
    {
        std::ofstream output(path);
        output << "<robot name='chain'><link name='base'/>";
        for (int joint = 0; joint < 96; ++joint) {
            const std::string parent =
                joint == 0 ? "base" : "link" + std::to_string(joint - 1);
            output << "<link name='link" << joint << "'/><joint name='joint"
                   << joint << "' type='prismatic'><parent link='" << parent
                   << "'/><child link='link" << joint
                   << "'/><axis xyz='1 0 0'/><limit lower='-1' upper='1' "
                      "velocity='1' effort='1'/></joint>";
        }
        output << "</robot>";
    }
    Robot robot(path.string());
    std::filesystem::remove(path);
    std::filesystem::remove(directory);
    auto chain = robot.CreateKinematics("base", "link95");
    if (!chain || chain->GetDOF() != 96) return 1;

    auto link = robot.GetLink("link20");
    auto joint = robot.GetJoint("joint20");
    link->name = "renamed_link";
    joint->name = "renamed_joint";
    link->parent_joint = joint->name;
    joint->child_link = link->name;
    robot.GetJoint("joint21")->parent_link = link->name;
    chain = robot.CreateKinematics("base", "link95");
    if (!chain || chain->GetDOF() != 96) {
        std::cerr << "chain lookup retained stale model names\n";
        return 1;
    }
    // Name resolution preserves the original first-match behavior even after
    // a public C++ edit creates a duplicate name in the model.
    robot.GetJoint("joint19")->name = joint->name;
    chain = robot.CreateKinematics("base", "link95");
    if (!chain || chain->GetDOF() != 95) {
        std::cerr << "duplicate name resolution changed\n";
        return 1;
    }
    robot.GetJoint("joint95")->parent_link = "link95";
    if (robot.CreateKinematics("base", "link95")) {
        std::cerr << "cyclic parent links were accepted\n";
        return 1;
    }
    return 0;
}
