#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "holistic_motion/robot/Robot.h"

using namespace holistic_motion::robotics;

int main() {
    Robot empty;
    if (empty.LoadVisuals()) return 1;
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("holistic-motion-robot-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(directory)) return 1;
    const auto path = directory / "model.urdf";
    {
        std::ofstream output(path);
        output << "<robot name='chain'><link name='base'><visual><geometry>"
                  "<sphere radius='0.1'/></geometry></visual></link>";
        for (int joint = 0; joint < 96; ++joint) {
            const std::string parent =
                joint == 0 ? "base" : "link" + std::to_string(joint - 1);
            output << "<link name='link" << joint << "'/><joint name='joint" << joint
                   << "' type='prismatic'><parent link='" << parent
                   << "'/><child link='link" << joint
                   << "'/><axis xyz='2 0 0'/><limit lower='-1' upper='1' "
                      "velocity='1' effort='1'/></joint>";
        }
        output << "</robot>";
    }
    Robot robot(path.string());
    const auto original_kinematics = robot.GetKinematics();
    const auto original_joint = robot.GetJoint("joint20");
    if (!original_joint->axis.isApprox(Eigen::Vector3d::UnitX(), 1e-15)) return 1;
    SE3d pose;
    if (!original_kinematics->GetFK(Eigen::VectorXd::Constant(96, 0.5), pose) ||
        !pose.GetTranslation().isApprox(Eigen::Vector3d(48, 0, 0), 1e-14)) {
        std::cerr << "URDF axis magnitude changed prismatic joint units\n";
        return 1;
    }
    for (const auto &invalid_fields :
         {"<axis xyz='0 0 0'/><limit lower='-1' upper='1' velocity='1' effort='1'/>",
          "<axis xyz='1 0 0'/><limit lower='1' upper='-1' velocity='1' effort='1'/>"}) {
        {
            std::ofstream output(path);
            output << "<robot name='invalid'><link name='base'/><link name='tip'/>"
                      "<joint name='bad' type='prismatic'><parent link='base'/>"
                      "<child link='tip'/>"
                   << invalid_fields << "</joint></robot>";
        }
        try {
            robot.LoadURDF(path.string(), true);
            std::cerr << "invalid model reload was accepted\n";
            return 1;
        } catch (const std::invalid_argument &) {
        }
        if (robot.GetName() != "chain" || robot.GetDoF() != 96 ||
            robot.GetJoint("joint20") != original_joint ||
            robot.GetKinematics() != original_kinematics || robot.HasVisualsLoaded()) {
            std::cerr << "failed reload changed the previous model\n";
            return 1;
        }
    }
    for (const auto &invalid_source : {directory, directory / "missing.urdf"}) {
        if (robot.LoadURDF(invalid_source.string()) ||
            robot.GetKinematics() != original_kinematics ||
            robot.GetName() != "chain") {
            std::cerr << "failed source read changed the previous model\n";
            return 1;
        }
    }
    std::filesystem::remove(path);
    std::filesystem::remove(directory);
    Robot copy = robot;
    if (!copy.LoadVisuals() || !robot.LoadVisuals() ||
        !copy.HasVisualsLoaded() || !robot.HasVisualsLoaded() ||
        robot.GetLink("base")->visuals.size() != 1 ||
        robot.GetLink("base")->visuals.front().radius != 0.1) {
        std::cerr << "deferred visuals lost the original model snapshot\n";
        return 1;
    }
    auto chain = robot.CreateKinematics("base", "link95");
    if (!chain || chain->GetDOF() != 96) return 1;

    auto link = robot.GetLink("link20");
    auto joint = robot.GetJoint("joint20");
    std::ostringstream report;
    report << *joint << static_cast<const Joint &>(*joint);
    if (report.str().find("joint20") == std::string::npos) return 1;
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
    const auto renamed_mapping = robot.CreateJointPositionMap({joint->name});
    if (renamed_mapping.GetJoints().front().idx_q != 20) {
        std::cerr << "joint mapping retained stale model names\n";
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
    try {
        robot.CreateJointPositionMap({joint->name});
        std::cerr << "joint mapping accepted ambiguous model names\n";
        return 1;
    } catch (const std::invalid_argument &) {
        // Existing chain lookup keeps its first-match behavior. The new named
        // mapping rejects ambiguity consistently with the Python model adapter.
    }
    robot.GetJoint("joint95")->parent_link = "link95";
    if (robot.CreateKinematics("base", "link95")) {
        std::cerr << "cyclic parent links were accepted\n";
        return 1;
    }
    return 0;
}
