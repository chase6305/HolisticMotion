#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "holistic_motion/robot/Robot.h"

using namespace holistic_motion::robotics;

namespace {
class Fixture {
public:
    Fixture() {
        directory =
            std::filesystem::temp_directory_path() /
            ("holistic-motion-model-" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()));
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error(
                "cannot create benchmark fixture directory");
        path = directory / "robot.urdf";
    }
    ~Fixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
        std::filesystem::remove(directory, error);
    }
    std::filesystem::path directory;
    std::filesystem::path path;
};

void WriteModel(const std::filesystem::path& path, int depth, int leaves) {
    std::ofstream output(path);
    output << "<robot name='benchmark'><link name='base'/>";
    for (int joint = 0; joint < depth; ++joint) {
        const std::string parent =
            joint == 0 ? "base" : "link" + std::to_string(joint - 1);
        output
            << "<link name='link" << joint << "'/><joint name='joint" << joint
            << "' type='prismatic'><parent link='" << parent
            << "'/><child link='link" << joint
            << "'/><origin xyz='0 0 0.01'/><axis xyz='1 0 0'/>"
               "<limit lower='-1' upper='1' velocity='1' effort='1'/></joint>";
    }
    for (int leaf = 0; leaf < leaves; ++leaf)
        output << "<link name='leaf" << leaf << "'/><joint name='tip" << leaf
               << "' type='fixed'><parent link='link" << depth - 1
               << "'/><child link='leaf" << leaf
               << "'/><origin xyz='0 0 0.02'/></joint>";
    output << "</robot>";
    if (!output) throw std::runtime_error("cannot write benchmark fixture");
}

template <class Function>
double Measure(Function function, int calls) {
    std::vector<double> times;
    for (int sample = -1; sample < 7; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int call = 0; call < calls; ++call) function();
        if (sample >= 0)
            times.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - start)
                                .count() /
                            calls);
    }
    std::sort(times.begin(), times.end());
    return times[3];
}
}  // namespace

int main() {
    Fixture fixture;
    std::cout << "depth,leaves,load_us,chain_us,dof,tip_z\n"
              << std::setprecision(12);
    for (const auto& size : std::vector<std::pair<int, int>>{
             {7, 1}, {64, 1}, {256, 1}, {1024, 1}, {256, 1024}}) {
        const auto [depth, leaves] = size;
        WriteModel(fixture.path, depth, leaves);
        const double load_us = Measure(
            [&]() {
                Robot model(fixture.path.string());
                if (model.GetDoF() != depth)
                    throw std::runtime_error("wrong model DOF");
            },
            8);
        Robot model(fixture.path.string());
        const double chain_us = Measure(
            [&]() {
                const auto chain = model.CreateKinematics("base", "leaf0");
                if (!chain || chain->GetDOF() != depth)
                    throw std::runtime_error("wrong chain DOF");
            },
            128);
        const auto chain = model.CreateKinematics("base", "leaf0");
        SE3d pose;
        if (!chain->GetFK(Eigen::VectorXd::Zero(depth), pose) ||
            std::abs(pose.GetTranslation().z() - (depth * 0.01 + 0.02)) > 1e-10)
            return 1;
        std::cout << depth << ',' << leaves << ',' << load_us << ',' << chain_us
                  << ',' << chain->GetDOF() << ',' << pose.GetTranslation().z()
                  << '\n';
    }
}
