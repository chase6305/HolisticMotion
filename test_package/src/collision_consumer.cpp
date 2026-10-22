#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "holistic_motion/collision/CollisionModel.h"

int main() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("holistic_motion_consumer_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(directory))
        return 1;
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } cleanup{directory};
    const auto urdf = directory / "consumer.urdf";
    {
        std::ofstream output(urdf);
        output << R"(<robot name="consumer">
          <link name="base"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
          <link name="arm"/>
          <link name="tool"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
          <joint name="turn" type="continuous"><parent link="base"/><child link="arm"/>
            <axis xyz="0 0 1.5"/><limit velocity="1" effort="1"/></joint>
          <joint name="slide" type="prismatic"><parent link="arm"/><child link="tool"/>
            <origin xyz="1 0 0"/><axis xyz="2 0 0"/>
            <limit lower="-0.5" upper="0.5" velocity="1" effort="1"/></joint>
        </robot>)";
        if (!output)
            return 1;
    }
    try {
        holistic_motion::robotics::collision::CollisionModel model(urdf.string());
        if (model.GetConfigurationSize() != 3 || model.GetVelocitySize() != 2)
            return 1;
        const auto mapping = model.CreateJointPositionMap({"slide", "turn"});
        Eigen::VectorXd positions(2);
        positions << 0.4, 0.3;
        const auto full = mapping.ConfigurationWithJointPositions(
            model.NeutralConfiguration(), positions);
        if (!mapping.JointPositions(full).isApprox(positions, 1e-12))
            return 1;
        std::filesystem::remove(urdf);
        auto worker = model.Clone();
        model.ClearCollisionPairs();
        if (model.GetCollisionPairCount() != 0 || worker.GetCollisionPairCount() != 1 ||
            !worker.CreateJointPositionMap({"slide", "turn"})
                 .JointPositions(full)
                 .isApprox(positions, 1e-12))
            return 1;
        return !worker.InCollision(full) &&
                       std::abs(worker.MinimumDistance(full).distance - 1.2) < 1e-12
                   ? 0
                   : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
