#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "holistic_motion/collision/CollisionModel.h"
#include "holistic_motion/collision/SphereCollisionModel.h"

using holistic_motion::robotics::collision::CollisionModel;
using holistic_motion::robotics::collision::CollisionSphere;
using holistic_motion::robotics::collision::SphereCollisionModel;

int main() {
  const auto path = std::filesystem::temp_directory_path() /
                    "holistic_motion_collision_smoke.urdf";
  {
    std::ofstream urdf(path);
    urdf << R"(<robot name="collision_smoke">
  <link name="base">
    <collision name="base_sphere"><geometry><sphere radius="0.2"/></geometry></collision>
  </link>
  <link name="slider">
    <collision name="slider_sphere"><geometry><sphere radius="0.2"/></geometry></collision>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="slider"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>)";
  }

  try {
    SphereCollisionModel sphere_model(
        path.string(),
        {CollisionSphere{"base", "base", Eigen::Vector3d::Zero(), 0.2},
         CollisionSphere{"slider", "slider", Eigen::Vector3d::Zero(), 0.2}});
    if (!sphere_model.InCollision(Eigen::VectorXd::Zero(1)) ||
        sphere_model.InCollision(Eigen::VectorXd::Ones(1)) ||
        std::abs(sphere_model.MinimumDistance(Eigen::VectorXd::Ones(1)).distance -
                 0.6) > 1e-12) {
      std::cerr << "sphere collision backend failed\n";
      return 1;
    }
    Eigen::MatrixXd batch(2, 1);
    batch << 0.0, 1.0;
    const auto batch_distances = sphere_model.BatchMinimumDistances(batch);
    if (std::abs(batch_distances[0] + 0.4) > 1e-12 ||
        std::abs(batch_distances[1] - 0.6) > 1e-12) {
      std::cerr << "sphere collision batch query failed\n";
      return 1;
    }

    CollisionModel model(path.string(), {}, false);
    if (model.GetConfigurationSize() != 1 || model.GetGeometryCount() != 2 ||
        model.GetCollisionPairCount() != 1) {
      std::cerr << "unexpected collision model dimensions\n";
      return 1;
    }
    const auto grouped_pairs = model.SetCollisionGroups(
        {{"left_arm", {"base"}}, {"right_arm", {"slider"}}},
        {{"left_arm", "right_arm"}});
    if (grouped_pairs != 1 ||
        model.GetCollisionPairs().front().first_link != "base" ||
        model.GetCollisionPairs().front().second_link != "slider") {
      std::cerr << "collision group filtering failed\n";
      return 1;
    }
    if (model.RemoveCollisionPairsByLinks("base", "slider") != 1 ||
        model.GetCollisionPairCount() != 0) {
      std::cerr << "link collision pair removal failed\n";
      return 1;
    }
    model.ResetCollisionPairs(false);

    const auto lower = model.GetJointLowerLimits({"slide"});
    const auto upper = model.GetJointUpperLimits({"slide"});
    if (lower.size() != 1 || lower[0] != 0.0 || upper[0] != 1.0) {
      std::cerr << "joint limit selection failed\n";
      return 1;
    }

    Eigen::VectorXd q(1);
    q << 0.0;
    if (!model.InCollision(q) || model.ComputeCollisions(q).size() != 1) {
      std::cerr << "overlapping spheres were not detected\n";
      return 1;
    }
    const auto report = model.Evaluate(q);
    if (!report.in_collision || !report.has_minimum_distance ||
        report.collisions.size() != 1 || report.query_time_ms < 0.0) {
      std::cerr << "combined collision report failed\n";
      return 1;
    }

    q << 1.0;
    const auto replaced = model.ConfigurationWithJointPositions(
        Eigen::VectorXd::Zero(1), {"slide"}, q);
    if ((replaced - q).norm() > 1e-12) {
      std::cerr << "joint configuration replacement failed\n";
      return 1;
    }
    if (model.InCollision(q)) {
      std::cerr << "separated spheres reported a collision\n";
      return 1;
    }
    if (!model.IsWithinDistance(q, 0.7) || model.IsWithinDistance(q, 0.5)) {
      std::cerr << "security-margin collision query failed\n";
      return 1;
    }
    const auto distance = model.MinimumDistance(q);
    if (std::abs(distance.distance - 0.6) > 1e-6) {
      std::cerr << "unexpected minimum distance: " << distance.distance << '\n';
      return 1;
    }
    if (model.ResetCollisionPairs(false) != 1) {
      std::cerr << "collision pair reset failed\n";
      return 1;
    }
    Eigen::Matrix4d obstacle_pose = Eigen::Matrix4d::Identity();
    obstacle_pose(0, 3) = 0.5;
    model.AddBoxObstacle("test_wall", Eigen::Vector3d(0.2, 0.5, 0.5),
                         obstacle_pose);
    if (model.SetCollisionGroups({{"robot", {"slider"}},
                                  {"environment", {"test_wall"}}},
                                 {{"robot", "environment"}}) != 1) {
      std::cerr << "environment collision pair setup failed\n";
      return 1;
    }
    q << 0.5;
    if (!model.InCollision(q) || !model.RemoveObstacle("test_wall") ||
        model.RemoveObstacle("test_wall")) {
      std::cerr << "box obstacle lifecycle failed\n";
      return 1;
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::filesystem::remove(path);
  return 0;
}
