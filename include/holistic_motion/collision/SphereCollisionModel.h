#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics::collision {

/// A collision sphere expressed in its owning link frame.
struct CollisionSphere {
  std::string name;
  std::string link_name;
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  double radius{0.0};
};

struct SpherePairInfo {
  std::size_t first{0};
  std::size_t second{0};
  std::string first_sphere;
  std::string second_sphere;
  std::string first_link;
  std::string second_link;
};

struct SphereDistanceResult {
  std::size_t pair_index{0};
  std::size_t first{0};
  std::size_t second{0};
  double distance{0.0};
  Eigen::Vector3d first_center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d second_center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitX()};
};

/// Pinocchio-backed collision-sphere model for inexpensive repeated queries.
///
/// Spheres are caller-provided and remain independent of the URDF collision
/// meshes. Coal should be used when exact geometry validation is required.
class SphereCollisionModel {
public:
  SphereCollisionModel(const std::string &urdf_path,
                       std::vector<CollisionSphere> spheres,
                       bool exclude_same_link = true);
  ~SphereCollisionModel();

  SphereCollisionModel(SphereCollisionModel &&) noexcept;
  SphereCollisionModel &operator=(SphereCollisionModel &&) noexcept;
  SphereCollisionModel(const SphereCollisionModel &) = delete;
  SphereCollisionModel &operator=(const SphereCollisionModel &) = delete;

  int GetConfigurationSize() const;
  int GetVelocitySize() const;
  std::size_t GetSphereCount() const;
  std::size_t GetCollisionPairCount() const;
  const std::vector<CollisionSphere> &GetSpheres() const;
  std::vector<SpherePairInfo> GetCollisionPairs() const;
  Eigen::VectorXd NeutralConfiguration() const;

  /// Replace active pairs with the Cartesian product of selected link groups.
  std::size_t SetCollisionGroups(
      const std::map<std::string, std::vector<std::string>> &groups,
      const std::vector<std::pair<std::string, std::string>> &group_pairs);
  std::size_t ResetCollisionPairs(bool exclude_same_link = true);
  void ClearCollisionPairs();

  /// Return world-space spheres as rows [x, y, z, radius].
  Eigen::Matrix<double, Eigen::Dynamic, 4>
  ComputeWorldSpheres(const Eigen::VectorXd &configuration);
  bool InCollision(const Eigen::VectorXd &configuration,
                   double security_margin = 0.0,
                   bool stop_at_first = true);
  SphereDistanceResult
  MinimumDistance(const Eigen::VectorXd &configuration);
  std::vector<SphereDistanceResult>
  ComputeDistances(const Eigen::VectorXd &configuration,
                   double maximum_distance);
  /// Each row is a complete Pinocchio configuration; output has one value per row.
  Eigen::VectorXd
  BatchMinimumDistances(const Eigen::MatrixXd &configurations);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace holistic_motion::robotics::collision
