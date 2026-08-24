#pragma once

#include <cstddef>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics::collision {

struct CollisionResult {
    std::size_t pair_index{0};
    std::string first_geometry;
    std::string second_geometry;
    double distance{0.0};
    Eigen::Vector3d nearest_point_first{Eigen::Vector3d::Zero()};
    Eigen::Vector3d nearest_point_second{Eigen::Vector3d::Zero()};
};

struct CollisionPairInfo {
    std::string first_geometry;
    std::string second_geometry;
    std::string first_link;
    std::string second_link;
};

struct CollisionReport {
    bool in_collision{false};
    bool has_minimum_distance{false};
    std::vector<CollisionResult> collisions;
    CollisionResult minimum_distance;
    double query_time_ms{0.0};
};

// Pinocchio/Coal-backed collision geometry kept separate from Robot so meshes
// are loaded only when collision checking is explicitly requested.
class CollisionModel {
public:
    explicit CollisionModel(const std::string& urdf_path,
                            const std::vector<std::string>& package_dirs = {},
                            bool exclude_adjacent = true);
    ~CollisionModel();

    CollisionModel(CollisionModel&&) noexcept;
    CollisionModel& operator=(CollisionModel&&) noexcept;
    CollisionModel(const CollisionModel&) = delete;
    CollisionModel& operator=(const CollisionModel&) = delete;

    int GetConfigurationSize() const;
    int GetVelocitySize() const;
    std::size_t GetGeometryCount() const;
    std::size_t GetCollisionPairCount() const;
    std::vector<std::string> GetGeometryNames() const;
    std::vector<std::string> GetCollisionLinkNames() const;
    std::vector<CollisionPairInfo> GetCollisionPairs() const;
    Eigen::VectorXd NeutralConfiguration() const;
    Eigen::VectorXd ConfigurationFromJointPositions(
            const std::map<std::string, double>& joint_positions) const;

    bool RemoveCollisionPair(const std::string& first_geometry,
                             const std::string& second_geometry);
    std::size_t RemoveCollisionPairsByLinks(const std::string& first_link,
                                            const std::string& second_link);
    std::size_t RemoveAdjacentCollisionPairs();
    std::size_t ResetCollisionPairs(bool exclude_adjacent = true);
    void ClearCollisionPairs();
    std::size_t SetCollisionGroups(
            const std::map<std::string, std::vector<std::string>>& groups,
            const std::vector<std::pair<std::string, std::string>>& group_pairs);

    bool InCollision(const Eigen::VectorXd& configuration,
                     bool stop_at_first = true);
    bool IsWithinDistance(const Eigen::VectorXd& configuration,
                          double security_margin,
                          bool stop_at_first = true);
    std::vector<CollisionResult> ComputeCollisions(
            const Eigen::VectorXd& configuration);
    CollisionResult MinimumDistance(const Eigen::VectorXd& configuration);
    CollisionReport Evaluate(const Eigen::VectorXd& configuration);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace holistic_motion::robotics::collision
