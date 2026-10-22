#include "holistic_motion/collision/CollisionModel.h"
#include "JointCoordinates.h"
#include "../robot/URDFModel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/collision/collision.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <coal/shape/geometric_shapes.h>

namespace holistic_motion::robotics::collision {

class CollisionModel::Impl {
public:
  Impl(const std::string &urdf_path,
       const std::vector<std::string> &package_dirs, bool exclude_adjacent) {
    if (urdf_path.empty()) {
      throw std::invalid_argument("URDF path must not be empty");
    }
    std::string xml_snapshot;
    const auto urdf_model = urdf_internal::LoadModel(urdf_path, &xml_snapshot);
    if (!urdf_model)
      throw std::invalid_argument("Unable to load URDF: " + urdf_path);
    pinocchio::urdf::buildModel(urdf_model, model);
    // Geometry and kinematics consume the same source bytes, even if the
    // caller replaces or streams the URDF while construction is in progress.
    std::istringstream geometry_source(xml_snapshot);
    pinocchio::urdf::buildGeom(model, geometry_source, pinocchio::COLLISION,
                               geometry_model, package_dirs);
    if (geometry_model.geometryObjects.empty()) {
      throw std::runtime_error("URDF contains no collision geometry: " +
                               urdf_path);
    }
    geometry_model.addAllCollisionPairs();
    if (exclude_adjacent)
      RemoveAdjacentCollisionPairs();
    // Kinematic topology never changes after construction. Geometry/pair
    // edits need new geometry buffers, while this FK workspace stays valid.
    data = std::make_unique<pinocchio::Data>(model);
    RebuildGeometryData();
  }

  explicit Impl(const Impl &source)
      : model(source.model), geometry_model(source.geometry_model.clone()) {
      data = std::make_unique<pinocchio::Data>(model);
      RebuildGeometryData();
  }

  void Validate(const Eigen::VectorXd &q) const {
    if (q.size() != model.nq) {
      throw std::invalid_argument("configuration size must equal model.nq (" +
                                  std::to_string(model.nq) + ")");
    }
    if (!q.allFinite()) {
      throw std::invalid_argument("configuration must contain finite values");
    }
    if (model.nq != model.nv && !pinocchio::isNormalized(model, q, 1e-8)) {
        throw std::invalid_argument(
            "configuration must satisfy the model manifold (normalized "
            "continuous pairs/quaternions)");
    }
  }

  void RebuildGeometryData() {
    geometry_data = std::make_unique<pinocchio::GeometryData>(geometry_model);
    for (auto &request : geometry_data->distanceRequests) {
      request.enable_nearest_points = true;
      request.enable_signed_distance = true;
    }
  }

  void UpdatePlacements(const Eigen::VectorXd &q) {
    pinocchio::updateGeometryPlacements(model, *data, geometry_model,
                                        *geometry_data, q);
    for (const auto &placement : geometry_data->oMg) {
      if (!placement.translation().allFinite() ||
          !placement.rotation().allFinite()) {
        throw std::overflow_error("collision geometry placement is not finite");
      }
    }
  }

  std::size_t MinimumDistanceIndex() {
    const auto index = pinocchio::computeDistances(geometry_model, *geometry_data);
    // Coal may leave every result at its sentinel when finite placements are
    // too far apart for its distance arithmetic. That index is not a pair.
    if (index >= geometry_model.collisionPairs.size())
      throw std::overflow_error("collision distance is not representable");
    return index;
  }

  bool IsAdjacent(std::size_t first_geometry,
                  std::size_t second_geometry) const {
    const auto first =
        geometry_model.geometryObjects[first_geometry].parentJoint;
    const auto second =
        geometry_model.geometryObjects[second_geometry].parentJoint;
    return first == second ||
           (first < model.parents.size() && model.parents[first] == second) ||
           (second < model.parents.size() && model.parents[second] == first);
  }

  std::string LinkName(std::size_t geometry) const {
    const auto frame = geometry_model.geometryObjects.at(geometry).parentFrame;
    if (frame >= model.frames.size())
      return geometry_model.geometryObjects.at(geometry).name;
    return model.frames[frame].name;
  }

  std::size_t RemoveAdjacentCollisionPairs() {
    const auto before = geometry_model.collisionPairs.size();
    std::vector<pinocchio::CollisionPair> keep;
    keep.reserve(before);
    for (const auto &pair : geometry_model.collisionPairs) {
      if (!IsAdjacent(pair.first, pair.second))
        keep.push_back(pair);
    }
    geometry_model.removeAllCollisionPairs();
    for (const auto &pair : keep)
      geometry_model.addCollisionPair(pair);
    return before - keep.size();
  }

  CollisionResult MakeResult(std::size_t index,
                             const coal::DistanceResult &distance) const {
    if (!std::isfinite(distance.min_distance))
      throw std::overflow_error("collision distance is not finite");
    if (!distance.nearest_points[0].allFinite() ||
        !distance.nearest_points[1].allFinite())
      throw std::overflow_error("collision nearest points are not finite");
    const auto &pair = geometry_model.collisionPairs.at(index);
    CollisionResult result;
    result.pair_index = index;
    result.first_geometry = geometry_model.geometryObjects[pair.first].name;
    result.second_geometry = geometry_model.geometryObjects[pair.second].name;
    result.distance = distance.min_distance;
    result.nearest_point_first = distance.nearest_points[0];
    result.nearest_point_second = distance.nearest_points[1];
    return result;
  }

  pinocchio::Model model;
  pinocchio::GeometryModel geometry_model;
  std::unique_ptr<pinocchio::Data> data;
  std::unique_ptr<pinocchio::GeometryData> geometry_data;
};

CollisionModel::CollisionModel(const std::string &urdf_path,
                               const std::vector<std::string> &package_dirs,
                               bool exclude_adjacent)
    : impl_(std::make_unique<Impl>(urdf_path, package_dirs, exclude_adjacent)) {
}

CollisionModel::~CollisionModel() = default;
CollisionModel::CollisionModel(CollisionModel &&) noexcept = default;
CollisionModel::CollisionModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

CollisionModel CollisionModel::Clone() const {
    if (!impl_)
        throw std::logic_error("cannot clone a moved-from collision model");
    return CollisionModel(std::make_unique<Impl>(*impl_));
}
CollisionModel &CollisionModel::operator=(CollisionModel &&) noexcept = default;

int CollisionModel::GetConfigurationSize() const { return impl_->model.nq; }
int CollisionModel::GetVelocitySize() const { return impl_->model.nv; }
void CollisionModel::ValidateConfiguration(const Eigen::VectorXd &q) const {
    impl_->Validate(q);
}
std::size_t CollisionModel::GetGeometryCount() const {
  return impl_->geometry_model.geometryObjects.size();
}
std::size_t CollisionModel::GetCollisionPairCount() const {
  return impl_->geometry_model.collisionPairs.size();
}
std::vector<std::string> CollisionModel::GetGeometryNames() const {
  std::vector<std::string> names;
  names.reserve(impl_->geometry_model.geometryObjects.size());
  for (const auto &object : impl_->geometry_model.geometryObjects)
    names.push_back(object.name);
  return names;
}

std::vector<std::string> CollisionModel::GetCollisionLinkNames() const {
  std::set<std::string> unique;
  for (std::size_t index = 0;
       index < impl_->geometry_model.geometryObjects.size(); ++index) {
    unique.insert(impl_->LinkName(index));
  }
  return {unique.begin(), unique.end()};
}

std::vector<CollisionPairInfo> CollisionModel::GetCollisionPairs() const {
  std::vector<CollisionPairInfo> result;
  result.reserve(impl_->geometry_model.collisionPairs.size());
  for (const auto &pair : impl_->geometry_model.collisionPairs) {
    result.push_back({impl_->geometry_model.geometryObjects[pair.first].name,
                      impl_->geometry_model.geometryObjects[pair.second].name,
                      impl_->LinkName(pair.first),
                      impl_->LinkName(pair.second)});
  }
  return result;
}

Eigen::VectorXd CollisionModel::NeutralConfiguration() const {
  return pinocchio::neutral(impl_->model);
}

std::vector<JointCoordinate> CollisionModel::GetJointCoordinates() const {
    return detail::JointCoordinates(impl_->model);
}

JointPositionMap CollisionModel::CreateJointPositionMap(
    const std::vector<std::string> &names) const {
    return detail::PositionMap(impl_->model, names);
}

Eigen::VectorXd CollisionModel::ConfigurationFromJointPositions(
    const std::map<std::string, double> &positions) const {
    std::vector<std::string> names;
    Eigen::VectorXd values(static_cast<Eigen::Index>(positions.size()));
    names.reserve(positions.size());
    Eigen::Index i = 0;
    for (const auto &entry : positions) {
        names.push_back(entry.first);
        values[i++] = entry.second;
    }
    return CreateJointPositionMap(names).ConfigurationWithJointPositions(
        pinocchio::neutral(impl_->model), values);
}

Eigen::VectorXd CollisionModel::ConfigurationWithJointPositions(
    const Eigen::VectorXd &reference, const std::vector<std::string> &names,
    const Eigen::VectorXd &positions) const {
    auto result = CreateJointPositionMap(names).ConfigurationWithJointPositions(
        reference, positions);
    ValidateConfiguration(result);
    return result;
}

Eigen::VectorXd CollisionModel::GetJointLowerLimits(
    const std::vector<std::string> &names) const {
    return detail::JointLimits(impl_->model, names, false);
}

Eigen::VectorXd CollisionModel::GetJointUpperLimits(
    const std::vector<std::string> &names) const {
    return detail::JointLimits(impl_->model, names, true);
}

void CollisionModel::AddBoxObstacle(const std::string &name,
                                    const Eigen::Vector3d &size,
                                    const Eigen::Matrix4d &pose) {
  if (name.empty() || !size.allFinite() || (size.array() <= 0.0).any() ||
      !pose.allFinite()) {
    throw std::invalid_argument(
        "box obstacle requires a name, positive size, and finite pose");
  }
  if (impl_->geometry_model.existGeometryName(name)) {
    throw std::invalid_argument("collision geometry already exists: " + name);
  }
  const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
  if ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() >
          1e-6 ||
      std::abs(rotation.determinant() - 1.0) > 1e-6 ||
      (pose.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).norm() >
          1e-9) {
    throw std::invalid_argument("box obstacle pose must be a rigid transform");
  }
  auto geometry = std::make_shared<coal::Box>(size[0], size[1], size[2]);
  impl_->geometry_model.addGeometryObject(pinocchio::GeometryObject(
      name, 0,
      pinocchio::SE3(rotation, pose.topRightCorner<3, 1>()), geometry));
  impl_->RebuildGeometryData();
}

bool CollisionModel::RemoveObstacle(const std::string &name) {
  if (!impl_->geometry_model.existGeometryName(name))
    return false;
  const auto id = impl_->geometry_model.getGeometryId(name);
  if (id >= impl_->geometry_model.geometryObjects.size() ||
      impl_->geometry_model.geometryObjects[id].parentFrame <
          impl_->model.frames.size()) {
    throw std::invalid_argument("geometry is not a world obstacle: " + name);
  }
  impl_->geometry_model.removeGeometryObject(name);
  impl_->RebuildGeometryData();
  return true;
}

bool CollisionModel::RemoveCollisionPair(const std::string &first,
                                         const std::string &second) {
  const auto first_id = impl_->geometry_model.getGeometryId(first);
  const auto second_id = impl_->geometry_model.getGeometryId(second);
  if (first_id >= impl_->geometry_model.geometryObjects.size() ||
      second_id >= impl_->geometry_model.geometryObjects.size()) {
    throw std::invalid_argument("unknown collision geometry name");
  }
  const auto before = impl_->geometry_model.collisionPairs.size();
  impl_->geometry_model.removeCollisionPair(
      pinocchio::CollisionPair(first_id, second_id));
  if (impl_->geometry_model.collisionPairs.size() != before)
    impl_->RebuildGeometryData();
  return impl_->geometry_model.collisionPairs.size() != before;
}

std::size_t
CollisionModel::RemoveCollisionPairsByLinks(const std::string &first_link,
                                            const std::string &second_link) {
  const auto before = impl_->geometry_model.collisionPairs.size();
  std::vector<pinocchio::CollisionPair> keep;
  keep.reserve(before);
  for (const auto &pair : impl_->geometry_model.collisionPairs) {
    const auto first = impl_->LinkName(pair.first);
    const auto second = impl_->LinkName(pair.second);
    const bool matches = (first == first_link && second == second_link) ||
                         (first == second_link && second == first_link);
    if (!matches)
      keep.push_back(pair);
  }
  impl_->geometry_model.removeAllCollisionPairs();
  for (const auto &pair : keep) {
    impl_->geometry_model.addCollisionPair(pair);
  }
  const auto removed = before - keep.size();
  if (removed)
    impl_->RebuildGeometryData();
  return removed;
}

std::size_t CollisionModel::RemoveAdjacentCollisionPairs() {
  const auto removed = impl_->RemoveAdjacentCollisionPairs();
  if (removed)
    impl_->RebuildGeometryData();
  return removed;
}

std::size_t CollisionModel::ResetCollisionPairs(bool exclude_adjacent) {
  impl_->geometry_model.removeAllCollisionPairs();
  impl_->geometry_model.addAllCollisionPairs();
  if (exclude_adjacent)
    impl_->RemoveAdjacentCollisionPairs();
  impl_->RebuildGeometryData();
  return impl_->geometry_model.collisionPairs.size();
}

void CollisionModel::ClearCollisionPairs() {
  impl_->geometry_model.removeAllCollisionPairs();
  impl_->RebuildGeometryData();
}

std::size_t CollisionModel::SetCollisionGroups(
    const std::map<std::string, std::vector<std::string>> &groups,
    const std::vector<std::pair<std::string, std::string>> &group_pairs) {
  if (groups.empty() || group_pairs.empty()) {
    throw std::invalid_argument(
        "collision groups and group pairs must not be empty");
  }
  std::map<std::string, std::string> link_to_group;
  std::set<std::string> model_links;
  for (std::size_t index = 0;
       index < impl_->geometry_model.geometryObjects.size(); ++index) {
    model_links.insert(impl_->LinkName(index));
  }
  for (const auto &[group, links] : groups) {
    if (group.empty() || links.empty()) {
      throw std::invalid_argument(
          "collision group names and link lists must not be empty");
    }
    for (const auto &link : links) {
      if (!model_links.count(link)) {
        throw std::invalid_argument("unknown collision link: " + link);
      }
      if (!link_to_group.emplace(link, group).second) {
        throw std::invalid_argument(
            "collision link belongs to multiple groups: " + link);
      }
    }
  }
  std::set<std::pair<std::string, std::string>> enabled;
  for (auto pair : group_pairs) {
    if (!groups.count(pair.first) || !groups.count(pair.second)) {
      throw std::invalid_argument("collision pair references an unknown group");
    }
    if (pair.second < pair.first)
      std::swap(pair.first, pair.second);
    enabled.insert(std::move(pair));
  }

  impl_->geometry_model.removeAllCollisionPairs();
  const auto count = impl_->geometry_model.geometryObjects.size();
  for (std::size_t first = 0; first < count; ++first) {
    const auto first_group = link_to_group.find(impl_->LinkName(first));
    if (first_group == link_to_group.end())
      continue;
    for (std::size_t second = first + 1; second < count; ++second) {
      const auto second_group = link_to_group.find(impl_->LinkName(second));
      if (second_group == link_to_group.end())
        continue;
      auto key = std::make_pair(first_group->second, second_group->second);
      if (key.second < key.first)
        std::swap(key.first, key.second);
      const bool within_group = first_group->second == second_group->second;
      if (enabled.count(key) &&
          !(within_group && impl_->IsAdjacent(first, second))) {
        impl_->geometry_model.addCollisionPair(
            pinocchio::CollisionPair(first, second));
      }
    }
  }
  impl_->RebuildGeometryData();
  return impl_->geometry_model.collisionPairs.size();
}

bool CollisionModel::InCollision(const Eigen::VectorXd &q, bool stop_at_first) {
  impl_->Validate(q);
  impl_->UpdatePlacements(q);
  return pinocchio::computeCollisions(impl_->geometry_model,
                                      *impl_->geometry_data, stop_at_first);
}

bool CollisionModel::IsWithinDistance(const Eigen::VectorXd &q,
                                      double security_margin,
                                      bool stop_at_first) {
  impl_->Validate(q);
  if (!std::isfinite(security_margin) || security_margin < 0.0) {
    throw std::invalid_argument(
        "security margin must be finite and non-negative");
  }
  impl_->UpdatePlacements(q);
  struct ResetMargins {
    pinocchio::GeometryData &data;
    ~ResetMargins() {
      for (auto &request : data.collisionRequests) {
        request.security_margin = 0.0;
        request.distance_upper_bound = 1e-6;
      }
    }
  } reset_margins{*impl_->geometry_data};
  for (auto &request : impl_->geometry_data->collisionRequests) {
    request.security_margin = security_margin;
    request.distance_upper_bound = security_margin + 1e-6;
    request.enable_contact = false;
    request.num_max_contacts = 1;
  }
  return pinocchio::computeCollisions(
      impl_->geometry_model, *impl_->geometry_data, stop_at_first);
}

std::vector<CollisionResult>
CollisionModel::ComputeCollisions(const Eigen::VectorXd &q) {
  impl_->Validate(q);
  impl_->UpdatePlacements(q);
  pinocchio::computeCollisions(impl_->geometry_model, *impl_->geometry_data, false);
  pinocchio::computeDistances(impl_->geometry_model, *impl_->geometry_data);
  std::vector<CollisionResult> results;
  results.reserve(impl_->geometry_model.collisionPairs.size());
  for (std::size_t index = 0;
       index < impl_->geometry_model.collisionPairs.size(); ++index) {
    if (impl_->geometry_data->collisionResults[index].isCollision()) {
      results.push_back(impl_->MakeResult(
          index, impl_->geometry_data->distanceResults[index]));
    }
  }
  return results;
}

CollisionResult CollisionModel::MinimumDistance(const Eigen::VectorXd &q) {
  impl_->Validate(q);
  if (impl_->geometry_model.collisionPairs.empty()) {
    throw std::runtime_error("collision model has no active collision pairs");
  }
  impl_->UpdatePlacements(q);
  const auto index = impl_->MinimumDistanceIndex();
  return impl_->MakeResult(index, impl_->geometry_data->distanceResults[index]);
}

CollisionReport CollisionModel::Evaluate(const Eigen::VectorXd &q) {
  impl_->Validate(q);
  const auto started = std::chrono::steady_clock::now();
  CollisionReport report;
  if (!impl_->geometry_model.collisionPairs.empty()) {
    impl_->UpdatePlacements(q);
    report.collisions.reserve(impl_->geometry_model.collisionPairs.size());
    report.in_collision = pinocchio::computeCollisions(
        impl_->geometry_model, *impl_->geometry_data, false);
    const auto minimum_index = impl_->MinimumDistanceIndex();
    report.has_minimum_distance = true;
    report.minimum_distance = impl_->MakeResult(
        minimum_index, impl_->geometry_data->distanceResults[minimum_index]);
    for (std::size_t index = 0;
         index < impl_->geometry_model.collisionPairs.size(); ++index) {
      if (impl_->geometry_data->collisionResults[index].isCollision()) {
        report.collisions.push_back(impl_->MakeResult(
            index, impl_->geometry_data->distanceResults[index]));
      }
    }
  }
  report.query_time_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  return report;
}

} // namespace holistic_motion::robotics::collision
