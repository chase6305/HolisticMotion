#include "holistic_motion/collision/SphereCollisionModel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace holistic_motion::robotics::collision {
namespace {

using Pair = std::pair<std::size_t, std::size_t>;

Pair OrderedPair(std::size_t first, std::size_t second) {
  return first < second ? Pair{first, second} : Pair{second, first};
}

Eigen::Matrix3d Skew(const Eigen::Vector3d &value) {
  Eigen::Matrix3d result;
  result << 0.0, -value.z(), value.y(), value.z(), 0.0, -value.x(), -value.y(),
      value.x(), 0.0;
  return result;
}

} // namespace

class SphereCollisionModel::Impl {
public:
  Impl(const std::string &urdf_path, std::vector<CollisionSphere> input_spheres,
       bool exclude_same_link)
      : spheres(std::move(input_spheres)) {
    if (urdf_path.empty())
      throw std::invalid_argument("URDF path must not be empty");
    pinocchio::urdf::buildModel(urdf_path, model);
    data = std::make_unique<pinocchio::Data>(model);
    if (spheres.empty())
      throw std::invalid_argument("collision sphere list must not be empty");

    std::set<std::string> names;
    frame_ids.reserve(spheres.size());
    for (const auto &sphere : spheres) {
      if (sphere.name.empty() || sphere.link_name.empty())
        throw std::invalid_argument(
            "sphere name and link name must not be empty");
      if (!sphere.center.allFinite() || !std::isfinite(sphere.radius) ||
          sphere.radius <= 0.0)
        throw std::invalid_argument(
            "sphere center and radius must be finite and radius positive");
      if (!names.insert(sphere.name).second)
        throw std::invalid_argument("duplicate collision sphere name: " +
                                    sphere.name);
      const auto frame_id = model.getFrameId(sphere.link_name);
      if (frame_id >= static_cast<pinocchio::FrameIndex>(model.nframes))
        throw std::invalid_argument("unknown Pinocchio link frame: " +
                                    sphere.link_name);
      frame_ids.push_back(frame_id);
    }
    world_spheres.resize(static_cast<Eigen::Index>(spheres.size()), 4);
    frame_jacobian.resize(6, model.nv);
    ResetPairs(exclude_same_link);
  }

  void ValidateConfiguration(const Eigen::VectorXd &q) const {
    if (q.size() != model.nq)
      throw std::invalid_argument("configuration size must equal model.nq (" +
                                  std::to_string(model.nq) + ")");
    if (!q.allFinite())
      throw std::invalid_argument("configuration must contain finite values");
  }

  void Update(const Eigen::VectorXd &q) {
    ValidateConfiguration(q);
    pinocchio::forwardKinematics(model, *data, q);
    pinocchio::updateFramePlacements(model, *data);
    for (std::size_t i = 0; i < spheres.size(); ++i) {
      const auto &placement = data->oMf[frame_ids[i]];
      world_spheres.row(static_cast<Eigen::Index>(i)).head<3>() =
          (placement.rotation() * spheres[i].center + placement.translation())
              .transpose();
      world_spheres(static_cast<Eigen::Index>(i), 3) = spheres[i].radius;
    }
  }

  void UpdateWithJacobians(const Eigen::VectorXd &q) {
    ValidateConfiguration(q);
    pinocchio::computeJointJacobians(model, *data, q);
    pinocchio::updateFramePlacements(model, *data);
    for (std::size_t i = 0; i < spheres.size(); ++i) {
      const auto &placement = data->oMf[frame_ids[i]];
      world_spheres.row(static_cast<Eigen::Index>(i)).head<3>() =
          (placement.rotation() * spheres[i].center + placement.translation())
              .transpose();
      world_spheres(static_cast<Eigen::Index>(i), 3) = spheres[i].radius;
    }
  }

  Eigen::Matrix<double, 3, Eigen::Dynamic>
  PointJacobian(std::size_t sphere_index) const {
    frame_jacobian.setZero();
    pinocchio::getFrameJacobian(model, *data, frame_ids[sphere_index],
                                pinocchio::LOCAL_WORLD_ALIGNED, frame_jacobian);
    const Eigen::Vector3d center =
        world_spheres.row(static_cast<Eigen::Index>(sphere_index)).head<3>();
    const Eigen::Vector3d offset =
        center - data->oMf[frame_ids[sphere_index]].translation();
    return frame_jacobian.topRows<3>() -
           Skew(offset) * frame_jacobian.bottomRows<3>();
  }

  void ResetPairs(bool exclude_same_link) {
    pairs.clear();
    for (std::size_t first = 0; first < spheres.size(); ++first) {
      for (std::size_t second = first + 1; second < spheres.size(); ++second) {
        if (!exclude_same_link ||
            spheres[first].link_name != spheres[second].link_name)
          pairs.emplace_back(first, second);
      }
    }
    ++pair_revision;
  }

  SphereDistanceResult Distance(std::size_t pair_index) const {
    const auto [first, second] = pairs[pair_index];
    const Eigen::Vector3d first_center =
        world_spheres.row(static_cast<Eigen::Index>(first)).head<3>();
    const Eigen::Vector3d second_center =
        world_spheres.row(static_cast<Eigen::Index>(second)).head<3>();
    const Eigen::Vector3d delta = second_center - first_center;
    const double center_distance = delta.norm();
    SphereDistanceResult result;
    result.pair_index = pair_index;
    result.first = first;
    result.second = second;
    result.distance =
        center_distance - spheres[first].radius - spheres[second].radius;
    result.first_center = first_center;
    result.second_center = second_center;
    if (center_distance > 1e-12)
      result.normal = delta / center_distance;
    return result;
  }

  double DistanceValue(std::size_t pair_index) const {
    const auto [first, second] = pairs[pair_index];
    const Eigen::Vector3d delta =
        world_spheres.row(static_cast<Eigen::Index>(second)).head<3>() -
        world_spheres.row(static_cast<Eigen::Index>(first)).head<3>();
    return delta.norm() - spheres[first].radius - spheres[second].radius;
  }

  double MinimumDistanceValue() const {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < pairs.size(); ++i)
      best = std::min(best, DistanceValue(i));
    return best;
  }

  std::size_t MinimumDistanceIndex() const {
    std::size_t best_index = 0;
    double best_distance = DistanceValue(0);
    for (std::size_t i = 1; i < pairs.size(); ++i) {
      const double distance = DistanceValue(i);
      if (distance < best_distance) {
        best_distance = distance;
        best_index = i;
      }
    }
    return best_index;
  }

  pinocchio::Model model;
  std::unique_ptr<pinocchio::Data> data;
  std::vector<CollisionSphere> spheres;
  std::vector<pinocchio::FrameIndex> frame_ids;
  std::vector<Pair> pairs;
  std::size_t pair_revision{0};
  Eigen::Matrix<double, Eigen::Dynamic, 4> world_spheres;
  mutable Eigen::Matrix<double, 6, Eigen::Dynamic> frame_jacobian;
};

SphereCollisionModel::SphereCollisionModel(const std::string &urdf_path,
                                           std::vector<CollisionSphere> spheres,
                                           bool exclude_same_link)
    : impl_(std::make_unique<Impl>(urdf_path, std::move(spheres),
                                   exclude_same_link)) {}
SphereCollisionModel::~SphereCollisionModel() = default;
SphereCollisionModel::SphereCollisionModel(SphereCollisionModel &&) noexcept =
    default;
SphereCollisionModel &
SphereCollisionModel::operator=(SphereCollisionModel &&) noexcept = default;

int SphereCollisionModel::GetConfigurationSize() const {
  return impl_->model.nq;
}
int SphereCollisionModel::GetVelocitySize() const { return impl_->model.nv; }
std::size_t SphereCollisionModel::GetSphereCount() const {
  return impl_->spheres.size();
}
std::size_t SphereCollisionModel::GetCollisionPairCount() const {
  return impl_->pairs.size();
}
std::size_t SphereCollisionModel::GetCollisionPairRevision() const {
  return impl_->pair_revision;
}
const std::vector<CollisionSphere> &SphereCollisionModel::GetSpheres() const {
  return impl_->spheres;
}

std::vector<SpherePairInfo> SphereCollisionModel::GetCollisionPairs() const {
  std::vector<SpherePairInfo> result;
  result.reserve(impl_->pairs.size());
  for (const auto &[first, second] : impl_->pairs) {
    result.push_back(
        {first, second, impl_->spheres[first].name, impl_->spheres[second].name,
         impl_->spheres[first].link_name, impl_->spheres[second].link_name});
  }
  return result;
}

Eigen::VectorXd SphereCollisionModel::NeutralConfiguration() const {
  return pinocchio::neutral(impl_->model);
}

std::size_t SphereCollisionModel::SetCollisionGroups(
    const std::map<std::string, std::vector<std::string>> &groups,
    const std::vector<std::pair<std::string, std::string>> &group_pairs) {
  std::map<std::string, std::vector<std::size_t>> resolved;
  for (const auto &[group_name, links] : groups) {
    if (group_name.empty() || links.empty())
      throw std::invalid_argument(
          "collision groups require a name and at least one link");
    auto &indices = resolved[group_name];
    for (const auto &link : links) {
      bool found = false;
      for (std::size_t i = 0; i < impl_->spheres.size(); ++i) {
        if (impl_->spheres[i].link_name == link) {
          indices.push_back(i);
          found = true;
        }
      }
      if (!found)
        throw std::invalid_argument(
            "collision group references unknown sphere link: " + link);
    }
  }
  std::set<Pair> selected;
  for (const auto &[first_group, second_group] : group_pairs) {
    const auto first = resolved.find(first_group);
    const auto second = resolved.find(second_group);
    if (first == resolved.end() || second == resolved.end())
      throw std::invalid_argument("collision pair references an unknown group");
    for (std::size_t i : first->second) {
      for (std::size_t j : second->second) {
        if (i != j)
          selected.insert(OrderedPair(i, j));
      }
    }
  }
  impl_->pairs.assign(selected.begin(), selected.end());
  ++impl_->pair_revision;
  return impl_->pairs.size();
}

std::size_t SphereCollisionModel::ResetCollisionPairs(bool exclude_same_link) {
  impl_->ResetPairs(exclude_same_link);
  return impl_->pairs.size();
}
void SphereCollisionModel::ClearCollisionPairs() {
  impl_->pairs.clear();
  ++impl_->pair_revision;
}

Eigen::Matrix<double, Eigen::Dynamic, 4>
SphereCollisionModel::ComputeWorldSpheres(
    const Eigen::VectorXd &configuration) {
  impl_->Update(configuration);
  return impl_->world_spheres;
}

bool SphereCollisionModel::InCollision(const Eigen::VectorXd &configuration,
                                       double security_margin,
                                       bool stop_at_first) {
  if (!std::isfinite(security_margin) || security_margin < 0.0)
    throw std::invalid_argument(
        "security margin must be finite and non-negative");
  impl_->Update(configuration);
  bool collision = false;
  for (const auto &[first, second] : impl_->pairs) {
    const Eigen::Vector3d delta =
        impl_->world_spheres.row(static_cast<Eigen::Index>(second)).head<3>() -
        impl_->world_spheres.row(static_cast<Eigen::Index>(first)).head<3>();
    const double threshold = impl_->spheres[first].radius +
                             impl_->spheres[second].radius + security_margin;
    if (delta.squaredNorm() <= threshold * threshold) {
      collision = true;
      if (stop_at_first)
        break;
    }
  }
  return collision;
}

SphereDistanceResult
SphereCollisionModel::MinimumDistance(const Eigen::VectorXd &configuration) {
  if (impl_->pairs.empty())
    throw std::runtime_error(
        "sphere collision model has no active collision pairs");
  impl_->Update(configuration);
  return impl_->Distance(impl_->MinimumDistanceIndex());
}

Eigen::VectorXd SphereCollisionModel::MinimumDistanceGradient(
    const Eigen::VectorXd &configuration) {
  return MinimumDistanceWithGradient(configuration).gradient;
}

SphereDistanceGradientResult SphereCollisionModel::MinimumDistanceWithGradient(
    const Eigen::VectorXd &configuration) {
  if (impl_->pairs.empty())
    throw std::runtime_error(
        "sphere collision model has no active collision pairs");
  impl_->UpdateWithJacobians(configuration);
  const std::size_t best_index = impl_->MinimumDistanceIndex();
  SphereDistanceResult best = impl_->Distance(best_index);
  const auto [first, second] = impl_->pairs[best_index];
  const double center_distance = best.distance + impl_->spheres[first].radius +
                                 impl_->spheres[second].radius;
  SphereDistanceGradientResult result;
  result.distance_result = best;
  if (center_distance <= 1e-12) {
    result.gradient = Eigen::VectorXd::Zero(impl_->model.nv);
  } else {
    result.gradient = (best.normal.transpose() * (impl_->PointJacobian(second) -
                                                  impl_->PointJacobian(first)))
                          .transpose();
  }
  return result;
}

std::vector<SphereDistanceResult>
SphereCollisionModel::ComputeDistances(const Eigen::VectorXd &configuration,
                                       double maximum_distance) {
  if (!std::isfinite(maximum_distance))
    throw std::invalid_argument("maximum distance must be finite");
  impl_->Update(configuration);
  std::vector<SphereDistanceResult> result;
  result.reserve(impl_->pairs.size());
  for (std::size_t i = 0; i < impl_->pairs.size(); ++i) {
    auto distance = impl_->Distance(i);
    if (distance.distance <= maximum_distance)
      result.push_back(std::move(distance));
  }
  return result;
}

Eigen::VectorXd SphereCollisionModel::BatchMinimumDistances(
    const Eigen::MatrixXd &configurations) {
  if (configurations.cols() != impl_->model.nq || configurations.rows() == 0 ||
      !configurations.allFinite())
    throw std::invalid_argument(
        "configurations must be a finite, non-empty [batch, nq] matrix");
  if (impl_->pairs.empty())
    throw std::runtime_error(
        "sphere collision model has no active collision pairs");
  Eigen::VectorXd output(configurations.rows());
  Eigen::VectorXd configuration(impl_->model.nq);
  for (Eigen::Index row = 0; row < configurations.rows(); ++row) {
    configuration = configurations.row(row).transpose();
    impl_->Update(configuration);
    output[row] = impl_->MinimumDistanceValue();
  }
  return output;
}

} // namespace holistic_motion::robotics::collision
