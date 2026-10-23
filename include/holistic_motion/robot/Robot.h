#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/robot/Joint.h"
#include "holistic_motion/robot/JointPositionMap.h"
#include "holistic_motion/robot/Link.h"

namespace holistic_motion::robotics {

/// Parsed model and chain factories. Concurrent reads require unchanged model
/// fields and visuals; synchronize LoadURDF/LoadVisuals and public Joint/Link
/// edits. Copying Robot shares Joint/Link objects and the default solver;
/// construct independent models when workers need to edit model/solver state.
class Robot {
public:
    Robot() = default;
    explicit Robot(const std::string& urdf_path, bool load_visuals = false);

    static std::shared_ptr<Robot> Create(const std::string& urdf_path,
                                         bool load_visuals = false);

    /// Import scalar URDF joint axes as unit directions. Failed parsing or
    /// model validation leaves the previously loaded model unchanged.
    bool LoadURDF(const std::string& urdf_path, bool load_visuals = false);
    /// Publish visual metadata from the loaded URDF snapshot. Later directory
    /// or source-file changes do not alter it; mesh assets are not loaded.
    bool LoadVisuals();
    bool HasVisualsLoaded() const noexcept { return visuals_loaded_; }
    bool HasKinematics() const noexcept { return static_cast<bool>(kinematics_); }
    const std::string& GetName() const noexcept { return name_; }
    const std::string& GetURDFPath() const noexcept { return urdf_path_; }
    const std::string& GetRootLinkName() const noexcept { return root_link_name_; }
    int GetDoF() const noexcept { return static_cast<int>(actuated_joints_.size()); }
    int GetModelDoF() const noexcept {
        return static_cast<int>(all_actuated_joints_.size());
    }
    const std::vector<std::shared_ptr<Joint>>& GetJoints() const noexcept {
        return joints_;
    }
    const std::vector<std::shared_ptr<Joint>>& GetActuatedJoints() const noexcept {
        return actuated_joints_;
    }
    const std::vector<std::shared_ptr<Joint>>&
    GetAllActuatedJoints() const noexcept {
        return all_actuated_joints_;
    }
    const std::vector<std::shared_ptr<Link>>& GetLinks() const noexcept {
        return links_;
    }
    std::shared_ptr<Link> GetLink(const std::string& name) const noexcept;
    std::shared_ptr<Joint> GetJoint(const std::string& name) const noexcept;
    std::shared_ptr<NumericalKinematics> GetKinematics() const noexcept {
        return kinematics_;
    }
    /// Build from the current model names and parent relations. Return nullptr
    /// for a missing, unsupported, or cyclic path, or one with no active joints.
    std::shared_ptr<NumericalKinematics> CreateKinematics(
            const std::string& base_link,
            const std::string& tip_link) const;
    /// Scalar full-model coordinates, in GetAllActuatedJoints() order.
    /// This representation uses one angle per continuous joint (nq=nv).
    std::vector<JointCoordinate> GetJointCoordinates() const;
    /// Mapping requires unambiguous, nonempty scalar model joint names.
    JointPositionMap
    CreateJointPositionMap(const std::vector<std::string> &joint_names) const;
    /// Ordered active names for the same supported path as CreateKinematics().
    /// Returns an empty vector for an invalid or fixed-only path.
    std::vector<std::string>
    GetChainJointNames(const std::string &base_link,
                       const std::string &tip_link) const;
    std::shared_ptr<NumericalKinematics> CreateFEPKinematics(
            const std::string& base_link,
            const std::string& tip_link) const;

private:
  std::vector<const Joint *> GetChainPath(const std::string &base_link,
                                          const std::string &tip_link) const;
  std::string name_;
  std::string urdf_path_;
  std::string root_link_name_;
  std::vector<std::shared_ptr<Joint>> joints_;
  std::vector<std::shared_ptr<Joint>> actuated_joints_;
  std::vector<std::shared_ptr<Joint>> all_actuated_joints_;
  std::vector<std::shared_ptr<Link>> links_;
  std::vector<std::pair<std::size_t, std::vector<VisualGeometry>>> pending_visuals_;
  std::vector<JointNode> joint_nodes_;
  std::shared_ptr<NumericalKinematics> kinematics_;
  bool visuals_loaded_{false};
};

}  // namespace holistic_motion::robotics
