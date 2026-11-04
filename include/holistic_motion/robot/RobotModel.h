#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holistic_motion/robot/Joint.h"
#include "holistic_motion/robot/JointPositionMap.h"
#include "holistic_motion/robot/Link.h"

namespace holistic_motion::robotics {

/// Parsed robot tree and scalar model coordinates, independent of IK solvers.
/// Copies share Joint/Link objects. Synchronize loading, visual publication and
/// public Joint/Link edits with readers; use independent models for mutation.
class RobotModel {
  public:
    RobotModel() = default;
    explicit RobotModel(const std::string &urdf_path,
                        bool load_visuals = false);

    /// Import unit scalar axes. Failure preserves the previously loaded model.
    bool LoadURDF(const std::string &urdf_path, bool load_visuals = false);
    /// Publish metadata from the original parse without reopening the URDF.
    bool LoadVisuals();
    bool HasVisualsLoaded() const noexcept { return visuals_loaded_; }
    const std::string &GetName() const noexcept { return name_; }
    const std::string &GetURDFPath() const noexcept { return urdf_path_; }
    const std::string &GetRootLinkName() const noexcept {
        return root_link_name_;
    }
    /// Independent scalar joints, excluding mimic, floating and planar joints.
    int GetModelDoF() const noexcept {
        return static_cast<int>(all_actuated_joints_.size());
    }
    const std::vector<std::shared_ptr<Joint>> &GetJoints() const noexcept {
        return joints_;
    }
    const std::vector<std::shared_ptr<Joint>> &
    GetAllActuatedJoints() const noexcept {
        return all_actuated_joints_;
    }
    const std::vector<std::shared_ptr<Link>> &GetLinks() const noexcept {
        return links_;
    }
    std::shared_ptr<Link> GetLink(const std::string &name) const noexcept;
    std::shared_ptr<Joint> GetJoint(const std::string &name) const noexcept;
    /// All joints on the tree path, including fixed, mimic and nonscalar
    /// joints. Missing or cyclic paths return an empty vector. Returned objects
    /// stay alive across reloads; their public fields retain shared ownership.
    std::vector<std::shared_ptr<Joint>>
    GetChainJoints(const std::string &base_link,
                   const std::string &tip_link) const;
    std::vector<JointCoordinate> GetJointCoordinates() const;
    JointPositionMap
    CreateJointPositionMap(const std::vector<std::string> &joint_names) const;

  private:
    std::string name_;
    std::string urdf_path_;
    std::string root_link_name_;
    std::vector<std::shared_ptr<Joint>> joints_;
    std::vector<std::shared_ptr<Joint>> all_actuated_joints_;
    std::vector<std::shared_ptr<Link>> links_;
    std::vector<std::pair<std::size_t, std::vector<VisualGeometry>>>
        pending_visuals_;
    bool visuals_loaded_{false};
};

} // namespace holistic_motion::robotics
