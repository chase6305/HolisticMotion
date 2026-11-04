#pragma once

#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/robot/RobotModel.h"

namespace holistic_motion::robotics {

/// Compatibility entry point combining a model with a default chain solver.
/// Loading chooses the longest supported actuated chain as before. Use
/// RobotModel directly when only tree, coordinate or visual data is needed.
/// Copies share Joint/Link objects and the default solver. Synchronize loading
/// and public model/solver edits with readers.
class Robot {
  public:
    Robot() = default;
    explicit Robot(const std::string &urdf_path, bool load_visuals = false);
    static std::shared_ptr<Robot> Create(const std::string &urdf_path,
                                         bool load_visuals = false);

    /// Failed model parsing or solver construction preserves the old state.
    bool LoadURDF(const std::string &urdf_path, bool load_visuals = false);
    bool LoadVisuals();
    bool HasVisualsLoaded() const noexcept { return model_.HasVisualsLoaded(); }
    bool HasKinematics() const noexcept {
        return static_cast<bool>(kinematics_);
    }
    const RobotModel &GetModel() const noexcept { return model_; }
    const std::string &GetName() const noexcept { return model_.GetName(); }
    const std::string &GetURDFPath() const noexcept {
        return model_.GetURDFPath();
    }
    const std::string &GetRootLinkName() const noexcept {
        return model_.GetRootLinkName();
    }
    int GetDoF() const noexcept {
        return static_cast<int>(actuated_joints_.size());
    }
    int GetModelDoF() const noexcept { return model_.GetModelDoF(); }
    const std::vector<std::shared_ptr<Joint>> &GetJoints() const noexcept {
        return model_.GetJoints();
    }
    const std::vector<std::shared_ptr<Joint>> &
    GetActuatedJoints() const noexcept {
        return actuated_joints_;
    }
    const std::vector<std::shared_ptr<Joint>> &
    GetAllActuatedJoints() const noexcept {
        return model_.GetAllActuatedJoints();
    }
    const std::vector<std::shared_ptr<Link>> &GetLinks() const noexcept {
        return model_.GetLinks();
    }
    std::shared_ptr<Link> GetLink(const std::string &name) const noexcept;
    std::shared_ptr<Joint> GetJoint(const std::string &name) const noexcept;
    std::shared_ptr<NumericalKinematics> GetKinematics() const noexcept {
        return kinematics_;
    }
    /// Missing, unsupported, cyclic and fixed-only chains return nullptr.
    std::shared_ptr<NumericalKinematics>
    CreateKinematics(const std::string &base_link,
                     const std::string &tip_link) const;
    std::shared_ptr<NumericalKinematics>
    CreateFEPKinematics(const std::string &base_link,
                        const std::string &tip_link) const;
    std::vector<JointCoordinate> GetJointCoordinates() const;
    JointPositionMap
    CreateJointPositionMap(const std::vector<std::string> &joint_names) const;
    /// Active names for the same supported path as CreateKinematics().
    std::vector<std::string>
    GetChainJointNames(const std::string &base_link,
                       const std::string &tip_link) const;

  private:
    RobotModel model_;
    std::vector<std::shared_ptr<Joint>> actuated_joints_;
    std::shared_ptr<NumericalKinematics> kinematics_;
};

} // namespace holistic_motion::robotics
