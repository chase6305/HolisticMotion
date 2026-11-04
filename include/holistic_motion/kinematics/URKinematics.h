#pragma once
#include "holistic_motion/kinematics/KinematicsBase.h"

namespace holistic_motion {
namespace robotics {

/// \brief Variant in OPW Kinematics,
/// https://github.com/pyni/ur_inverse_solutions
class URKinematics : public KinematicsBase {
public:
  KinematicsCapabilities GetCapabilities() const override;

  /// Require at least six valid coordinate nodes; any remaining nodes must
  /// be FIXED or UNKNOWN. Invalid geometry, offsets, directions, transforms,
  /// or joint models throw std::invalid_argument.
  explicit URKinematics(URParameters urp, const std::vector<JointNode> &joint_node);
  virtual ~URKinematics() = default;

  /// \brief get the forward kinematics
  ///
  /// \param target_joint
  /// \param pose
  /// \return bool success or not
  virtual bool GetFK(const Eigen::VectorXd &target_joint,
                     SE3d &pose) const override final;

  /// \brief get all inverse kinematics
  ///
  /// \param target_pose
  /// \param ik_solutions
  /// \param joint_seed
  /// \param dist Per-solution weighted coordinate movement from joint_seed
  /// \return bool success or not
  virtual bool GetIK(const SE3d &target_pose, IkRtn &ik_solutions,
                     Eigen::VectorXd &joint_seed,
                     std::vector<double> &dist) const override final;


private:
    URParameters params_;  ///< Contains 6 geometric parameters and
                           ///< transformation relations(UR)
};

}  // namespace robotics
}  // namespace holistic_motion
