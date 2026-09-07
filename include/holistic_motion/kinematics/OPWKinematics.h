#pragma once
#include "holistic_motion/kinematics/KinematicsBase.h"

namespace holistic_motion {
namespace robotics {

/// \brief OPW available through papers: An Analytical Solution of the Inverse
/// Kinematics Problem of Industrial Serial Manipulators with an Ortho-parallel
/// Basis and a Spherical Wrist
class OPWKinematics : public KinematicsBase {
public:
    /// Require at least six valid coordinate nodes; any remaining nodes must
    /// be FIXED or UNKNOWN. Invalid joint models throw std::invalid_argument.
    explicit OPWKinematics(OPWParameters opwp,
                           const std::vector<JointNode> &joint_node)
        : params_(opwp) {
        InitializeJointModel(joint_node, 6);
        holistic_motion::utility::LogDebug("Constructing OPWKinematics...");
    };
    virtual ~OPWKinematics() {
        holistic_motion::utility::LogDebug("Destructing OPWKinematics...");
    };

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
    /// \param joints_seed
    /// \return bool success or not
    virtual bool GetIK(const SE3d &target_pose,
                       IkRtn &ik_solutions,
                       Eigen::VectorXd &joint_seed,
                       std::vector<double> &dist) const override final;

    /// \brief get the nearst inverse kinematics
    ///
    /// \param target_pose
    /// \param ik_solutions
    /// \param joints_seed
    /// \return bool success or not
    virtual bool GetNearstIK(const SE3d &target_pose,
                             IkRtn &ik_solutions,
                             Eigen::VectorXd &joint_seed,
                             double &min_dist) const override final;

private:
    OPWParameters params_;  ///< Contains 7 geometric parameters and
                            ///< transformation relations(OPW)
};

}  // namespace robotics
}  // namespace holistic_motion
