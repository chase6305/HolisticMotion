#pragma once
#include "holistic_motion/kinematics/utility.h"

namespace holistic_motion {
namespace robotics {
// TODO:
// 1.may add numerical Solution to base, etc
// 2.add some utility functions
class KinematicsBase : public std::enable_shared_from_this<KinematicsBase> {
public:
    virtual ~KinematicsBase() {
        holistic_motion::utility::LogDebug("Destructing KinematicsBase...");
    };

    /// \brief Get the forward kinematics for each joint
    ///
    /// \param target_joint
    /// \param pose_list: the forward kinematics for each joint
    /// \return bool
    virtual bool GetAllFK(const Eigen::VectorXd& target_joint,
                          std::vector<SE3d>& pose_list) const;

    /// \brief Get the forward kinematics
    ///
    /// \param target_joint
    /// \param pose: the forward kinematics
    /// \return bool
    virtual bool GetFK(const Eigen::VectorXd& target_joint,
                       SE3d& pose) const = 0;

    /// \brief Determine whether the robot joint corresponding to the target TCP
    /// pose is reachable
    ///
    /// \param target_tcp_pose
    /// \return bool
    virtual bool IsReachable(const SE3d& target_tcp_pose) const;

    /// \brief Get the inverse kinematics
    ///
    /// \param target_pose: target TCP pose
    /// \param ik_solutions: the inverse kinematics
    /// \param joint_seed: most recently solved joint
    /// \param dist: the distance of each joint from joint_seed
    /// \return bool
    virtual bool GetIK(const SE3d& target_pose,
                       IkRtn& ik_solutions,
                       Eigen::VectorXd& joint_seed,
                       std::vector<double>& dist) const = 0;

    /// \brief Get the nearst inverse kinematics
    ///
    /// \param target_pose: target TCP pose
    /// \param ik_solutions: the nearst inverse kinematics
    /// \param joint_seed: most recently solved joint
    /// \param dist: the distance from joint_seed
    /// \return bool
    virtual bool GetNearstIK(const SE3d& target_pose,
                             IkRtn& ik_solutions,
                             Eigen::VectorXd& joint_seed,
                             double& min_dist) const = 0;

    /// Correctly spelled compatibility-neutral alias for GetNearstIK().
    bool GetNearestIK(const SE3d& target_pose,
                      IkRtn& ik_solutions,
                      Eigen::VectorXd& joint_seed,
                      double& min_dist) const {
        return GetNearstIK(target_pose, ik_solutions, joint_seed, min_dist);
    }

    // TODO: TCP needs to consider where the only subsequent maintenance should
    // be placed
    virtual bool SetTCP(const SE3d& pose);

    SE3d GetTCP() const { return this->tcp_; };

    void ClearTCP();

    /// \brief Set the user frame
    ///
    virtual bool SetUserFrame(const SE3d& pose);

    /// \brief Get the user frame
    ///
    SE3d GetUserFrame() const { return this->userframe_; };

    void ClearUserFrame();

    /// Apply the configured user-frame transform to a base-frame pose.
    SE3d ApplyUserFrame(const SE3d& base_pose) const;

    /// Remove the configured user-frame transform from a user-frame pose.
    SE3d RemoveUserFrame(const SE3d& user_pose) const;

    /// \brief Set the kinematic parameters of the joint
    ///
    /// \param joint_node: the kinematic parameters of the joint
    /// \return bool
    bool SetJointNode(const std::vector<JointNode>& joint_node) {
        this->joint_nodes_ = joint_node;
        return true;
    };

    /// \brief Get the kinematic parameters of the joint
    ///
    /// \return std::vector<JointNode>
    std::vector<JointNode> GetJointNode() const { return this->joint_nodes_; };

    /// \brief Set the home joints to solve IK
    ///
    /// \param joints: the home joints
    /// \return bool
    bool SetHomeJoints(const Eigen::VectorXd& joints) {
        if (joints.size() != this->dof_ || !joints.allFinite()) return false;
        this->home_joints_ = joints;
        return true;
    };

    /// \brief Get the home joints to solve IK
    ///
    /// \return Eigen::VectorXd
    Eigen::VectorXd GetHomeJoints() const { return this->home_joints_; };

    /// \brief set ik nearst weight
    ///
    /// \param weight
    /// \return bool
    bool SetIkNearstWeight(const Eigen::VectorXd& weight);

    /// \brief get ik nearst weight
    ///
    /// \param weight
    /// \return bool
    bool GetIkNearstWeight(Eigen::VectorXd& weight) const;

    /// \brief get degrees of freedom
    int GetDOF() const { return this->dof_; };

    /// \brief Sets the joint limits for the robot's joints.
    ///
    /// \param upper_limits A vector containing the upper limits for each joint.
    /// \param lower_limits A vector containing the lower limits for each joint.
    /// \return bool
    bool SetJointLimits(const Eigen::VectorXd& upper_limits,
                        const Eigen::VectorXd& lower_limits);

    /// \brief Gets the joint limits for the robot's joints.
    ///
    /// \param upper_limits A vector containing the upper limits for each joint.
    /// \param lower_limits A vector containing the lower limits for each joint.
    /// \return bool
    bool GetJointLimits(Eigen::VectorXd& upper_limits,
                        Eigen::VectorXd& lower_limits) const;

    /// \brief Sets the user-defined joint limits for the robot's joints.
    ///
    /// \param upper_limits A vector containing the upper limits for each joint.
    /// \param lower_limits A vector containing the lower limits for each joint.
    /// \return bool
    bool SetUserJointLimits(const Eigen::VectorXd& upper_limits,
                            const Eigen::VectorXd& lower_limits);

    /// \brief Gets the user-defined joint limits for the robot's joints.
    ///
    /// \param upper_limits A vector containing the upper limits for each joint.
    /// \param lower_limits A vector containing the lower limits for each joint.
    /// \return bool
    bool GetUserJointLimits(Eigen::VectorXd& upper_limits,
                            Eigen::VectorXd& lower_limits) const;

    /// \brief Set the base orientation to front or back.
    ///
    /// \param enable Set to true to face front, false to face back.
    void SetBaseFront(bool enable) { this->robot_config_.SetBaseFront(enable); }

    /// \brief Set the elbow orientation to up or down.
    ///
    /// \param enable Set to true to face up, false to face down.
    void SetElbowUp(bool enable) { this->robot_config_.SetElbowUp(enable); }

    /// \brief Set the wrist orientation to flip or not flip.
    ///
    /// \param enable Set to true to flip, false to not flip.
    void SetWristFlip(bool enable) { this->robot_config_.SetWristFlip(enable); }

    /// \brief Get the base orientation.
    /// \return True if facing front, false if facing back.
    bool IsBaseFront() const { return this->robot_config_.IsBaseFront(); }

    /// \brief Get the elbow orientation.
    /// \return True if facing up, false if facing down.
    bool IsElbowUp() const { return this->robot_config_.IsElbowUp(); }

    /// \brief Get the wrist orientation.
    /// \return True if flipped, false if not flipped.
    bool IsWristFlip() const { return this->robot_config_.IsWristFlip(); }

    /// \brief Get disable or not.
    /// \return True if Disable, false if valid.
    bool IsDisableRobotConfig() const {
        return this->robot_config_.IsDisable();
    }

    /// \brief Disables all configurations.
    void DisableRobotConfig() { return this->robot_config_.Disable(); }

    /// \brief Set filter mode for a single joint
    ///
    /// \param joint_index Index of the joint
    /// \param mode Filter mode to set
    void SetJointFilter(size_t joint_index,
                        JointFilterManager::FilterMode mode) {
        this->joint_filter_config_.SetJointFilter(joint_index, mode);
    }

    /// \brief Set same filter mode for all joints
    ///
    /// \param mode Filter mode to set for all joints
    void SetAllJointFilters(JointFilterManager::FilterMode mode) {
        this->joint_filter_config_.SetAllJointFilters(mode);
    }

    /// \brief Set filter modes for multiple specified joints
    ///
    /// \param settings Vector of joint index and mode pairs
    void SetMultipleJointFilters(
            const std::vector<
                    std::pair<size_t, JointFilterManager::FilterMode>>&
                    settings) {
        this->joint_filter_config_.SetMultipleJointFilters(settings);
    }

    /// \brief Reset all joint filters to NONE mode
    void ResetJointFilters() { this->joint_filter_config_.Reset(); }

    /// \brief Get filter mode of specified joint
    ///
    /// \param joint_index Index of the joint
    /// \return FilterMode of the specified joint
    JointFilterManager::FilterMode GetJointFilter(size_t joint_index) const {
        return this->joint_filter_config_.GetJointFilter(joint_index);
    }

    /// \brief Set angle thresholds for all joints using a vector
    ///
    /// \param thresholds Vector containing threshold values for all joints
    /// \return true if successful, false if vector size doesn't match
    bool SetJointAngleThresholds(const std::vector<double>& thresholds) {
        return this->joint_filter_config_.SetJointAngleThresholds(thresholds);
    }

    /// \brief Set angle threshold for a specific joint
    ///
    /// \param joint_index Index of the joint
    /// \param threshold Threshold value for the joint
    /// \return true if successful, false if index out of range
    bool SetJointAngleThreshold(size_t joint_index, double threshold) {
        return this->joint_filter_config_.SetJointAngleThreshold(joint_index,
                                                                 threshold);
    }

    /// \brief Get angle threshold for a specific joint
    ///
    /// \param joint_index Index of the joint
    /// \return Threshold value for the specified joint, returns 0.0 if index is
    /// out of range
    double GetJointAngleThreshold(size_t joint_index) const {
        return this->joint_filter_config_.GetJointAngleThreshold(joint_index);
    }

    /// \brief Get current angle thresholds for all joints
    ///
    /// \return Vector of current threshold values
    const std::vector<double>& GetJointAngleThresholds() const {
        return this->joint_filter_config_.GetJointAngleThresholds();
    }

    /// \brief Check if joint angle is valid according to filter mode
    ///
    /// \param joint_index Index of the joint to check
    /// \param angle Angle value to validate
    /// \return true if angle is valid, false otherwise
    bool IsValidJointAngle(size_t joint_index, double angle) const {
        return this->joint_filter_config_.IsValidJointAngle(joint_index, angle);
    }

protected:
    KinematicsBase() = default;

    bool initalize_ = false;  ///< Whether it has been initialized successfully
    int dof_;  ///< The number of controllable joints of the robot
    SE3d tcp_ = SE3d(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0);  ///< Transformation matrix from "tool end" to "flange end"
    SE3d userframe_ =
            SE3d(0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 1.0);  ///< Transformation matrix from "base" to "user frame"
    Eigen::VectorXd ik_nearst_weight_;    ///< Delta weight for each joint
    Eigen::VectorXd home_joints_;         ///< home joints in urdf
    std::vector<JointNode> joint_nodes_;  ///< joint upper and lower limits
    std::vector<IkRtn> joint_solutions_;  ///< joint solutions

    const double Epsilon = std::numeric_limits<double>::epsilon();

    RobotConfigManager robot_config_;
    JointFilterManager joint_filter_config_;
};

}  // namespace robotics
}  // namespace holistic_motion
