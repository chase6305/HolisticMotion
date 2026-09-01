#include "holistic_motion/kinematics/KinematicsBase.h"
#include <algorithm>
#include <cmath>

namespace holistic_motion {
namespace robotics {

namespace {

bool ValidLimitInterval(double lower, double upper) {
    return !std::isnan(lower) && !std::isnan(upper) && lower <= upper;
}

}  // namespace

bool KinematicsBase::GetAllFK(const Eigen::VectorXd& target_joint,
                              std::vector<SE3d>& pose_list) const {
    pose_list.clear();
    int num = this->joint_nodes_.size();
    pose_list.reserve(num + 1);
    if (target_joint.size() != this->dof_ || !target_joint.allFinite()) {
        holistic_motion::utility::LogWarning(
                "Joint vector must contain [{}] finite values; got size [{}]",
                this->dof_, target_joint.size());
        return false;
    }
    SE3d pre_pose = SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < num; i++) {
        SE3d transform = SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        Eigen::Vector3d xyz(0.0, 0.0, 0.0);
        SO3d rot = SO3d(0.0, 0.0, 0.0, 1.0);
        if (JointType::REVOLUTE == this->joint_nodes_[i].joint_type) {
            // generate rotation vector
            Eigen::Vector3d rota_vec =
                    target_joint[i] * this->joint_nodes_[i].axis;
            rot = SO3d(rota_vec);
        }
        if (JointType::PRISMATIC == this->joint_nodes_[i].joint_type) {
            // generate translation vector
            xyz = this->joint_nodes_[i].axis * target_joint[i];
        }
        transform = SE3d(xyz, rot);
        pose_list.push_back(pre_pose * this->joint_nodes_[i].origin_pose *
                            transform);
        pre_pose = pose_list.back();
        // holistic_motion::utility::LogDebug("Joint{} pose:[SE3({})].", i + 1,
        //                         fmt::join(pre_pose.Coeffs(), ","));
    }
    // update tcp pose
    pose_list.push_back(pose_list.back() * this->GetTCP());

    return true;
}

bool KinematicsBase::SetTCP(const SE3d& pose) {
    this->tcp_ = pose;
    return true;
}

void KinematicsBase::ClearTCP() {
    this->tcp_ = SE3d();
}

bool KinematicsBase::SetUserFrame(const SE3d& pose) {
    this->userframe_ = pose;
    return true;
}

void KinematicsBase::ClearUserFrame() {
    this->userframe_ = SE3d();
}

SE3d KinematicsBase::ApplyUserFrame(const SE3d& base_pose) const {
    return this->userframe_ * base_pose;
}

SE3d KinematicsBase::RemoveUserFrame(const SE3d& user_pose) const {
    return this->userframe_.Inverse() * user_pose;
}

bool KinematicsBase::IsReachable(const SE3d& target_tcp_pose) const {
    Eigen::VectorXd joint_seed = this->home_joints_;
    IkRtn rtn;
    double distance = 0.0;
    return this->GetNearestIK(target_tcp_pose, rtn, joint_seed, distance);
}

bool KinematicsBase::SetIkNearstWeight(const Eigen::VectorXd& weight) {
    if (weight.size() != this->dof_ || !weight.allFinite() ||
        (weight.array() < 0.0).any() || !(weight.array() > 0.0).any()) {
        holistic_motion::utility::LogWarning("weight size must be [{}], instead of {}",
                                  this->dof_, weight.size());
        return false;
    }
    this->ik_nearst_weight_ = weight;
    return true;
}

bool KinematicsBase::GetIkNearstWeight(Eigen::VectorXd& weight) const {
    if (this->ik_nearst_weight_.size() != this->dof_) {
        holistic_motion::utility::LogWarning("weight size[{}] is not match dof[{}]",
                                  this->ik_nearst_weight_.size(), this->dof_);
        return false;
    }
    weight = this->ik_nearst_weight_;
    return true;
}

bool KinematicsBase::SetJointLimits(const Eigen::VectorXd& upper_limits,
                                    const Eigen::VectorXd& lower_limits) {
    if (upper_limits.size() != this->dof_ ||
        lower_limits.size() != this->dof_) {
        return false;  // Ensure the limits match the degrees of freedom
    }

    for (int i = 0; i < this->dof_; ++i) {
        if (!ValidLimitInterval(lower_limits[i], upper_limits[i])) {
            return false;
        }
    }

    for (int i = 0; i < this->dof_; ++i) {
        this->joint_nodes_[i].upper_limit = upper_limits[i];
        this->joint_nodes_[i].lower_limit = lower_limits[i];
    }
    return true;
}

bool KinematicsBase::GetJointLimits(Eigen::VectorXd& upper_limits,
                                    Eigen::VectorXd& lower_limits) const {
    if (upper_limits.size() != this->dof_ ||
        lower_limits.size() != this->dof_) {
        upper_limits.resize(this->dof_);
        lower_limits.resize(this->dof_);
    }

    for (int i = 0; i < this->dof_; ++i) {
        upper_limits[i] = this->joint_nodes_[i].upper_limit;
        lower_limits[i] = this->joint_nodes_[i].lower_limit;
    }
    return true;
}

bool KinematicsBase::SetUserJointLimits(const Eigen::VectorXd& upper_limits,
                                        const Eigen::VectorXd& lower_limits) {
    if (upper_limits.size() != this->dof_ ||
        lower_limits.size() != this->dof_) {
        return false;  // Ensure the limits match the degrees of freedom
    }

    for (int i = 0; i < this->dof_; ++i) {
        if (!ValidLimitInterval(lower_limits[i], upper_limits[i]) ||
            std::max(lower_limits[i], this->joint_nodes_[i].lower_limit) >
                    std::min(upper_limits[i],
                             this->joint_nodes_[i].upper_limit)) {
            return false;
        }
    }

    for (int i = 0; i < this->dof_; ++i) {
        this->joint_nodes_[i].upper_limit_set = upper_limits[i];
        this->joint_nodes_[i].lower_limit_set = lower_limits[i];
    }
    return true;
}

bool KinematicsBase::GetUserJointLimits(Eigen::VectorXd& upper_limits,
                                        Eigen::VectorXd& lower_limits) const {
    if (upper_limits.size() != this->dof_ ||
        lower_limits.size() != this->dof_) {
        upper_limits.resize(this->dof_);
        lower_limits.resize(this->dof_);
    }

    for (int i = 0; i < this->dof_; ++i) {
        upper_limits[i] = this->joint_nodes_[i].upper_limit_set;
        lower_limits[i] = this->joint_nodes_[i].lower_limit_set;
    }
    return true;
}

}  // namespace robotics
}  // namespace holistic_motion
