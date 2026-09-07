#include "holistic_motion/kinematics/KinematicsBase.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace holistic_motion {
namespace robotics {

namespace {

bool ValidLimitInterval(double lower, double upper) {
    return !std::isnan(lower) && !std::isnan(upper) && lower <= upper;
}

bool ValidFrame(const SE3d& pose) {
    return pose.Coeffs().allFinite() &&
           std::abs(pose.Coeffs().tail<4>().squaredNorm() - 1.0) <= 1e-9;
}

bool ValidJointNode(const JointNode& node) {
    if (!ValidFrame(node.origin_pose) || !node.axis.allFinite() ||
        !ValidLimitInterval(node.lower_limit, node.upper_limit) ||
        !ValidLimitInterval(node.lower_limit_set, node.upper_limit_set) ||
        std::max(node.lower_limit, node.lower_limit_set) >
                std::min(node.upper_limit, node.upper_limit_set)) {
        return false;
    }
    switch (node.joint_type) {
        case JointType::REVOLUTE:
        case JointType::PRISMATIC:
        case JointType::CONTINUOUS: {
            const double norm = node.axis.norm();
            return std::isfinite(norm) && norm > 1e-12;
        }
        case JointType::FIXED:
        case JointType::UNKNOWN:
            return true;
        default:
            return false;
    }
}

bool ValidJointModel(const std::vector<JointNode>& nodes, int dof) {
    if (dof < 0 || nodes.empty() ||
        nodes.size() < static_cast<std::size_t>(dof)) return false;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (!ValidJointNode(nodes[i])) return false;
        if (i >= static_cast<std::size_t>(dof) &&
            nodes[i].joint_type != JointType::FIXED &&
            nodes[i].joint_type != JointType::UNKNOWN) return false;
    }
    return true;
}

}  // namespace

void KinematicsBase::InitializeJointModel(
        const std::vector<JointNode>& joint_nodes, int dof) {
    if (!ValidJointModel(joint_nodes, dof)) {
        throw std::invalid_argument(
                "invalid joint model: require valid nodes and a coordinate "
                "for every actuated node");
    }
    joint_nodes_ = joint_nodes;
    dof_ = dof;
    home_joints_ = Eigen::VectorXd::Zero(dof_);
    ik_nearst_weight_ = Eigen::VectorXd::Ones(dof_);
    joint_filter_config_.Resize(dof_);
    initalize_ = true;
}

bool KinematicsBase::GetAllFK(const Eigen::VectorXd& target_joint,
                              std::vector<SE3d>& pose_list) const {
    pose_list.clear();
    const std::size_t num = this->joint_nodes_.size();
    if (num == 0 || this->dof_ < 0 ||
        static_cast<std::size_t>(this->dof_) > num ||
        target_joint.size() != this->dof_ || !target_joint.allFinite()) {
        holistic_motion::utility::LogWarning(
                "Invalid FK model or joint vector (DOF {}, nodes {}, joints {})",
                this->dof_, num, target_joint.size());
        return false;
    }
    pose_list.reserve(num + 1);
    SE3d pre_pose = SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    for (std::size_t i = 0; i < num; ++i) {
        const auto type = this->joint_nodes_[i].joint_type;
        if ((type != JointType::FIXED && type != JointType::UNKNOWN &&
             i >= static_cast<std::size_t>(this->dof_)) ||
            type == JointType::PLANAR || type == JointType::FLOATING) {
            pose_list.clear();
            return false;
        }
        SE3d transform = SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        Eigen::Vector3d xyz(0.0, 0.0, 0.0);
        SO3d rot = SO3d(0.0, 0.0, 0.0, 1.0);
        if (JointType::REVOLUTE == this->joint_nodes_[i].joint_type ||
            JointType::CONTINUOUS == this->joint_nodes_[i].joint_type) {
            // SO3d(Vector3d) interprets Euler angles, not a rotation vector.
            const Eigen::Vector3d rotation_vector =
                    target_joint[i] * this->joint_nodes_[i].axis;
            const double angle = rotation_vector.norm();
            if (!std::isfinite(angle)) {
                pose_list.clear();
                return false;
            }
            if (angle > 0.0) {
                rot = SO3d(Eigen::AngleAxisd(angle, rotation_vector / angle));
            }
        }
        if (JointType::PRISMATIC == this->joint_nodes_[i].joint_type) {
            // generate translation vector
            xyz = this->joint_nodes_[i].axis * target_joint[i];
            if (!xyz.allFinite()) {
                pose_list.clear();
                return false;
            }
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
    if (!pose_list.back().Coeffs().allFinite()) {
        pose_list.clear();
        return false;
    }
    return true;
}

bool KinematicsBase::SetTCP(const SE3d& pose) {
    if (!ValidFrame(pose)) return false;
    this->tcp_ = pose;
    return true;
}

void KinematicsBase::ClearTCP() {
    this->tcp_ = SE3d();
}

bool KinematicsBase::SetUserFrame(const SE3d& pose) {
    if (!ValidFrame(pose)) return false;
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

bool KinematicsBase::SetJointNode(
        const std::vector<JointNode>& joint_node) {
    if (joint_node.size() != this->joint_nodes_.size() ||
        !ValidJointModel(joint_node, this->dof_)) {
        return false;
    }
    this->joint_nodes_ = joint_node;
    OnKinematicModelChanged();
    return true;
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
        if (!ValidLimitInterval(lower_limits[i], upper_limits[i]) ||
            std::max(lower_limits[i], this->joint_nodes_[i].lower_limit_set) >
                std::min(upper_limits[i], this->joint_nodes_[i].upper_limit_set)) {
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
