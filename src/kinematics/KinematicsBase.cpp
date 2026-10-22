#include "holistic_motion/kinematics/KinematicsBase.h"

#include "FrameValidation.h"
#include "ForwardKinematics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace holistic_motion {
namespace robotics {

namespace {

bool ValidLimitInterval(double lower, double upper) {
    return !std::isnan(lower) && !std::isnan(upper) && lower <= upper;
}

bool ValidJointNode(const JointNode& node) {
    if (!detail::IsValidFrame(node.origin_pose) || !node.axis.allFinite() ||
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

KinematicsCapabilities KinematicsBase::GetCapabilities() const {
    KinematicsCapabilities result;
    result.dof = GetDOF();
    return result;
}

IKSolveReport KinematicsBase::SolveIK(const SE3d &target, const Eigen::VectorXd &seed,
                                      IKSelection selection) const {
    IKSolveReport report;
    report.algorithm = GetCapabilities().default_ik_algorithm;
    report.selection = selection;
    if (!detail::IsValidFrame(target) || seed.size() != GetDOF() || !seed.allFinite() ||
        (selection != IKSelection::NEAREST && selection != IKSelection::ALL)) {
        report.status = IKSolveStatus::INVALID_INPUT;
        return report;
    }

    Eigen::VectorXd working_seed = seed;
    IkRtn solutions;
    std::vector<double> distances;
    if (selection == IKSelection::NEAREST) {
        double distance = 0.0;
        if (!GetNearestIK(target, solutions, working_seed, distance) ||
            solutions.ik_joints.size() != 1)
            return report;
        distances.push_back(distance);
    } else if (!GetIK(target, solutions, working_seed, distances)) {
        return report;
    }
    if (solutions.ik_joints.empty() || distances.size() != solutions.ik_joints.size() ||
        !std::all_of(distances.begin(), distances.end(),
                     [](double distance) {
                         return std::isfinite(distance) && distance >= 0.0;
                     }) ||
        !std::all_of(solutions.ik_joints.begin(), solutions.ik_joints.end(),
                     [this](const Eigen::VectorXd &joints) {
                         return joints.size() == GetDOF() && joints.allFinite();
                     }))
        return report;
    report.solutions = std::move(solutions.ik_joints);
    report.distances = std::move(distances);
    report.status = IKSolveStatus::SUCCESS;
    return report;
}

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

bool KinematicsBase::GetAllFK(const Eigen::VectorXd &joints,
                              std::vector<SE3d> &poses) const {
    return detail::ComputeAllFK(*this, joints, poses);
}

bool KinematicsBase::SetTCP(const SE3d& pose) {
    if (!detail::IsValidFrame(pose))
        return false;
    this->tcp_ = pose;
    return true;
}

void KinematicsBase::ClearTCP() {
    this->tcp_ = SE3d();
}

bool KinematicsBase::SetUserFrame(const SE3d& pose) {
    if (!detail::IsValidFrame(pose))
        return false;
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
