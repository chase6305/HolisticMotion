#include "holistic_motion/kinematics/NumericalKinematics.h"

#include "DampedIKWorkspace.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace holistic_motion {
namespace robotics {

NumericalKinematics::NumericalKinematics(
        const std::vector<JointNode>& joint_node) {
    if (joint_node.empty() ||
        joint_node.size() - 1 >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
                "numerical joint model must contain a terminal node and "
                "a representable number of coordinates");
    }
    InitializeJointModel(joint_node, static_cast<int>(joint_node.size() - 1));
    holistic_motion::utility::LogDebug("Constructing NumericalKinematics...");
}

void NumericalKinematics::SetDOF(int dof) {
    if (dof != this->dof_) {
        throw std::invalid_argument("DOF is fixed by the numerical joint model");
    }
}

bool NumericalKinematics::GetFK(const Eigen::VectorXd& target_joint,
                                SE3d& pose) const {
    std::vector<SE3d> pose_list;
    if (this->GetAllFK(target_joint, pose_list) && !pose_list.empty()) {
        pose = pose_list.back();
        return true;
    }
    return false;
}

bool NumericalKinematics::GetIK(const SE3d& target_pose,
                                IkRtn& ik_solutions,
                                Eigen::VectorXd& joint_seed,
                                std::vector<double>& dist) const {
    holistic_motion::utility::LogDebug("GetNearstIK:Begin to get position IK.");
    ik_solutions.Clear();
    dist.clear();

    if (joint_seed.size() != this->dof_) {
        joint_seed = this->home_joints_;
    }
    if (!joint_seed.allFinite() ||
        !target_pose.GetTransform().allFinite()) {
        ik_solutions.success = false;
        return false;
    }

    SE3d tcp_pose;
    std::vector<SE3d> pose_list;
    if (this->GetAllFK(joint_seed, pose_list) &&
        pose_list.size() > static_cast<std::size_t>(this->dof_)) {
        tcp_pose = pose_list.back();
    } else {
        ik_solutions.success = false;
        return false;
    }

    Eigen::VectorXd iter_joint = joint_seed;
    const auto task_error = [&target_pose](const SE3d& current) {
        // The geometric Jacobian maps to TCP linear velocity and angular
        // velocity in the world frame. A relative SE3 translation also rotates
        // the current position and is not the corresponding position error.
        const Eigen::Matrix3d rotation =
                target_pose.GetRotation() * current.GetRotation().transpose();
        return SE3d(target_pose.GetTranslation() - current.GetTranslation(),
                    SO3d(rotation));
    };
    SE3d delta_x = task_error(tcp_pose);
    double angle_err = std::abs(GetRAngle(delta_x.GetRotation()));
    double translation_err = delta_x.GetTranslation().norm();
    bool converged = translation_err < this->trans_err_th_ &&
                     angle_err < this->angle_err_th_;
    // Exact seeds avoid allocating iteration storage altogether.
    if (!converged && this->dof_ > 0) {
        Eigen::MatrixXd jacobian(6, this->dof_);
        Eigen::VectorXd delta_theta(this->dof_);
        detail::DampedIKWorkspace workspace(this->dof_);
        for (int i = 0; i < this->max_iter_num_; ++i) {
            if (!this->_GetKinJacobian(jacobian, pose_list)) break;
            Eigen::Matrix<double, 6, 1> error;
            error.head<3>() = delta_x.GetTranslation();
            const Eigen::AngleAxisd rotation_error(
                    Eigen::Quaterniond(delta_x.GetRotation()));
            error.tail<3>() = rotation_error.angle() * rotation_error.axis();
            if (!workspace.Solve(jacobian, error, this->damp_coeff_, this->eps_,
                                 delta_theta)) break;
            iter_joint += this->step_size_ * delta_theta;
            for (int joint = 0; joint < this->dof_; ++joint) {
                const auto& node = this->joint_nodes_[joint];
                iter_joint[joint] = clamp(iter_joint[joint], node.lower_limit,
                                         node.upper_limit);
            }
            if (!this->GetAllFK(iter_joint, pose_list) ||
                pose_list.size() <= static_cast<std::size_t>(this->dof_)) break;
            tcp_pose = pose_list.back();

            delta_x = task_error(tcp_pose);
            angle_err = std::abs(GetRAngle(delta_x.GetRotation()));
            translation_err = delta_x.GetTranslation().norm();

            if (translation_err < this->trans_err_th_ &&
                angle_err < this->angle_err_th_) {
                converged = true;
                break;
            }
        }
    }

    if (!converged) {
        ik_solutions.success = false;
    } else {
        ik_solutions.success = ik_solutions.PushBack(iter_joint);
    }

    // Remove repeated IK, if there are.
    ik_solutions.RemoveRepeatedIK();
    // Finds IK that is within joint limits, removes IK that is out of range.
    ik_solutions.GetLimitsIK(this->joint_nodes_);

    for (int i = 0; i < ik_solutions.ik_number; ++i) {
        double temp_dist = 0.0;
        for (int j = 0; j < this->dof_; ++j) {
            temp_dist +=
                    std::abs((ik_solutions.ik_joints[i][j] - joint_seed[j]) *
                             this->ik_nearst_weight_[j]);
        }
        dist.push_back(temp_dist);
    }

    if (ik_solutions.success && ik_solutions.ik_number > 0) {
        holistic_motion::utility::LogDebug(
                "GetIK: Success to get position IK, ik number:[{}].",
                ik_solutions.ik_number);
        return true;
    }

    holistic_motion::utility::LogDebug(
            "GetIK: Failed to get position IK, success:[{}], ik_number:[{}]",
            ik_solutions.success, ik_solutions.ik_number);

    return false;
}

bool NumericalKinematics::GetNearstIK(const SE3d& target_pose,
                                      IkRtn& ik_solutions,
                                      Eigen::VectorXd& joint_seed,
                                      double& min_dist) const {
    holistic_motion::utility::LogDebug("GetNearstIK:Begin to get nearst IK.");

    if (joint_seed.size() != this->dof_) {
        holistic_motion::utility::LogWarning("Joint seed size[{}] not match dof[{}].",
                                  joint_seed.size(), this->dof_);
        joint_seed = this->home_joints_;
    }

    std::vector<double> dist;
    if (this->GetIK(target_pose, ik_solutions, joint_seed, dist)) {
        // ik_solutions
        if (ik_solutions.success) {
            min_dist = dist.front();
            holistic_motion::utility::LogDebug("GetNearstIK: Success to get nearst IK.");
            return true;
        }
    }

    ik_solutions.Clear();
    Eigen::VectorXd zero_joint_seed = Eigen::VectorXd::Zero(this->dof_);
    if (this->GetIK(target_pose, ik_solutions, zero_joint_seed, dist)) {
        // ik_solutions
        if (ik_solutions.success) {
            min_dist = dist.front();
            holistic_motion::utility::LogDebug("GetNearstIK: Success to get nearst IK.");
            return true;
        }
    }

    holistic_motion::utility::LogDebug("GetNearstIK: Failed to get nearst IK.");
    return false;
}

bool NumericalKinematics::SetMaxIterNum(const int& max_iter_num) {
    if (max_iter_num <= 0) {
        return false;
    }
    this->max_iter_num_ = max_iter_num;
    return true;
}

bool NumericalKinematics::GetMaxIterNum(int& max_iter_num) const {
    max_iter_num = this->max_iter_num_;
    return true;
}

bool NumericalKinematics::SetTransErrTh(const double& trans_err_th) {
    if (!std::isfinite(trans_err_th) || trans_err_th <= 0) {
        return false;
    }
    this->trans_err_th_ = trans_err_th;
    return true;
}

bool NumericalKinematics::GetTransErrTh(double& trans_err_th) const {
    trans_err_th = this->trans_err_th_;
    return true;
}

bool NumericalKinematics::SetAngleErrTh(const double& angle_err_th) {
    if (!std::isfinite(angle_err_th) || angle_err_th <= 0) {
        return false;
    }
    this->angle_err_th_ = angle_err_th;
    return true;
}

bool NumericalKinematics::GetAngleErrTh(double& angle_err_th) const {
    angle_err_th = this->angle_err_th_;
    return true;
}

bool NumericalKinematics::SetStepSize(const double& step_size) {
    if (!std::isfinite(step_size) || step_size <= 0) {
        return false;
    }
    this->step_size_ = step_size;
    return true;
}

bool NumericalKinematics::GetStepSize(double& step_size) const {
    step_size = this->step_size_;
    return true;
}

bool NumericalKinematics::SetDamp(const double& damp) {
    if (!std::isfinite(damp) || damp <= 0) {
        return false;
    }
    this->damp_coeff_ = damp;
    return true;
}

bool NumericalKinematics::GetDamp(double& damp) const {
    damp = this->damp_coeff_;
    return true;
}

bool NumericalKinematics::GetJacobian(const Eigen::VectorXd& joint_pos,
                                      Eigen::MatrixXd& jacobian) const {
    // if (joint_pos.size() != this->dof_) {
    //     holistic_motion::utility::LogWarning("Joint position size[{}] not match
    //     dof[{}].",
    //                               joint_pos.size(), this->dof_);
    //     return false;
    // }
    std::vector<SE3d> pose_list;
    if (!this->GetAllFK(joint_pos, pose_list)) {
        return false;
    }
    return this->_GetKinJacobian(jacobian, pose_list);
}

bool NumericalKinematics::_GetKinJacobian(
        Eigen::MatrixXd& jacobian, const std::vector<SE3d>& pose_list) const {
    if (pose_list.size() <= static_cast<std::size_t>(this->dof_)) return false;
    jacobian.resize(6, this->dof_);
    const Eigen::Vector3d ee_position = pose_list.back().GetTranslation();

    Eigen::Matrix<double, 6, 1> twist;
    for (int i = 0; i < this->dof_; ++i) {
        twist.setZero();
        auto joint_type = this->joint_nodes_[i].joint_type;

        Eigen::Vector3d joint_start_position = pose_list[i].GetTranslation();
        const Eigen::Vector3d joint_axis =
                pose_list[i].GetRotation() * this->joint_nodes_[i].axis;

        if (JointType::REVOLUTE == joint_type ||
            JointType::CONTINUOUS == joint_type) {
            twist.head<3>() =
                    joint_axis.cross(ee_position - joint_start_position);
            twist.tail<3>() = joint_axis;
        } else if (JointType::PRISMATIC == joint_type) {
            twist.head<3>() = joint_axis;
        }
        jacobian.col(i) = twist;
    }

    return true;
}

}  // namespace robotics
}  // namespace holistic_motion
