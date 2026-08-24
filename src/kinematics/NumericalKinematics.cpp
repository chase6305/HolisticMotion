#include "holistic_motion/kinematics/NumericalKinematics.h"

namespace holistic_motion {
namespace robotics {

bool NumericalKinematics::GetFK(const Eigen::VectorXd& target_joint,
                                SE3d& pose) const {
    std::vector<SE3d> pose_list;
    if (this->GetAllFK(target_joint, pose_list)) {
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
    if (this->GetAllFK(joint_seed, pose_list)) {
        tcp_pose = pose_list.back();
    } else {
        ik_solutions.success = false;
        return false;
    }

    Eigen::VectorXd iter_joint = joint_seed;
    auto relative_pose = tcp_pose * target_pose.Inverse();
    double angle_err = std::abs(GetRAngle(relative_pose.GetRotation()));
    double translation_err = relative_pose.GetTranslation().norm();
    bool converged = translation_err < this->trans_err_th_ &&
                     angle_err < this->angle_err_th_;
    int i = 0;
    for (; !converged && i < this->max_iter_num_; ++i) {
        SE3d delta_x = target_pose * tcp_pose.Inverse();
        Eigen::MatrixXd jaco;
        this->_GetKinJacobian(jaco, pose_list);
        Eigen::VectorXd delta_theta;
        this->_CalDeltaTheta(delta_x, jaco, delta_theta);
        iter_joint += this->step_size_ * delta_theta;
        for (int joint = 0; joint < this->dof_; ++joint) {
            const auto& node = this->joint_nodes_[joint];
            iter_joint[joint] = clamp(iter_joint[joint], node.lower_limit,
                                      node.upper_limit);
        }
        if (!this->GetAllFK(iter_joint, pose_list)) break;
        tcp_pose = pose_list.back();

        relative_pose = tcp_pose * target_pose.Inverse();
        angle_err = std::abs(GetRAngle(relative_pose.GetRotation()));
        translation_err = relative_pose.GetTranslation().norm();

        if (translation_err < this->trans_err_th_ &&
            angle_err < this->angle_err_th_) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        ik_solutions.success = false;
    } else {
        ik_solutions.success = true;
        ik_solutions.PushBack(iter_joint);
    }

    // Remove repeated IK, if there are.
    ik_solutions.RemoveRepeatedIK();
    // Finds IK that is within joint limits, removes IK that is out of range.
    ik_solutions.GetLimitsIK(this->GetJointNode());

    dist.clear();
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
    if (trans_err_th <= 0) {
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
    if (angle_err_th <= 0) {
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
    if (step_size <= 0) {
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
    if (damp <= 0) {
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
    Eigen::MatrixXd axis_list;
    this->_GetCurrentJointAxisInWorld(axis_list, pose_list);

    jacobian.resize(6, this->dof_);
    const Eigen::Vector3d ee_position = pose_list.back().GetTranslation();

    Eigen::Matrix<double, 6, 1> twist;
    for (int i = 0; i < this->dof_; ++i) {
        twist.setZero();
        auto joint_type = this->joint_nodes_[i].joint_type;

        Eigen::Vector3d joint_start_position = pose_list[i].GetTranslation();
        Eigen::Vector3d joint_axis = axis_list.block(0, i, 3, 1);

        if (JointType::REVOLUTE == joint_type) {
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

bool NumericalKinematics::_GetCurrentJointAxisInWorld(
        Eigen::MatrixXd& axis_list, const std::vector<SE3d>& pose_list) const {
    axis_list.resize(3, this->dof_);
    for (int i = 0; i < this->dof_; ++i) {
        Eigen::Vector3d axis = this->joint_nodes_[i].axis;
        Eigen::Matrix3d joint_rotation = pose_list[i].GetRotation();
        axis_list.col(i) = joint_rotation * axis;
    }
    return true;
}

bool NumericalKinematics::_CalDeltaTheta(SE3d& delta_x,
                                         Eigen::MatrixXd& jacobian,
                                         Eigen::VectorXd& delta_theta,
                                         const IkMethod& method) const {
    if (IkMethod::SVD == method) {
        if (this->_GetDeltaThetaSVD(delta_x, jacobian, delta_theta)) {
            return true;
        }
    }
    if (IkMethod::DAMP_SVD == method) {
        if (this->_GetDeltaThetaDampedSVD(delta_x, jacobian, delta_theta,
                                          this->damp_coeff_)) {
            return true;
        }
    }
    return false;
}

bool NumericalKinematics::_GetDeltaThetaSVD(
        SE3d& delta_x,
        const Eigen::MatrixXd& jacobian,
        Eigen::VectorXd& delta_theta) const {
    // compute SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd sigma = svd.singularValues();
    Eigen::MatrixXd u = svd.matrixU();
    Eigen::MatrixXd v = svd.matrixV();

    // Compute the inverse singular value matrix
    Eigen::MatrixXd inv_sigma =
            Eigen::MatrixXd::Zero(jacobian.cols(), jacobian.rows());
    for (int i = 0; i < sigma.size(); ++i) {
        if (std::abs(sigma(i)) < this->eps_) {
            inv_sigma(i, i) = 0.0;
        } else {
            inv_sigma(i, i) = 1.0 / sigma(i);
        }
    }

    // Compute the inverse Jacobian matrix
    Eigen::MatrixXd inv_jacobian = v * inv_sigma * u.transpose();

    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() = delta_x.GetTranslation();
    // Assuming delta_x.rotation() returns a quaternion, we convert it to a
    // rotation vector
    Eigen::Quaterniond quat(delta_x.GetRotation());
    Eigen::AngleAxisd angleAxis(quat);
    delta.tail<3>() = angleAxis.angle() * angleAxis.axis();

    // compute delta_theta
    delta_theta = inv_jacobian * delta;

    return true;
}

bool NumericalKinematics::_GetDeltaThetaDampedSVD(
        SE3d& delta_x,
        const Eigen::MatrixXd& jacobian,
        Eigen::VectorXd& delta_theta,
        const double& damp_coeffs) const {
    // compute SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd sigma = svd.singularValues();
    Eigen::MatrixXd u = svd.matrixU();
    Eigen::MatrixXd v = svd.matrixV();

    // Compute the inverse singular value matrix of the damping
    Eigen::MatrixXd inv_sigma =
            Eigen::MatrixXd::Zero(jacobian.cols(), jacobian.rows());
    for (int i = 0; i < sigma.size(); ++i) {
        if (std::abs(sigma(i)) > this->eps_) {
            inv_sigma(i, i) = sigma(i) /
                              (sigma(i) * sigma(i) + damp_coeffs * damp_coeffs);
        }
    }

    // Compute the damped inverse Jacobian matrix
    Eigen::MatrixXd damped_inv_jacobian = v * inv_sigma * u.transpose();

    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() = delta_x.GetTranslation();
    // Assuming delta_x.rotation() returns a quaternion, we convert it to a
    // rotation vector
    Eigen::Quaterniond quat(delta_x.GetRotation());
    Eigen::AngleAxisd angleAxis(quat);
    delta.tail<3>() = angleAxis.angle() * angleAxis.axis();

    // compute delta_theta
    delta_theta = damped_inv_jacobian * delta;

    return true;
}

}  // namespace robotics
}  // namespace holistic_motion
