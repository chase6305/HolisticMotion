#include "holistic_motion/kinematics/NumericalKinematics.h"

#include "DampedIKWorkspace.h"
#include "FrameValidation.h"
#include "ForwardKinematics.h"
#include "SeedAlias.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <typeinfo>

namespace holistic_motion {
namespace robotics {

KinematicsCapabilities NumericalKinematics::GetCapabilities() const {
    auto result = KinematicsBase::GetCapabilities();
    result.family = KinematicsFamily::NUMERICAL;
    result.default_ik_algorithm = IKAlgorithm::NUMERICAL;
    result.supports_jacobian = true;
    result.supports_concurrent_queries = typeid(*this) == typeid(NumericalKinematics);
    return result;
}
namespace {

double TranslationNorm(const Eigen::Vector3d &value) {
    const double squared = value.squaredNorm();
    return (std::isnormal(squared) ||
            (value.x() == 0.0 && value.y() == 0.0 && value.z() == 0.0))
               ? std::sqrt(squared)
               : std::hypot(value.x(), value.y(), value.z());
}

} // namespace

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

bool NumericalKinematics::UsesDefaultForwardKinematics() const noexcept {
    return typeid(*this) == typeid(NumericalKinematics);
}

bool NumericalKinematics::GetFK(const Eigen::VectorXd& target_joint,
                                SE3d& pose) const {
    if (UsesDefaultForwardKinematics())
        return detail::ComputeTerminalFK(*this, target_joint, pose);
    std::vector<SE3d> pose_list;
    if (this->GetAllFK(target_joint, pose_list) && !pose_list.empty()) {
        pose = pose_list.back();
        return true;
    }
    return false;
}

bool NumericalKinematics::GetIK(const SE3d &target_pose, IkRtn &ik_solutions,
                                Eigen::VectorXd &joint_seed,
                                std::vector<double> &dist) const {
    if (detail::SeedAliasesSolutions(ik_solutions, joint_seed)) {
        Eigen::VectorXd stable_seed = joint_seed;
        return GetIK(target_pose, ik_solutions, stable_seed, dist);
    }

    return SolveIKImpl(target_pose, ik_solutions, joint_seed, dist, nullptr);
}

bool NumericalKinematics::SolveIKImpl(const SE3d &target_pose, IkRtn &ik_solutions,
                                      Eigen::VectorXd &joint_seed,
                                      std::vector<double> &dist,
                                      const Eigen::VectorXd *nearest_reference) const {
    holistic_motion::utility::LogDebug("GetNearstIK:Begin to get position IK.");
    ik_solutions.Clear();
    dist.clear();

    if (joint_seed.size() != this->dof_) {
        joint_seed = this->home_joints_;
    }
    if (!joint_seed.allFinite() || !detail::IsValidFrame(target_pose)) {
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
    double translation_err = TranslationNorm(delta_x.GetTranslation());
    bool converged = translation_err < this->trans_err_th_ &&
                     angle_err < this->angle_err_th_;
    // Exact seeds avoid allocating iteration storage altogether.
    if (!converged && this->dof_ > 0) {
        Eigen::MatrixXd jacobian(6, this->dof_);
        Eigen::VectorXd delta_theta(this->dof_);
        detail::DampedIKWorkspace workspace(this->dof_);
        for (int i = 0; i < this->max_iter_num_; ++i) {
            if (!this->_GetKinJacobian(jacobian, pose_list))
                break;
            Eigen::Matrix<double, 6, 1> error;
            error.head<3>() = delta_x.GetTranslation();
            const Eigen::AngleAxisd rotation_error(
                Eigen::Quaterniond(delta_x.GetRotation()));
            error.tail<3>() = rotation_error.angle() * rotation_error.axis();
            if (rotation_error.angle() == 0.0 && angle_err > 0.0) {
                // Eigen's quaternion-vector norm can square away a tiny
                // nonzero angle. At this scale sin(angle) equals angle to
                // floating precision, so use the rotation's skew directly.
                const auto rotation = delta_x.GetRotation();
                error.tail<3>() =
                    0.5 * Eigen::Vector3d(rotation(2, 1) - rotation(1, 2),
                                          rotation(0, 2) - rotation(2, 0),
                                          rotation(1, 0) - rotation(0, 1));
            }
            if (!workspace.Solve(jacobian, error, this->damp_coeff_, this->eps_,
                                 delta_theta))
                break;
            iter_joint += this->step_size_ * delta_theta;
            for (int joint = 0; joint < this->dof_; ++joint) {
                const auto &node = this->joint_nodes_[joint];
                iter_joint[joint] = clamp(iter_joint[joint], node.lower_limit,
                                          node.upper_limit);
            }
            if (!this->GetAllFK(iter_joint, pose_list) ||
                pose_list.size() <= static_cast<std::size_t>(this->dof_))
                break;
            tcp_pose = pose_list.back();

            delta_x = task_error(tcp_pose);
            angle_err = std::abs(GetRAngle(delta_x.GetRotation()));
            translation_err = TranslationNorm(delta_x.GetTranslation());

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

    // The numerical iteration produces at most one branch. Nearest-only
    // queries wrap that branch directly instead of expanding its Cartesian
    // product of equivalent turns. All-solutions queries retain their budget.
    if (nearest_reference)
        ik_solutions.WrapToLimitsNear(this->joint_nodes_, *nearest_reference);
    else
        ik_solutions.GetLimitsIK(this->joint_nodes_);

    const auto &reference = nearest_reference ? *nearest_reference : joint_seed;
    for (int i = 0; i < ik_solutions.ik_number; ++i) {
        double temp_dist = 0.0;
        for (int j = 0; j < this->dof_; ++j) {
            temp_dist +=
                std::abs((ik_solutions.ik_joints[i][j] - reference[j]) *
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

bool NumericalKinematics::GetNearstIK(const SE3d &target_pose,
                                      IkRtn &ik_solutions,
                                      Eigen::VectorXd &seed_input,
                                      double &min_dist) const {
    const bool aliased = detail::SeedAliasesSolutions(ik_solutions, seed_input);
    Eigen::VectorXd seed_copy;
    if (aliased) {
        seed_copy = seed_input;
    }
    Eigen::VectorXd &joint_seed = aliased ? seed_copy : seed_input;

    holistic_motion::utility::LogDebug("GetNearstIK:Begin to get nearst IK.");

    if (joint_seed.size() != this->dof_) {
        holistic_motion::utility::LogWarning(
            "Joint seed size[{}] not match dof[{}].", joint_seed.size(),
            this->dof_);
        joint_seed = this->home_joints_;
    }

    if (!joint_seed.allFinite()) {
        ik_solutions.Clear();
        return false;
    }

    std::vector<double> dist;
    if (SolveIKImpl(target_pose, ik_solutions, joint_seed, dist, &joint_seed)) {
        // ik_solutions
        if (ik_solutions.success) {
            min_dist = dist.front();
            holistic_motion::utility::LogDebug(
                "GetNearstIK: Success to get nearst IK.");
            return true;
        }
    }

    ik_solutions.Clear();
    Eigen::VectorXd zero_joint_seed = Eigen::VectorXd::Zero(this->dof_);
    if (SolveIKImpl(target_pose, ik_solutions, zero_joint_seed, dist, &joint_seed)) {
        // ik_solutions
        if (ik_solutions.success) {
            min_dist = dist.front();
            holistic_motion::utility::LogDebug(
                "GetNearstIK: Success to get nearst IK.");
            return true;
        }
    }

    holistic_motion::utility::LogDebug("GetNearstIK: Failed to get nearst IK.");
    return false;
}

bool NumericalKinematics::SetMaxIterNum(const int &max_iter_num) {
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
