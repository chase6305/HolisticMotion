#include "holistic_motion/kinematics/OPWKinematics.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kDomainTolerance = 1e-10;

double SafeAcos(double value) {
    return std::acos(std::clamp(value, -1.0, 1.0));
}

bool IsFiniteVector(const Eigen::VectorXd& values) {
    return values.allFinite();
}

}  // namespace

namespace holistic_motion {
namespace robotics {

bool OPWKinematics::GetFK(const Eigen::VectorXd &target_joint,
                          SE3d &pose) const {
    holistic_motion::utility::LogDebug("GetFK:Begin to get FK");

    if (target_joint.size() != 6 || !IsFiniteVector(target_joint)) {
        return false;
    }

    double q[6];
    Eigen::VectorXd s(6), c(6);
    for (size_t i = 0; i < 6; ++i) {
        q[i] = target_joint[i] * this->params_.rotation_directions[i] -
               this->params_.offsets[i];
        s[i] = std::sin(q[i]);
        c[i] = std::cos(q[i]);
    }

    double pis_3 = std::atan2(this->params_.a2, this->params_.c3);
    double k = std::sqrt(this->params_.a2 * this->params_.a2 +
                         this->params_.c3 * this->params_.c3);

    double c_x1 = this->params_.c2 * std::sin(q[1]) +
                  k * std::sin(q[1] + q[2] + pis_3) + this->params_.a1;
    double c_y1 = this->params_.b;
    double c_z1 = this->params_.c2 * std::cos(q[1]) +
                  k * std::cos(q[1] + q[2] + pis_3);

    double c_x0 = c_x1 * std::cos(q[0]) - c_y1 * std::sin(q[0]);
    double c_y0 = c_x1 * std::sin(q[0]) + c_y1 * std::cos(q[0]);
    double c_z0 = c_z1 + this->params_.c1;

    Eigen::Matrix3d R_c_oe, R_ob_c, R_ob_oe;
    /* clang-format off */
    R_ob_c << 
    c[0]*c[1]*c[2]-c[0]*s[1]*s[2],  -s[0],  c[0]*c[1]*s[2]+c[0]*s[1]*c[2],
    s[0]*c[1]*c[2]-s[0]*s[1]*s[2],   c[0],  s[0]*c[1]*s[2]+s[0]*s[1]*c[2], 
             -s[1]*c[2]-c[1]*s[2],      0,           -s[1]*s[2]+c[1]*c[2];

    R_c_oe << 
    c[3]*c[4]*c[5]-s[3]*s[5],   -c[3]*c[4]*s[5]-s[3]*c[5],   c[3]*s[4],
    s[3]*c[4]*c[5]+c[3]*s[5],   -s[3]*c[4]*s[5]+c[3]*c[5],   s[3]*s[4], 
                  -s[4]*c[5],                   s[4]*s[5],        c[4];
    /* clang-format on */

    R_ob_oe = R_ob_c * R_c_oe;

    Eigen::Vector3d xyz_b_e =
            Eigen::Vector3d(c_x0, c_y0, c_z0) +
            this->params_.c4 * R_ob_oe * Eigen::Vector3d(0, 0, 1);

    Eigen::Matrix4d T_ob_oe;
    /* clang-format off */
    T_ob_oe <<  R_ob_oe(0, 0), R_ob_oe(0, 1), R_ob_oe(0, 2), xyz_b_e[0], 
                R_ob_oe(1, 0), R_ob_oe(1, 1), R_ob_oe(1, 2), xyz_b_e[1], 
                R_ob_oe(2, 0), R_ob_oe(2, 1), R_ob_oe(2, 2), xyz_b_e[2], 
                0, 0, 0, 1;
    /* clang-format on */

    pose = this->params_.T_b_ob * SE3d(T_ob_oe) * this->params_.T_oe_e;
    pose = pose * this->tcp_;
    holistic_motion::utility::LogDebug("GetFK:Success to get FK");
    return true;
}

bool OPWKinematics::GetIK(const SE3d &target_pose,
                          IkRtn &ik_solutions,
                          Eigen::VectorXd &joint_seed,
                          std::vector<double> &dist) const {
    holistic_motion::utility::LogDebug("GetIK:Begin to get position IK");

    ik_solutions.Clear();
    dist.clear();

    if (joint_seed.size() != this->dof_) {
        joint_seed = this->home_joints_;
    }
    if (!IsFiniteVector(joint_seed)) {
        joint_seed = this->home_joints_;
    }

    SE3d T_ob_oe = target_pose * this->tcp_.Inverse();
    T_ob_oe = this->params_.T_b_ob.Inverse() * T_ob_oe *
              this->params_.T_oe_e.Inverse();

    // get wrist c
    Eigen::Matrix3d R_o = T_ob_oe.GetRotation();
    Eigen::Vector3d xyz_o = T_ob_oe.GetTranslation();
    Eigen::Vector3d c =
            xyz_o - this->params_.c4 * R_o * Eigen::Vector3d(0, 0, 1);

    // c part: compute angle1, angle2, angle3
    const double radial_squared = c[0] * c[0] + c[1] * c[1] -
                                  this->params_.b * this->params_.b;
    if (radial_squared < -kDomainTolerance) {
        return false;
    }
    double n_x1 = std::sqrt(std::max(0.0, radial_squared)) -
                  this->params_.a1;
    double s1_2 = n_x1 * n_x1 + std::pow(c[2] - this->params_.c1, 2);
    double s2_2 = std::pow(n_x1 + 2 * this->params_.a1, 2) +
                  std::pow(c[2] - this->params_.c1, 2);
    double k_2 = this->params_.a2 * this->params_.a2 +
                 this->params_.c3 * this->params_.c3;

    double s1 = std::sqrt(s1_2);
    double s2 = std::sqrt(s2_2);

    double c2_2 = this->params_.c2 * this->params_.c2;
    const double c2_abs = std::abs(this->params_.c2);
    if (s1 < kDomainTolerance || s2 < kDomainTolerance ||
        c2_abs < kDomainTolerance || k_2 < kDomainTolerance) {
        return false;
    }

    // double angle1_i, angle1_ii;
    // double angle2_i, angle2_ii, angle2_iii, angle2_iv;
    // double angle3_i, angle3_ii, angle3_iii, angle3_iv;
    double angle1[4], angle2[4], angle3[4];

    ///< compute angle1
    if (std::abs(c[1]) < Epsilon && std::abs(c[0]) < Epsilon) {
        // shoulder singularity
        angle1[0] = joint_seed[0];
        angle1[2] = joint_seed[0] - M_PI;
        holistic_motion::utility::LogWarning(
                "GetIK: There is a singularity of the shoulder.");
    } else {
        angle1[0] = std::atan2(c[1], c[0]) -
                    std::atan2(this->params_.b, n_x1 + this->params_.a1);
        angle1[2] = std::atan2(c[1], c[0]) +
                    std::atan2(this->params_.b, n_x1 + this->params_.a1) - M_PI;
    }
    angle1[1] = angle1[0];
    angle1[3] = angle1[2];
    ///< compute angle2
    angle2[0] = -SafeAcos((s1_2 + c2_2 - k_2) / (2 * s1 * this->params_.c2)) +
                std::atan2(n_x1, c[2] - this->params_.c1);
    angle2[1] = SafeAcos((s1_2 + c2_2 - k_2) / (2 * s1 * this->params_.c2)) +
                std::atan2(n_x1, c[2] - this->params_.c1);
    angle2[2] =
            -SafeAcos((s2_2 + c2_2 - k_2) / (2 * s2 * this->params_.c2)) -
            std::atan2(n_x1 + 2 * this->params_.a1, c[2] - this->params_.c1);
    angle2[3] =
            SafeAcos((s2_2 + c2_2 - k_2) / (2 * s2 * this->params_.c2)) -
            std::atan2(n_x1 + 2 * this->params_.a1, c[2] - this->params_.c1);

    ///< compute angle3
    double c2k = 2 * this->params_.c2 * std::sqrt(k_2);
    double a2_c3 = std::atan2(this->params_.a2, this->params_.c3);
    angle3[0] = -a2_c3 + SafeAcos((s1_2 - c2_2 - k_2) / c2k);
    angle3[1] = -a2_c3 - SafeAcos((s1_2 - c2_2 - k_2) / c2k);
    angle3[2] = -a2_c3 + SafeAcos((s2_2 - c2_2 - k_2) / c2k);
    angle3[3] = -a2_c3 - SafeAcos((s2_2 - c2_2 - k_2) / c2k);

    // orientation part: compute angle4, angle5, angle6
    double sin1[4], cos1[4], sin23[4], cos23[4];

    sin1[0] = std::sin(angle1[0]);
    sin1[1] = std::sin(angle1[1]);
    sin1[2] = std::sin(angle1[2]);
    sin1[3] = std::sin(angle1[3]);

    cos1[0] = std::cos(angle1[0]);
    cos1[1] = std::cos(angle1[1]);
    cos1[2] = std::cos(angle1[2]);
    cos1[3] = std::cos(angle1[3]);

    double m[4];
    for (unsigned i = 0; i < 4; i++) {
        sin23[i] = std::sin(angle2[i] + angle3[i]);
        cos23[i] = std::cos(angle2[i] + angle3[i]);

        m[i] = R_o(0, 2) * sin23[i] * cos1[i] + R_o(1, 2) * sin23[i] * sin1[i] +
               R_o(2, 2) * cos23[i];
    }

    double angle4[8], angle5[8], angle6[8];

    double m_minus[4];
    for (unsigned i = 0; i < 4; i++) {
        // m_minus = 1 - m * m;
        m_minus[i] = 1 - m[i] * m[i];
        if (m_minus[i] < Epsilon) {
            m_minus[i] = 0;
        }
        ///< compute angle5
        angle5[i] = std::atan2(std::sqrt(m_minus[i]), m[i]);
        angle5[i + 4] = -angle5[i];
    }

    double angle_plus, angle_minus;
    bool is_singularity[4] = {false};

    ///< compute angle4 and angle6
    for (unsigned i = 0; i < 4; i++) {
        if (std::abs(angle5[i]) < Epsilon) {
            angle_minus = std::atan2(R_o(1, 0) * cos1[i] - R_o(0, 0) * sin1[i],
                                     R_o(1, 1) * cos1[i] - R_o(0, 1) * sin1[i]);
            angle4[i] = joint_seed[3];
            angle6[i] = angle_minus - angle4[i];
            is_singularity[i] = true;
        } else if (std::abs(std::abs(angle5[i]) - M_PI) < Epsilon) {
            angle_plus = std::atan2(R_o(1, 0) * cos1[i] - R_o(0, 0) * sin1[i],
                                    R_o(1, 1) * cos1[i] - R_o(0, 1) * sin1[i]);
            angle4[i] = joint_seed[3];
            angle6[i] = angle_plus + angle4[i];
            is_singularity[i] = true;
        }
        if (!is_singularity[i]) {
            /* clang-format off */
            angle4[i] = std::atan2(
                R_o(1, 2) * cos1[i] - R_o(0, 2) * sin1[i],
                R_o(0, 2) * cos23[i] * cos1[i] + 
                R_o(1, 2) * cos23[i] * sin1[i] -
                R_o(2, 2) * sin23[i]);
            angle6[i] = std::atan2(
                R_o(0, 1) * sin23[i] * cos1[i] +
                R_o(1, 1) * sin23[i] * sin1[i] +
                R_o(2, 1) * cos23[i], -
                R_o(0, 0) * sin23[i] * cos1[i] -
                R_o(1, 0) * sin23[i] * sin1[i] -
                R_o(2, 0) * cos23[i]);
            /* clang-format on */
        } else {
            holistic_motion::utility::LogWarning(
                    "GetIK: There is a singularity of the wrist.");
        }
        angle4[i + 4] = angle4[i] + M_PI;
        angle6[i + 4] = angle6[i] - M_PI;
    }

    SE3d temp_pose;
    for (unsigned i = 0; i < 8; i++) {
        holistic_motion::utility::LogDebug(
                "GetIK: result "
                "joint:\n[{:10.8f},{:10.8f},{:10.8f},{:10.8f},{:10."
                "8f},{:10.8f}]",
                angle1[i % 4], angle2[i % 4], angle3[i % 4], angle4[i],
                angle5[i], angle6[i]);
        if (std::isfinite(angle1[i % 4]) && std::isfinite(angle2[i % 4]) &&
            std::isfinite(angle3[i % 4]) && std::isfinite(angle4[i]) &&
            std::isfinite(angle5[i]) && std::isfinite(angle6[i])) {
            Eigen::VectorXd joints(6);
            joints << (angle1[i % 4] + this->params_.offsets[0]) *
                              this->params_.rotation_directions[0],
                    (angle2[i % 4] + this->params_.offsets[1]) *
                            this->params_.rotation_directions[1],
                    (angle3[i % 4] + this->params_.offsets[2]) *
                            this->params_.rotation_directions[2],
                    (angle4[i] + this->params_.offsets[3]) *
                            this->params_.rotation_directions[3],
                    (angle5[i] + this->params_.offsets[4]) *
                            this->params_.rotation_directions[4],
                    (angle6[i] + this->params_.offsets[5]) *
                            this->params_.rotation_directions[5];
            GetFK(joints, temp_pose);
            if (target_pose.IsApprox(temp_pose)) {
                // joint_seed
                ik_solutions.PushBack(joints);
            } else {
                holistic_motion::utility::LogDebug(
                        "GetIK: target pose is not approx to temp "
                        "pose:\ntarget pose:\t[{:10.8f}]\ntemp "
                        "pose:\t[{:10.8f}]",
                        fmt::join(target_pose.Coeffs(), " , "),
                        fmt::join(temp_pose.Coeffs(), " , "));
            }
        } else {
            holistic_motion::utility::LogDebug("GetIK: There is [nan] data in the joint.");
        }
    }

    // Remove repeated IK, if there are.
    ik_solutions.RemoveRepeatedIK();
    // Finds IK that is within joint limits, removes IK that is out of range.
    ik_solutions.WrapToLimitsNear(this->GetJointNode(), joint_seed);
    // Filter some IK results with joint filter config(each joint)
    ik_solutions.FilterJointSolutions(this->joint_filter_config_);
    // Filter some IK results with robot config(base\elbow\wrist)
    ik_solutions.LimitRobotConfig(this->GetJointNode(), this->robot_config_);

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

    // Sorting with minimum movement of joint_seed.
    ik_solutions.SortMinMovement(joint_seed, dist);

    if (ik_solutions.success && ik_solutions.ik_number > 0) {
        holistic_motion::utility::LogDebug(
                "GetIK: Success to get position IK, ik number:[{}].",
                ik_solutions.ik_number);
        return true;
    }

    holistic_motion::utility::LogDebug("GetIK: Failed to get position IK.");
    return false;
}

bool OPWKinematics::GetNearstIK(const SE3d &target_pose,
                                IkRtn &ik_solutions,
                                Eigen::VectorXd &joint_seed,
                                double &min_dist) const {
    holistic_motion::utility::LogDebug("GetNearstIK:Begin to get nearst IK");

    if (joint_seed.size() != this->dof_) {
        joint_seed = this->home_joints_;
    }

    std::vector<double> dist;
    if (!GetIK(target_pose, ik_solutions, joint_seed, dist)) {
        return false;
    }
    // ik_solutions
    if (ik_solutions.success) {
        if (!dist.empty()) {
            min_dist = dist.front();
        }
        if (ik_solutions.ik_number != 1) {
            ik_solutions.ik_number = 1;
            ik_solutions.ik_joints.erase(ik_solutions.ik_joints.begin() + 1,
                                         ik_solutions.ik_joints.end());
        }
        holistic_motion::utility::LogDebug("GetNearstIK: Success to get nearst IK.");
        return true;
    }

    holistic_motion::utility::LogWarning("GetNearstIK: Failed to get nearst IK.");
    return false;
}

}  // namespace robotics
}  // namespace holistic_motion
