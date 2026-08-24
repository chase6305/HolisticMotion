#include "holistic_motion/kinematics/URKinematics.h"

#include <algorithm>
#include <cmath>

namespace {

double SafeAcos(double value) {
    return std::acos(std::clamp(value, -1.0, 1.0));
}

double SafeAsin(double value) {
    return std::asin(std::clamp(value, -1.0, 1.0));
}

}  // namespace

namespace holistic_motion {
namespace robotics {

bool URKinematics::GetFK(const Eigen::VectorXd &target_joint,
                         SE3d &pose) const {
    holistic_motion::utility::LogDebug("GetFK:Begin to get FK");

    if (target_joint.size() != 6 || !target_joint.allFinite()) {
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

    double s23 = std::sin(q[1] + q[2]);
    double c23 = std::cos(q[1] + q[2]);
    double s234 = std::sin(q[1] + q[2] + q[3]);
    double c234 = std::cos(q[1] + q[2] + q[3]);

    Eigen::Matrix4d T_ob_oe;
    /* clang-format off */
    T_ob_oe << c234 * c[0] * s[4] - c[4] * s[0],
        c[5] * (s[0] * s[4] + c234 * c[0] * c[4]) - s234 * c[0] * s[5],
        -s[5] * (s[0] * s[4] + c234 * c[0] * c[4]) - s234 * c[0] * c[5],
        this->params_.d6 * c234 * c[0] * s[4] - this->params_.a3 * c23 * c[0]
            - this->params_.a2 * c[0] * c[1] - this->params_.d6 * c[4] * s[0]
            - this->params_.d5 * s234 * c[0] - this->params_.d4 * s[0],
        c[0] * c[4] + c234 * s[0] * s[4],
        -c[5] * (c[0] * s[4] - c234 * c[4] * s[0]) - s234 * s[0] * s[5],
        s[5] * (c[0] * s[4] - c234 * c[4] * s[0]) - s234 * c[5] * s[0],
        this->params_.d6 * (c[0] * c[4] + c234 * s[0] * s[4]) + this->params_.d4 * c[0]
            - this->params_.a3 * c23 * s[0] - this->params_.a2 * c[1] * s[0]
            - this->params_.d5 * s234 * s[0],
        -s234 * s[4], 
        -c234 * s[5] - s234 * c[4] * c[5], 
        s234 * c[4] * s[5] - c234 * c[5],
        this->params_.d1 + this->params_.a3 * s23 + this->params_.a2 * s[1]
            - this->params_.d5 * (c23 * c[3] - s23 * s[3])
            - this->params_.d6 * s[4] * (c23 * s[3] + s23 * c[3]),
        0, 0, 0, 1;
    /* clang-format on */

    pose = this->params_.T_b_ob * SE3d(T_ob_oe) * this->params_.T_oe_e;
    pose = pose * this->tcp_;
    holistic_motion::utility::LogDebug("GetFK:Success to get FK");
    return true;
}

bool URKinematics::GetIK(const SE3d &target_pose,
                         IkRtn &ik_solutions,
                         Eigen::VectorXd &joint_seed,
                         std::vector<double> &dist) const {
    holistic_motion::utility::LogDebug("GetIK:Begin to get position IK");

    ik_solutions.Clear();
    dist.clear();

    if (joint_seed.size() != this->dof_) {
        joint_seed = this->home_joints_;
    }
    if (!joint_seed.allFinite()) {
        joint_seed = this->home_joints_;
    }

    SE3d T_ob_oe = target_pose * this->tcp_.Inverse();
    T_ob_oe = this->params_.T_b_ob.Inverse() * T_ob_oe *
              this->params_.T_oe_e.Inverse();

    Eigen::Matrix4d T_b_e = T_ob_oe.GetTransform();

    double T00 = T_b_e(0, 0), T01 = T_b_e(0, 1), T02 = T_b_e(0, 2),
           T03 = T_b_e(0, 3);
    double T10 = T_b_e(1, 0), T11 = T_b_e(1, 1), T12 = T_b_e(1, 2),
           T13 = T_b_e(1, 3);
    double T20 = T_b_e(2, 0), T21 = T_b_e(2, 1), T22 = T_b_e(2, 2),
           T23 = T_b_e(2, 3);

    double q6_des = joint_seed[5];

    while (q6_des > 2.0 * M_PI) {
        q6_des -= 2.0 * M_PI;
    }
    while (q6_des < 0) {
        q6_des += 2.0 * M_PI;
    }

    // angle1
    double angle1[2];
    {
        double A = T03 - this->params_.d6 * T00;
        double B = this->params_.d6 * T10 - T13;
        double R = A * A + B * B;

        if (std::abs(A) < Epsilon) {
            double div;
            if (std::abs(std::abs(params_.d4) - std::abs(B)) < Epsilon) {
                div = -static_cast<double>(Sign(this->params_.d4)) *
                      static_cast<double>(Sign(B));
            } else {
                div = -this->params_.d4 / B;
            }
            angle1[0] = SafeAcos(div);
            angle1[1] = M_PI - angle1[0];
        } else if (std::abs(B) < Epsilon) {
            double div;
            if (std::abs(std::abs(this->params_.d4) - std::abs(A)) < Epsilon) {
                div = -static_cast<double>(Sign(this->params_.d4)) *
                      static_cast<double>(Sign(A));
            } else {
                div = -this->params_.d4 / A;
            }
            double arcsin = SafeAsin(div);
            if (std::abs(arcsin) < Epsilon) {
                arcsin = 0.0;
            }
            if (arcsin < 0.0) {
                angle1[0] = arcsin + 2.0 * M_PI;
            } else {
                angle1[0] = arcsin;
            }
            angle1[1] = M_PI - angle1[0];
        } else if (this->params_.d4 * this->params_.d4 > R) {
            holistic_motion::utility::LogWarning("Can not found angle1");
            return false;
        } else {
            double arccos = SafeAcos(-this->params_.d4 / std::sqrt(R));
            double arctan = std::atan2(A, B);
            double pos = arccos + arctan;
            double neg = -arccos + arctan;
            if (std::abs(pos) < Epsilon) {
                pos = 0.0;
            }
            if (std::abs(neg) < Epsilon) {
                neg = 0.0;
            }
            if (pos >= 0.0) {
                angle1[0] = pos;
            } else {
                angle1[0] = 2.0 * M_PI + pos;
            }
            if (neg >= 0.0) {
                angle1[1] = neg;
            } else {
                angle1[1] = 2.0 * M_PI + neg;
            }
        }
    }

    // angle5
    double angle5[2][2];
    {
        for (size_t i = 0; i < 2; ++i) {
            double div = T10 * std::cos(angle1[i]) - T00 * std::sin(angle1[i]);
            if (std::abs(div - 1) < Epsilon) {
                div = 1.0;
            }
            if (std::abs(div + 1) < Epsilon) {
                div = -1.0;
            }
            if (div < -1.0 - Epsilon || div > 1.0 + Epsilon) {
                continue;
            }
            double arccos = SafeAcos(div);
            angle5[i][0] = arccos;
            angle5[i][1] = 2.0 * M_PI - arccos;
        }
    }

    SE3d temp_pose;
    {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                double c1 = std::cos(angle1[i]), s1 = std::sin(angle1[i]);
                // double c5 = std::cos(angle5[i][j]);
                double s5 = std::sin(angle5[i][j]);

                // angle6
                double angle6;
                if (std::abs(s5) < Epsilon) {
                    angle6 = q6_des;
                } else {
                    angle6 = std::atan2(
                            static_cast<double>(Sign(s5)) *
                                    (T12 * c1 - T02 * s1),
                            static_cast<double>(Sign(s5) *
                                                (T01 * s1 - T11 * c1)));
                    if (std::abs(angle6) < Epsilon) {
                        angle6 = 0.0;
                    }
                    if (angle6 < 0.0) {
                        angle6 += 2.0 * M_PI;
                    }
                }
                // angle2, angle3, angle4
                double angle2[2], angle3[2], angle4[2];
                double c6 = std::cos(angle6), s6 = std::sin(angle6);

                double x04x = -T02 * c1 * c6 - T12 * s1 * c6 - T01 * c1 * s6 -
                              T11 * s1 * s6;
                double x04y = -T22 * c6 - T21 * s6;

                double p13x = this->params_.d6 * T00 * c1 - T13 * s1 -
                              T03 * c1 + this->params_.d6 * T10 * s1 -
                              this->params_.d5 * x04x;
                double p13y = T23 - this->params_.d1 - this->params_.d6 * T20 +
                              this->params_.d5 * x04y;
                double m = p13x * p13x + p13y * p13y;
                double c3 = (m + this->params_.a2 * this->params_.a2 -
                             this->params_.a3 * this->params_.a3) /
                            (2.0 * this->params_.a2);

                if (std::abs(p13x) < Epsilon) {
                    double div;
                    if (std::abs(std::abs(c3) - std::abs(p13y)) < Epsilon)
                        div = static_cast<double>(Sign(c3)) *
                              static_cast<double>(Sign(p13y));
                    else
                        div = c3 / p13y;
                    if (std::abs(div) > 1) {
                        // todo: no solution
                        continue;
                    }
                    double arcsin = SafeAsin(div);
                    if (std::abs(arcsin) < Epsilon) {
                        arcsin = 0;
                    }
                    if (arcsin < 0) {
                        angle2[0] = arcsin + 2.0 * M_PI;
                    } else {
                        angle2[0] = arcsin;
                    }
                    angle2[1] = M_PI - arcsin;
                } else if (std::abs(p13y) < Epsilon) {
                    double div;
                    if (std::abs(std::abs(c3) - std::abs(p13x)) < Epsilon)
                        div = static_cast<double>(Sign(c3)) *
                              static_cast<double>(Sign(p13x));
                    else
                        div = c3 / p13x;
                    if (std::abs(div) > 1) {
                        // todo: no solution
                        continue;
                    }
                    angle2[0] = SafeAcos(div);
                    angle2[1] = 2.0 * M_PI - angle2[0];
                } else if (std::abs(m - c3 * c3) < Epsilon) {
                    angle2[0] = std::acos(static_cast<double>(Sign(c3))) +
                                std::atan2(p13y, p13x);
                    angle2[1] = 2.0 * M_PI -
                                std::acos(static_cast<double>(Sign(c3))) +
                                std::atan2(p13y, p13x);
                } else if (m < c3 * c3) {
                    // todo: no solution
                    continue;
                } else {
                    double arctan = std::atan2(p13y, p13x);
                    double arccos = SafeAcos(c3 / std::sqrt(m));
                    double pos = arccos + arctan;
                    double neg = -arccos + arctan;
                    if (std::abs(pos) < Epsilon) pos = 0;
                    if (std::abs(neg) < Epsilon) neg = 0;
                    if (pos >= 0) {
                        angle2[0] = pos;
                    } else {
                        angle2[0] = pos + 2.0 * M_PI;
                    }
                    if (neg >= 0) {
                        angle2[1] = neg;
                    } else {
                        angle2[1] = neg + 2.0 * M_PI;
                    }
                }

                for (size_t k = 0; k < 2; ++k) {
                    double s2 = std::sin(angle2[k]);
                    double c2 = std::cos(angle2[k]);
                    double q23 = std::atan2(
                            (p13y - this->params_.a2 * s2) *
                                    static_cast<double>(Sign(this->params_.a3)),
                            (p13x - this->params_.a2 * c2) *
                                    static_cast<double>(
                                            Sign(this->params_.a3)));
                    angle3[k] = q23 - angle2[k];
                    angle4[k] = std::atan2(x04x, x04y) - q23;
                    if (angle3[k] < 0) angle3[k] += 2.0 * M_PI;
                    if (angle4[k] < 0) angle4[k] += 2.0 * M_PI;

                    if (std::abs(angle4[k]) < Epsilon ||
                        std::abs(angle4[k] - 2.0 * M_PI) < Epsilon) {
                        angle4[k] = 0.0;
                    }
                    holistic_motion::utility::LogDebug(
                            "GetIK: result "
                            "joint:\n[{:10.8f},{:10.8f},{:10.8f},{:10.8f},{:10."
                            "8f},{:10.8f}]",
                            angle1[i], angle2[k], angle3[k], angle4[k],
                            angle5[i][j], angle6);
                    if (std::isfinite(angle1[i]) && std::isfinite(angle2[k]) &&
                        std::isfinite(angle3[k]) && std::isfinite(angle4[k]) &&
                        std::isfinite(angle5[i][j]) && std::isfinite(angle6)) {
                        Eigen::VectorXd joints(6);
                        joints << (angle1[i] + this->params_.offsets[0]) *
                                          this->params_.rotation_directions[0],
                                (angle2[k] + this->params_.offsets[1]) *
                                        this->params_.rotation_directions[1],
                                (angle3[k] + this->params_.offsets[2]) *
                                        this->params_.rotation_directions[2],
                                (angle4[k] + this->params_.offsets[3]) *
                                        this->params_.rotation_directions[3],
                                (angle5[i][j] + this->params_.offsets[4]) *
                                        this->params_.rotation_directions[4],
                                (angle6 + this->params_.offsets[5]) *
                                        this->params_.rotation_directions[5];
                        GetFK(joints, temp_pose);
                        if (target_pose.IsApprox(temp_pose)) {
                            // joint_seed
                            ik_solutions.PushBack(joints);
                        } else {
                            holistic_motion::utility::LogDebug(
                                    "GetIK: target pose is not approx to temp"
                                    "pose:\ntarget pose:\t[{:10.8f}]\ntemp"
                                    "pose:\t[{:10.8f}]",
                                    fmt::join(target_pose.Coeffs(), " , "),
                                    fmt::join(temp_pose.Coeffs(), " , "));
                        }
                    } else {
                        holistic_motion::utility::LogDebug(
                                "GetIK: There is [nan] data in the joint.");
                    }
                }
            }
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

    holistic_motion::utility::LogDebug(
            "GetIK: Failed to get position IK, success:[{}], ik_number:[{}]",
            ik_solutions.success, ik_solutions.ik_number);
    return false;
}

bool URKinematics::GetNearstIK(const SE3d &target_pose,
                               IkRtn &ik_solutions,
                               Eigen::VectorXd &joint_seed,
                               double &min_dist) const {
    holistic_motion::utility::LogDebug("GetNearstIK:Begin to get nearst IK");

    if (joint_seed.size() != this->dof_) {
        joint_seed = this->home_joints_;
    }

    std::vector<double> dist;
    if (GetIK(target_pose, ik_solutions, joint_seed, dist)) {
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
    }

    holistic_motion::utility::LogWarning("GetNearstIK: Failed to get nearst IK.");
    return false;
}

}  // namespace robotics
}  // namespace holistic_motion
