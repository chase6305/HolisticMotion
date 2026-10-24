#include "holistic_motion/kinematics/URKinematics.h"

#include "FrameValidation.h"

#include "SeedAlias.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <typeinfo>

namespace {

constexpr double kPoseTolerance = 1e-10;

double SafeAcos(double value) {
    return std::acos(std::clamp(value, -1.0, 1.0));
}

double SafeAsin(double value) {
    return std::asin(std::clamp(value, -1.0, 1.0));
}

}  // namespace

namespace holistic_motion {
namespace robotics {

KinematicsCapabilities URKinematics::GetCapabilities() const {
    auto result = KinematicsBase::GetCapabilities();
    result.family = KinematicsFamily::UR;
    result.default_ik_algorithm = IKAlgorithm::ANALYTIC;
    result.supports_closed_form_ik = true;
    result.supports_concurrent_queries = typeid(*this) == typeid(URKinematics);
    return result;
}

bool URKinematics::GetFK(const Eigen::VectorXd &target_joint,
                         SE3d &pose) const {
    holistic_motion::utility::LogDebug("GetFK:Begin to get FK");

    if (target_joint.size() != 6 || !target_joint.allFinite()) {
        return false;
    }

    std::array<double, 6> q, s, c;
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
    if (detail::SeedAliasesSolutions(ik_solutions, joint_seed)) {
        Eigen::VectorXd stable_seed = joint_seed;
        return GetIK(target_pose, ik_solutions, stable_seed, dist);
    }

    holistic_motion::utility::LogDebug("GetIK:Begin to get position IK");

    ik_solutions.Clear();
    dist.clear();
    if (!detail::IsValidFrame(target_pose))
        return false;

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

    const double wrist_seed_angle =
        joint_seed[5] * this->params_.rotation_directions[5] -
        this->params_.offsets[5];
    const double seed_sine = std::sin(wrist_seed_angle);
    const double seed_cosine = std::cos(wrist_seed_angle);
    const double wrist_seed = std::abs(wrist_seed_angle) <= M_PI
                                  ? wrist_seed_angle
                                  : std::atan2(seed_sine, seed_cosine);
    const double wrist_reach = std::abs(this->params_.d5) +
                               std::abs(this->params_.d6) +
                               this->params_.T_oe_e.GetTranslation().norm() +
                               this->tcp_.GetTranslation().norm();
    const double seed_axis_tolerance =
        kPoseTolerance / std::max(1.0, wrist_reach);
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

    // Recover each wrist branch before solving its planar arm.
    for (size_t i = 0; i < 2; ++i) {
        const double c1 = std::cos(angle1[i]);
        const double s1 = std::sin(angle1[i]);
        const double wrist_cosine = T10 * c1 - T00 * s1;
        const double wrist_x = T01 * s1 - T11 * c1;
        const double wrist_y = T12 * c1 - T02 * s1;
        const double transverse_error =
            -seed_sine * wrist_x + seed_cosine * wrist_y;
        double first_wrist_bend, first_wrist_roll;
        if (std::abs(transverse_error) < seed_axis_tolerance) {
            first_wrist_bend = std::atan2(
                seed_cosine * wrist_x + seed_sine * wrist_y, wrist_cosine);
            first_wrist_roll = wrist_seed;
        } else {
            first_wrist_bend =
                std::atan2(std::hypot(wrist_x, wrist_y), wrist_cosine);
            first_wrist_roll = std::atan2(wrist_y, wrist_x);
        }
        // The circular wrist limit differs by O(sin(q5)^2) near a singularity.
        const bool near_singular_wrist =
            wrist_x * wrist_x + wrist_y * wrist_y < seed_axis_tolerance;
        const auto project_wrist_roll = [&](double roll, double cosine,
                                            double sine) {
            if (!near_singular_wrist)
                return roll;
            // Project the wrist's circular limit onto the arm's reach annulus.
            const double center_x = this->params_.d6 * T00 * c1 - T13 * s1 -
                                    T03 * c1 + this->params_.d6 * T10 * s1;
            const double center_y =
                T23 - this->params_.d1 - this->params_.d6 * T20;
            const double cosine_x = this->params_.d5 * (T02 * c1 + T12 * s1);
            const double cosine_y = -this->params_.d5 * T22;
            const double sine_x = this->params_.d5 * (T01 * c1 + T11 * s1);
            const double sine_y = -this->params_.d5 * T21;
            const double radius =
                std::hypot(center_x + cosine_x * cosine + sine_x * sine,
                           center_y + cosine_y * cosine + sine_y * sine);
            const double minimum_radius = std::abs(std::abs(this->params_.a2) -
                                                   std::abs(this->params_.a3));
            const double maximum_radius =
                std::abs(this->params_.a2) + std::abs(this->params_.a3);
            if (radius + kPoseTolerance >= minimum_radius &&
                radius <= maximum_radius + kPoseTolerance)
                return roll;
            const double cosine_coefficient =
                center_x * cosine_x + center_y * cosine_y;
            const double sine_coefficient =
                center_x * sine_x + center_y * sine_y;
            const double amplitude =
                2.0 * std::hypot(cosine_coefficient, sine_coefficient);
            if (!(amplitude > 0.0))
                return roll;
            const double mean_radius_squared =
                center_x * center_x + center_y * center_y +
                this->params_.d5 * this->params_.d5;
            const double lower = std::max(
                -1.0, (minimum_radius * minimum_radius - mean_radius_squared) /
                          amplitude);
            const double upper = std::min(
                1.0, (maximum_radius * maximum_radius - mean_radius_squared) /
                         amplitude);
            if (lower > upper)
                return roll;
            const double phase =
                std::atan2(sine_coefficient, cosine_coefficient);
            const double offset =
                SafeAcos(std::clamp(std::cos(roll - phase), lower, upper));
            const double first = phase + offset, second = phase - offset;
            return std::abs(std::remainder(first - roll, 2.0 * M_PI)) <=
                           std::abs(std::remainder(second - roll, 2.0 * M_PI))
                       ? first
                       : second;
        };
        for (size_t j = 0; j < 2; ++j) {
            double angle6 = first_wrist_roll + (j == 0 ? 0.0 : M_PI);
            double c6 = std::cos(angle6);
            double s6 = std::sin(angle6);
            double angle5 =
                j == 0 ? first_wrist_bend : 2.0 * M_PI - first_wrist_bend;
            const double projected_roll = project_wrist_roll(angle6, c6, s6);
            if (projected_roll != angle6) {
                const double projected_cosine = std::cos(projected_roll);
                const double projected_sine = std::sin(projected_roll);
                if (std::abs(-projected_sine * wrist_x +
                             projected_cosine * wrist_y) <
                    seed_axis_tolerance) {
                    angle6 = projected_roll;
                    c6 = projected_cosine;
                    s6 = projected_sine;
                    angle5 =
                        std::atan2(c6 * wrist_x + s6 * wrist_y, wrist_cosine);
                }
            }
            // angle2, angle3, angle4

            double x04x =
                -T02 * c1 * c6 - T12 * s1 * c6 - T01 * c1 * s6 - T11 * s1 * s6;
            double x04y = -T22 * c6 - T21 * s6;

            double p13x = this->params_.d6 * T00 * c1 - T13 * s1 - T03 * c1 +
                          this->params_.d6 * T10 * s1 - this->params_.d5 * x04x;
            double p13y = T23 - this->params_.d1 - this->params_.d6 * T20 +
                          this->params_.d5 * x04y;
            const double radius_squared = p13x * p13x + p13y * p13y;
            const double elbow_cosine = std::clamp(
                (radius_squared - this->params_.a2 * this->params_.a2 -
                 this->params_.a3 * this->params_.a3) /
                    (2.0 * this->params_.a2 * this->params_.a3),
                -1.0, 1.0);
            const double elbow_sine = std::sqrt(
                std::max(0.0, (1.0 - elbow_cosine) * (1.0 + elbow_cosine)));
            const double elbow_angle = std::atan2(elbow_sine, elbow_cosine);
            const double bearing = std::atan2(p13y, p13x);
            const double bend =
                std::atan2(this->params_.a3 * elbow_sine,
                           this->params_.a2 + this->params_.a3 * elbow_cosine);
            for (size_t k = 0; k < 2; ++k) {
                const double angle2 = bearing + (k == 0 ? -bend : bend);
                const double angle3 = k == 0 ? elbow_angle : -elbow_angle;
                double angle4 = std::atan2(x04x, x04y) - angle2 - angle3;
                if (angle4 < 0.0)
                    angle4 += 2.0 * M_PI;

                if (std::abs(angle4) < Epsilon ||
                    std::abs(angle4 - 2.0 * M_PI) < Epsilon) {
                    angle4 = 0.0;
                }
                holistic_motion::utility::LogDebug(
                    "GetIK: result "
                    "joint:\n[{:10.8f},{:10.8f},{:10.8f},{:10.8f},{:10."
                    "8f},{:10.8f}]",
                    angle1[i], angle2, angle3, angle4, angle5, angle6);
                if (std::isfinite(angle1[i]) && std::isfinite(angle2) &&
                    std::isfinite(angle3) && std::isfinite(angle4) &&
                    std::isfinite(angle5) && std::isfinite(angle6)) {
                    Eigen::VectorXd joints(6);
                    joints << (angle1[i] + this->params_.offsets[0]) *
                                  this->params_.rotation_directions[0],
                        (angle2 + this->params_.offsets[1]) *
                            this->params_.rotation_directions[1],
                        (angle3 + this->params_.offsets[2]) *
                            this->params_.rotation_directions[2],
                        (angle4 + this->params_.offsets[3]) *
                            this->params_.rotation_directions[3],
                        (angle5 + this->params_.offsets[4]) *
                            this->params_.rotation_directions[4],
                        (angle6 + this->params_.offsets[5]) *
                            this->params_.rotation_directions[5];
                    ik_solutions.PushBack(joints);
                } else {
                    holistic_motion::utility::LogDebug(
                        "GetIK: There is [nan] data in the joint.");
                }
            }
        }
    }
    // Remove repeated IK, if there are.
    ik_solutions.RemoveRepeatedIK();
    // Finds IK that is within joint limits, removes IK that is out of range.
    ik_solutions.WrapToLimitsNear(this->joint_nodes_, joint_seed);
    // Check the final coordinates after wrapping into the joint limits.
    auto &candidates = ik_solutions.ik_joints;
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [&](const Eigen::VectorXd &candidate) {
            SE3d pose;
            return !GetFK(candidate, pose) || !target_pose.IsApprox(pose, kPoseTolerance);
        }), candidates.end());
    ik_solutions.ik_number = static_cast<int>(candidates.size());
    ik_solutions.success = !candidates.empty();
    // Filter some IK results with joint filter config(each joint)
    ik_solutions.FilterJointSolutions(this->joint_filter_config_);
    // Filter some IK results with robot config(base\elbow\wrist)
    ik_solutions.LimitRobotConfig(this->joint_nodes_, this->robot_config_);

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
    if (detail::SeedAliasesSolutions(ik_solutions, joint_seed)) {
        Eigen::VectorXd stable_seed = joint_seed;
        return GetNearstIK(target_pose, ik_solutions, stable_seed, min_dist);
    }

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
