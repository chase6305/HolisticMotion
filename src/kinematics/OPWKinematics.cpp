#include "holistic_motion/kinematics/OPWKinematics.h"

#include "FrameValidation.h"

#include "SeedAlias.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <typeinfo>

namespace {

constexpr double kDomainTolerance = 1e-10;

}  // namespace

namespace holistic_motion {
namespace robotics {

KinematicsCapabilities OPWKinematics::GetCapabilities() const {
    auto result = KinematicsBase::GetCapabilities();
    result.family = KinematicsFamily::OPW;
    result.default_ik_algorithm = IKAlgorithm::ANALYTIC;
    result.supports_closed_form_ik = true;
    result.supports_concurrent_queries = typeid(*this) == typeid(OPWKinematics);
    return result;
}

bool OPWKinematics::GetFK(const Eigen::VectorXd &target_joint,
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
    const double first_radial = std::sqrt(std::max(0.0, radial_squared)) -
                  this->params_.a1;
    const double height = c[2] - this->params_.c1;
    const double distal_length = std::hypot(this->params_.a2, this->params_.c3);
    if (std::abs(this->params_.c2) < kDomainTolerance ||
        distal_length < kDomainTolerance) {
        return false;
    }

    double angle1[4], angle2[4], angle3[4];
    if (std::abs(c[1]) < Epsilon && std::abs(c[0]) < Epsilon) {
        const double shoulder_seed =
            joint_seed[0] * this->params_.rotation_directions[0] -
            this->params_.offsets[0];
        angle1[0] =
            std::atan2(std::sin(shoulder_seed), std::cos(shoulder_seed));
        angle1[2] = angle1[0] - M_PI;
        holistic_motion::utility::LogWarning(
            "GetIK: There is a singularity of the shoulder.");
    } else {
        angle1[0] =
            std::atan2(c[1], c[0]) -
            std::atan2(this->params_.b, first_radial + this->params_.a1);
        angle1[2] =
            std::atan2(c[1], c[0]) +
            std::atan2(this->params_.b, first_radial + this->params_.a1) - M_PI;
    }
    angle1[1] = angle1[0];
    angle1[3] = angle1[2];

    // Each shoulder has two elbow branches. Derive both arm angles from the
    // same elbow triangle so rounding cannot make them inconsistent.
    const double elbow_offset = std::atan2(this->params_.a2, this->params_.c3);
    for (unsigned shoulder = 0; shoulder < 2; ++shoulder) {
        const double radial = shoulder == 0
                                  ? first_radial
                                  : -first_radial - 2.0 * this->params_.a1;
        const double elbow_cosine =
            std::clamp((radial * radial + height * height -
                        this->params_.c2 * this->params_.c2 -
                        distal_length * distal_length) /
                           (2.0 * this->params_.c2 * distal_length),
                       -1.0, 1.0);
        const double elbow_sine = std::sqrt(
            std::max(0.0, (1.0 - elbow_cosine) * (1.0 + elbow_cosine)));
        const double elbow_angle = std::atan2(elbow_sine, elbow_cosine);
        const double bearing = std::atan2(radial, height);
        const double bend =
            std::atan2(distal_length * elbow_sine,
                       this->params_.c2 + distal_length * elbow_cosine);
        const unsigned first = 2 * shoulder;
        angle2[first] = bearing - bend;
        angle2[first + 1] = bearing + bend;
        angle3[first] = elbow_angle - elbow_offset;
        angle3[first + 1] = -elbow_angle - elbow_offset;
    }

    // Express the target rotation in the frame before the spherical wrist.
    double angle4[8], angle5[8], angle6[8];
    const double wrist_seed_angle =
        joint_seed[3] * this->params_.rotation_directions[3] -
        this->params_.offsets[3];
    const double seed_cosine = std::cos(wrist_seed_angle);
    const double seed_sine = std::sin(wrist_seed_angle);
    const double wrist_seed = std::abs(wrist_seed_angle) <= M_PI
                                  ? wrist_seed_angle
                                  : std::atan2(seed_sine, seed_cosine);
    const double tool_reach = std::abs(this->params_.c4) +
                              this->params_.T_oe_e.GetTranslation().norm() +
                              this->tcp_.GetTranslation().norm();
    // Bound the tool-position effect as well as the angular error.
    const double seed_axis_tolerance =
        kDomainTolerance / std::max(1.0, tool_reach);
    for (unsigned i = 0; i < 4; ++i) {
        const double c1 = std::cos(angle1[i]);
        const double s1 = std::sin(angle1[i]);
        const double c23 = std::cos(angle2[i] + angle3[i]);
        const double s23 = std::sin(angle2[i] + angle3[i]);
        Eigen::Matrix3d arm_rotation;
        // clang-format off
        arm_rotation << c1 * c23, -s1, c1 * s23,
                        s1 * c23,  c1, s1 * s23,
                            -s23, 0.0, c23;
        // clang-format on
        const Eigen::Matrix3d wrist_rotation = arm_rotation.transpose() * R_o;
        const double wrist_sine =
            std::hypot(wrist_rotation(0, 2), wrist_rotation(1, 2));
        // At a singular wrist, the target cannot determine the azimuth.
        // Also retain the seed when its azimuth already fits the target axis.
        const double transverse_error = -seed_sine * wrist_rotation(0, 2) +
                                        seed_cosine * wrist_rotation(1, 2);
        double c4 = seed_cosine;
        double s4 = seed_sine;
        if (std::abs(transverse_error) < seed_axis_tolerance) {
            angle4[i] = wrist_seed;
            angle5[i] = std::atan2(seed_cosine * wrist_rotation(0, 2) +
                                       seed_sine * wrist_rotation(1, 2),
                                   wrist_rotation(2, 2));
            if (wrist_sine < kDomainTolerance) {
                holistic_motion::utility::LogWarning(
                    "GetIK: There is a singularity of the wrist.");
            }
        } else {
            c4 = wrist_rotation(0, 2) / wrist_sine;
            s4 = wrist_rotation(1, 2) / wrist_sine;
            angle4[i] = std::atan2(wrist_rotation(1, 2), wrist_rotation(0, 2));
            angle5[i] = std::atan2(wrist_sine, wrist_rotation(2, 2));
        }
        // Recover angle6 from unit-length rows instead of entries that vanish
        // at a wrist singularity, including both straight and folded wrists.
        angle6[i] =
            std::atan2(-s4 * wrist_rotation(0, 0) + c4 * wrist_rotation(1, 0),
                       -s4 * wrist_rotation(0, 1) + c4 * wrist_rotation(1, 1));
        angle4[i + 4] = angle4[i] + M_PI;
        angle5[i + 4] = -angle5[i];
        angle6[i + 4] = angle6[i] - M_PI;
    }
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
            ik_solutions.PushBack(joints);
        } else {
            holistic_motion::utility::LogDebug("GetIK: There is [nan] data in the joint.");
        }
    }

    // Remove repeated IK, if there are.
    ik_solutions.RemoveRepeatedIK();
    // Finds IK that is within joint limits, removes IK that is out of range.
    ik_solutions.WrapToLimitsNear(this->joint_nodes_, joint_seed);
    // Validate the final coordinates: whole-turn shifts and offsets can round
    // differently when a raw candidate contains a very large seed angle.
    auto &candidates = ik_solutions.ik_joints;
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [&](const Eigen::VectorXd &candidate) {
                           SE3d pose;
                           return !GetFK(candidate, pose) ||
                                  !target_pose.IsApprox(pose, kDomainTolerance);
                       }),
        candidates.end());
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

    holistic_motion::utility::LogDebug("GetIK: Failed to get position IK.");
    return false;
}

bool OPWKinematics::GetNearstIK(const SE3d &target_pose,
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
