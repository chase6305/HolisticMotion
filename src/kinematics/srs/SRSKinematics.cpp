#include "holistic_motion/kinematics/srs/SRSKinematics.h"
#include "SRSKinematicsInternal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace holistic_motion::robotics {

bool SRSKinematics::IsCompatible() const noexcept {
    if (GetDOF() != 7 || joint_nodes_.size() != 8) return false;
    return std::all_of(joint_nodes_.begin(), joint_nodes_.begin() + 7,
                       [](const JointNode& node) {
                           return node.joint_type == JointType::REVOLUTE;
                       });
}

namespace {

int Direction(double value) { return value < 0.0 ? -1 : 1; }

Eigen::Matrix3d RotationX(double angle) {
    return Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
}

Eigen::Matrix3d RotationZ(double angle) {
    return Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

Eigen::Matrix4d Mdh(double alpha, double distance, double theta) {
    const double st = std::sin(theta), ct = std::cos(theta);
    const double sa = std::sin(alpha), ca = std::cos(alpha);
    Eigen::Matrix4d result;
    result << ct, -st, 0.0, 0.0,
              st * ca, ct * ca, -sa, -distance * sa,
              st * sa, ct * sa, ca, distance * ca,
              0.0, 0.0, 0.0, 1.0;
    return result;
}

struct CanonicalSRSModel {
    Eigen::Matrix4d base{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d tool{Eigen::Matrix4d::Identity()};
    Eigen::Matrix<double, 7, 1> offsets;
    std::array<int, 7> directions;
    double upper{0.0};
    double lower{0.0};
};

bool BuildCanonicalModel(const SRSKinematics& solver,
                         CanonicalSRSModel& model) {
    const auto geometry = solver.AnalyzeGeometry();
    if (!geometry.closed_form_compatible) return false;
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(7);
    Eigen::MatrixXd jacobian;
    if (!solver.GetJacobian(zero, jacobian)) return false;
    std::array<Eigen::Vector3d, 7> observed;
    for (int i = 0; i < 7; ++i) observed[i] = jacobian.block<3, 1>(3, i).normalized();
    Eigen::Vector3d z = observed[0];
    Eigen::Vector3d y = observed[1] - z * z.dot(observed[1]);
    if (y.norm() < 1e-8) return false;
    y.normalize();
    Eigen::Vector3d x = y.cross(z).normalized();
    y = z.cross(x).normalized();
    model.base.block<3, 3>(0, 0) << x, y, z;
    model.base.block<3, 1>(0, 3) = geometry.shoulder_center;

    static const std::array<double, 7> alpha{
            0.0, -M_PI_2, M_PI_2, M_PI_2, -M_PI_2, -M_PI_2, M_PI_2};
    Eigen::Matrix3d rotation = model.base.block<3, 3>(0, 0);
    for (int i = 0; i < 7; ++i) {
        const Eigen::Matrix3d origin_rotation = rotation * RotationX(alpha[i]);
        const Eigen::Vector3d canonical_axis =
                origin_rotation * Eigen::Vector3d::UnitZ();
        model.directions[i] = canonical_axis.dot(observed[i]) < 0.0 ? -1 : 1;
        double offset = 0.0;
        if (i < 6) {
            const Eigen::Vector3d basis =
                    RotationX(alpha[i + 1]) * Eigen::Vector3d::UnitZ();
            double best_score = std::numeric_limits<double>::infinity();
            for (int next_direction : {-1, 1}) {
                const Eigen::Vector3d desired = origin_rotation.transpose() *
                        (next_direction * observed[i + 1]);
                const double candidate = std::atan2(
                        basis.x() * desired.y() - basis.y() * desired.x(),
                        basis.x() * desired.x() + basis.y() * desired.y());
                const double score = std::abs(std::remainder(candidate, 2.0 * M_PI));
                if (score < best_score) {
                    best_score = score;
                    offset = candidate;
                }
            }
        }
        model.offsets[i] = offset;
        rotation = origin_rotation * RotationZ(offset);
    }
    // A laterally offset elbow can have a non-zero geometric bend even when
    // the URDF joint value is zero. Map that bend into the canonical q4 and
    // recalibrate the downstream wrist gauges around the adjusted elbow.
    const double elbow_scale = static_cast<double>(model.directions[3]) /
                               geometry.elbow_angle_direction;
    const double canonical_elbow_offset =
            elbow_scale * geometry.elbow_angle_offset;
    if (std::abs(canonical_elbow_offset - model.offsets[3]) > 1e-10) {
        model.offsets[3] = canonical_elbow_offset;
        rotation = model.base.block<3, 3>(0, 0);
        for (int i = 0; i <= 3; ++i) {
            rotation = rotation * RotationX(alpha[i]) * RotationZ(model.offsets[i]);
        }
        for (int i = 4; i < 7; ++i) {
            const Eigen::Matrix3d origin_rotation = rotation * RotationX(alpha[i]);
            const Eigen::Vector3d canonical_axis =
                    origin_rotation * Eigen::Vector3d::UnitZ();
            model.directions[i] = canonical_axis.dot(observed[i]) < 0.0 ? -1 : 1;
            double offset = 0.0;
            if (i < 6) {
                const Eigen::Vector3d basis =
                        RotationX(alpha[i + 1]) * Eigen::Vector3d::UnitZ();
                double best_score = std::numeric_limits<double>::infinity();
                for (int next_direction : {-1, 1}) {
                    const Eigen::Vector3d desired = origin_rotation.transpose() *
                            (next_direction * observed[i + 1]);
                    const double candidate = std::atan2(
                            basis.x() * desired.y() - basis.y() * desired.x(),
                            basis.x() * desired.x() + basis.y() * desired.y());
                    const double score = std::abs(
                            std::remainder(candidate, 2.0 * M_PI));
                    if (score < best_score) {
                        best_score = score;
                        offset = candidate;
                    }
                }
            }
            model.offsets[i] = offset;
            rotation = origin_rotation * RotationZ(offset);
        }
    }
    model.upper = geometry.upper_arm_length;
    model.lower = geometry.forearm_length;
    Eigen::Matrix4d canonical_zero = Eigen::Matrix4d::Identity();
    static const std::array<double, 7> distances{
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    for (int i = 0; i < 7; ++i) {
        const double distance = distances[i] == 1.0
                ? (i == 2 ? model.upper : model.lower) : 0.0;
        canonical_zero *= Mdh(alpha[i], distance, model.offsets[i]);
    }
    SE3d actual_zero;
    if (!solver.GetFK(zero, actual_zero)) return false;
    model.tool = canonical_zero.inverse() * model.base.inverse() *
                 actual_zero.GetTransform();
    return model.tool.allFinite();
}

bool WrapToJointLimits(double value, const JointNode& node,
                       double reference, double& wrapped) {
    double best_distance = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int turns = -2; turns <= 2; ++turns) {
        const double candidate = value + turns * 2.0 * M_PI;
        if (candidate >= node.lower_limit - 1e-9 &&
            candidate <= node.upper_limit + 1e-9) {
            const double distance = std::abs(candidate - reference);
            if (distance < best_distance) {
                best_distance = distance;
                wrapped = candidate;
                found = true;
            }
        }
    }
    return found;
}

double WeightedDistance(const Eigen::VectorXd& lhs,
                        const Eigen::VectorXd& rhs) {
    // The elbow redundancy itself should not dominate branch selection.
    static const std::array<double, 7> weights{2.0, 2.0, 2.0, 0.0,
                                               1.0, 1.0, 1.0};
    double result = 0.0;
    for (Eigen::Index i = 0; i < lhs.size(); ++i) {
        const double delta = std::remainder(lhs[i] - rhs[i], 2.0 * M_PI);
        result += weights[static_cast<std::size_t>(i)] * delta * delta;
    }
    return result;
}

}  // namespace

SRSConfiguration SRSKinematics::GetConfiguration(
        const Eigen::VectorXd& joints) const {
    if (!IsCompatible() || joints.size() != 7 || !joints.allFinite()) {
        return {};
    }
    CanonicalSRSModel model;
    if (!BuildCanonicalModel(*this, model)) {
        return {Direction(joints[1]), Direction(joints[3]),
                Direction(joints[5]), joints[2]};
    }
    Eigen::Matrix<double, 7, 1> model_joints;
    for (int i = 0; i < 7; ++i) {
        model_joints[i] = model.directions[i] * joints[i] + model.offsets[i];
    }
    return {Direction(model_joints[1]), Direction(model_joints[3]),
            Direction(model_joints[5]), GetArmAngle(joints)};
}

double SRSKinematics::GetArmAngle(const Eigen::VectorXd& joints) const {
    CanonicalSRSModel model;
    if (joints.size() != 7 || !joints.allFinite() ||
        !BuildCanonicalModel(*this, model)) return 0.0;
    Eigen::Vector3d elbow, wrist, elbow_axis;
    const auto geometry = AnalyzeGeometry();
    if (!srs_detail::ChainCentres(*this, joints, geometry.shoulder_center,
                      elbow, wrist, elbow_axis)) return 0.0;
    const Eigen::Matrix4d inverse_base = model.base.inverse();
    const Eigen::Vector3d elbow_model =
            (inverse_base * elbow.homogeneous()).head<3>();
    const Eigen::Vector3d wrist_model =
            (inverse_base * wrist.homogeneous()).head<3>();
    const double distance = wrist_model.norm();
    if (distance < 1e-10) return 0.0;
    const Eigen::Vector3d direction = wrist_model / distance;
    const double elbow_sign = Direction(
            model.directions[3] * joints[3] + model.offsets[3]);
    const double cosine = clamp(
            (model.upper * model.upper + distance * distance -
             model.lower * model.lower) / (2.0 * model.upper * distance),
            -1.0, 1.0);
    const double q1 = std::atan2(wrist_model.y(), wrist_model.x());
    const double q2 = std::atan2(wrist_model.head<2>().norm(), wrist_model.z()) +
                      elbow_sign * std::acos(cosine);
    Eigen::Matrix4d reference = Eigen::Matrix4d::Identity();
    static const std::array<double, 3> alpha{0.0, -M_PI_2, M_PI_2};
    const std::array<double, 3> qref{q1, q2, 0.0};
    const std::array<double, 3> distances{0.0, 0.0, model.upper};
    for (int i = 0; i < 3; ++i) reference *= Mdh(alpha[i], distances[i], qref[i]);
    const Eigen::Vector3d center = direction *
            ((model.upper * model.upper - model.lower * model.lower +
              distance * distance) / (2.0 * distance));
    Eigen::Vector3d reference_radial = reference.block<3, 1>(0, 3) - center;
    Eigen::Vector3d actual_radial = elbow_model - center;
    if (reference_radial.norm() < 1e-10 || actual_radial.norm() < 1e-10) return 0.0;
    reference_radial.normalize();
    actual_radial.normalize();
    return std::atan2(direction.dot(reference_radial.cross(actual_radial)),
                      reference_radial.dot(actual_radial));
}

bool SRSKinematics::SolveConfiguration(
        const SE3d& target,
        const SRSConfiguration& configuration,
        const Eigen::VectorXd& seed,
        Eigen::VectorXd& solution) const {
    if (!IsCompatible() || seed.size() != 7 || !seed.allFinite() ||
        !std::isfinite(configuration.redundancy)) {
        return false;
    }
    Eigen::VectorXd analytic_candidate;
    if (GetAnalyticSolution(target, configuration, seed, analytic_candidate)) {
        solution = analytic_candidate;
        return true;
    }
    Eigen::VectorXd branch_seed;
    const bool has_full_candidate = analytic_candidate.size() == 7 &&
                                    analytic_candidate.allFinite();
    if (has_full_candidate) branch_seed = analytic_candidate;
    const bool has_analytic_elbow = has_full_candidate || GetAnalyticElbowSeed(
            target, configuration, seed, branch_seed);
    if (!has_analytic_elbow) {
        // Non-ideal 7R chains retain the configuration-seeded numerical path.
        branch_seed = seed;
    }
    const std::array<int, 3> indices{1, 3, 5};
        const std::array<int, 3> directions{
            configuration.shoulder, configuration.elbow,
            configuration.wrist};
    if (!has_full_candidate) {
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const int index = indices[i];
            if (has_analytic_elbow && index == 3) continue;
            const double magnitude = std::max(std::abs(branch_seed[index]), 0.35);
            branch_seed[index] = directions[i] < 0 ? -magnitude : magnitude;
        }
        branch_seed[2] = configuration.redundancy;
    }
    for (Eigen::Index i = 0; i < branch_seed.size(); ++i) {
        branch_seed[i] = clamp(branch_seed[i], joint_nodes_[i].lower_limit,
                               joint_nodes_[i].upper_limit);
    }
    IkRtn result;
    std::vector<double> distances;
    if (!NumericalKinematics::GetIK(target, result, branch_seed, distances) ||
        result.ik_joints.empty()) {
        return false;
    }
    solution = result.ik_joints.front();
    const auto solved_configuration = GetConfiguration(solution);
    return solved_configuration.shoulder == Direction(configuration.shoulder) &&
           solved_configuration.elbow == Direction(configuration.elbow) &&
           solved_configuration.wrist == Direction(configuration.wrist);
}

bool SRSKinematics::GetAnalyticElbowSeed(
        const SE3d& target,
        const SRSConfiguration& configuration,
        const Eigen::VectorXd& seed,
        Eigen::VectorXd& analytic_seed) const {
    const auto geometry = AnalyzeGeometry();
    if (!geometry.closed_form_compatible || seed.size() != 7 ||
        !seed.allFinite() || !target.GetTransform().allFinite()) {
        return false;
    }
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(7);
    SE3d zero_pose;
    if (!GetFK(zero, zero_pose)) return false;
    const Eigen::Vector4d wrist_homogeneous(
            geometry.wrist_center.x(), geometry.wrist_center.y(),
            geometry.wrist_center.z(), 1.0);
    const Eigen::Vector4d wrist_in_tool =
            zero_pose.GetTransform().inverse() * wrist_homogeneous;
    const Eigen::Vector3d target_wrist =
            (target.GetTransform() * wrist_in_tool).head<3>();
    const double distance =
            (target_wrist - geometry.shoulder_center).norm();
    const double upper = geometry.upper_arm_length;
    const double lower = geometry.forearm_length;
    if (distance > upper + lower + 1e-8 ||
        distance < std::abs(upper - lower) - 1e-8) {
        return false;
    }
    const double cosine = clamp(
            (distance * distance - upper * upper - lower * lower) /
                    (2.0 * upper * lower),
            -1.0, 1.0);
    const double model_elbow =
            Direction(static_cast<double>(configuration.elbow)) *
            std::acos(cosine);
    analytic_seed = seed;
    analytic_seed[3] =
            (model_elbow - geometry.elbow_angle_offset) /
            geometry.elbow_angle_direction;
    analytic_seed[3] = clamp(analytic_seed[3],
                             joint_nodes_[3].lower_limit,
                             joint_nodes_[3].upper_limit);
    return analytic_seed.allFinite();
}

bool SRSKinematics::GetAnalyticSolution(
        const SE3d& target,
        const SRSConfiguration& configuration,
        const Eigen::VectorXd& seed,
        Eigen::VectorXd& solution) const {
    CanonicalSRSModel model;
    if (seed.size() != 7 || !seed.allFinite() ||
        !target.GetTransform().allFinite() ||
        !BuildCanonicalModel(*this, model)) return false;
    const Eigen::Matrix4d pose = model.base.inverse() * target.GetTransform() *
                                 model.tool.inverse();
    const Eigen::Vector3d wrist = pose.block<3, 1>(0, 3);
    const Eigen::Matrix3d target_rotation = pose.block<3, 3>(0, 0);
    const double distance = wrist.norm();
    if (distance < 1e-10 || distance > model.upper + model.lower + 1e-8 ||
        distance < std::abs(model.upper - model.lower) - 1e-8) return false;
    const double elbow_sign = Direction(static_cast<double>(configuration.elbow));
    const double elbow_cosine = clamp(
            (distance * distance - model.upper * model.upper -
             model.lower * model.lower) / (2.0 * model.upper * model.lower),
            -1.0, 1.0);
    const double q4 = elbow_sign * std::acos(elbow_cosine);
    const double shoulder_cosine = clamp(
            (model.upper * model.upper + distance * distance -
             model.lower * model.lower) / (2.0 * model.upper * distance),
            -1.0, 1.0);
    const double q1_reference = std::atan2(wrist.y(), wrist.x());
    const double q2_reference =
            std::atan2(wrist.head<2>().norm(), wrist.z()) +
            elbow_sign * std::acos(shoulder_cosine);
    Eigen::Matrix4d shoulder_reference = Eigen::Matrix4d::Identity();
    shoulder_reference *= Mdh(0.0, 0.0, q1_reference);
    shoulder_reference *= Mdh(-M_PI_2, 0.0, q2_reference);
    shoulder_reference *= Mdh(M_PI_2, model.upper, 0.0);
    const Eigen::Matrix3d reference_rotation =
            shoulder_reference.block<3, 3>(0, 0);
    const Eigen::Vector3d axis = wrist / distance;
    Eigen::Matrix3d skew;
    skew << 0.0, -axis.z(), axis.y(),
            axis.z(), 0.0, -axis.x(),
            -axis.y(), axis.x(), 0.0;
    const Eigen::Matrix3d as = skew * reference_rotation;
    const Eigen::Matrix3d bs = -skew * skew * reference_rotation;
    const Eigen::Matrix3d cs = axis * axis.transpose() * reference_rotation;
    const double sine = std::sin(configuration.redundancy);
    const double cosine = std::cos(configuration.redundancy);
    const Eigen::Matrix3d shoulder_rotation = sine * as + cosine * bs + cs;
    const Eigen::Matrix3d elbow_rotation =
            Mdh(M_PI_2, 0.0, q4).block<3, 3>(0, 0);
    const Eigen::Matrix3d aw =
            elbow_rotation.transpose() * as.transpose() * target_rotation;
    const Eigen::Matrix3d bw =
            elbow_rotation.transpose() * bs.transpose() * target_rotation;
    const Eigen::Matrix3d cw =
            elbow_rotation.transpose() * cs.transpose() * target_rotation;
    const Eigen::Matrix3d wrist_rotation = sine * aw + cosine * bw + cw;

    Eigen::Matrix<double, 7, 1> model_solution;
    const double shoulder_sign =
            Direction(static_cast<double>(configuration.shoulder));
    model_solution[1] = shoulder_sign *
            std::acos(clamp(shoulder_rotation(2, 2), -1.0, 1.0));
    if (std::abs(std::sin(model_solution[1])) < 1e-8) {
        model_solution[0] = model.directions[0] * seed[0] + model.offsets[0];
        model_solution[2] = std::atan2(
                shoulder_rotation(1, 0), shoulder_rotation(0, 0)) -
                model_solution[0];
    } else {
        model_solution[0] = std::atan2(
                shoulder_sign * shoulder_rotation(1, 2),
                shoulder_sign * shoulder_rotation(0, 2));
        model_solution[2] = std::atan2(
                shoulder_sign * shoulder_rotation(2, 1),
                shoulder_sign * -shoulder_rotation(2, 0));
    }
    model_solution[3] = q4;
    const double wrist_sign = Direction(static_cast<double>(configuration.wrist));
    model_solution[5] = wrist_sign *
            std::acos(clamp(wrist_rotation(1, 2), -1.0, 1.0));
    if (std::abs(std::sin(model_solution[5])) < 1e-8) {
        model_solution[4] = model.directions[4] * seed[4] + model.offsets[4];
        model_solution[6] = std::atan2(
                -wrist_rotation(2, 0), wrist_rotation(0, 0)) -
                model_solution[4];
    } else {
        model_solution[4] = std::atan2(
                wrist_sign * -wrist_rotation(2, 2),
                wrist_sign * wrist_rotation(0, 2));
        model_solution[6] = std::atan2(
                wrist_sign * wrist_rotation(1, 1),
                wrist_sign * -wrist_rotation(1, 0));
    }

    solution.resize(7);
    const auto nodes = GetJointNode();
    for (int i = 0; i < 7; ++i) {
        const double user_value =
                (model_solution[i] - model.offsets[i]) / model.directions[i];
        if (!WrapToJointLimits(user_value, nodes[i], seed[i], solution[i])) {
            return false;
        }
    }
    SE3d actual;
    if (!GetFK(solution, actual)) return false;
    const Eigen::Matrix4d delta = actual.GetTransform().inverse() *
                                  target.GetTransform();
    const double position_error = delta.block<3, 1>(0, 3).norm();
    const double angle_error = Eigen::AngleAxisd(
            delta.block<3, 3>(0, 0)).angle();
    return position_error < 1e-5 && angle_error < 1e-5;
}

bool SRSKinematics::Solve(const SE3d& target,
                          const Eigen::VectorXd& seed,
                          SRSSolveMethod method,
                          std::vector<Eigen::VectorXd>& solutions) const {
    solutions.clear();
    if (!IsCompatible() || seed.size() != 7 || !seed.allFinite()) return false;

    auto add = [&](const Eigen::VectorXd& candidate) {
        const bool duplicate = std::any_of(
                solutions.begin(), solutions.end(), [&](const auto& existing) {
                    return (existing - candidate).norm() < 1e-4;
                });
        if (!duplicate) solutions.push_back(candidate);
    };
    auto solve_seed = [&](Eigen::VectorXd candidate_seed) {
        IkRtn result;
        std::vector<double> distances;
        if (NumericalKinematics::GetIK(target, result, candidate_seed,
                                       distances)) {
            for (const auto& candidate : result.ik_joints) add(candidate);
        }
    };
    const SRSConfiguration seed_configuration = GetConfiguration(seed);

    if (method == SRSSolveMethod::SEEDED_NUMERICAL) {
        solve_seed(seed);
    } else if (method == SRSSolveMethod::CONFIGURATION) {
        Eigen::VectorXd solution;
        if (SolveConfiguration(target, seed_configuration, seed,
                               solution)) {
            add(solution);
        }
    } else if (method == SRSSolveMethod::ALL_CONFIGURATIONS) {
        // Preserve the exact seed branch first, then enumerate S/E/W signs.
        solve_seed(seed);
        for (int shoulder : {-1, 1}) {
            for (int elbow : {-1, 1}) {
                for (int wrist : {-1, 1}) {
                    Eigen::VectorXd solution;
                    const SRSConfiguration configuration{
                            shoulder, elbow, wrist,
                            seed_configuration.redundancy};
                    if (SolveConfiguration(target, configuration, seed,
                                           solution)) {
                        add(solution);
                    }
                }
            }
        }
    } else {
        // Search the geometric arm angle, not joint 3.  This preserves the
        // seed configuration while expanding through nearby redundancy values.
        Eigen::VectorXd solution;
        if (SolveConfiguration(target, seed_configuration, seed, solution)) {
            add(solution);
        }
        constexpr double step = M_PI / 18.0;
        for (int layer = 1; layer <= 18 && solutions.empty(); ++layer) {
            for (double sign : {-1.0, 1.0}) {
                SRSConfiguration candidate = seed_configuration;
                candidate.redundancy = std::remainder(
                        seed_configuration.redundancy + sign * layer * step,
                        2.0 * M_PI);
                if (SolveConfiguration(target, candidate, seed, solution)) {
                    add(solution);
                }
            }
        }
    }

    std::sort(solutions.begin(), solutions.end(), [&](const auto& lhs,
                                                       const auto& rhs) {
        return WeightedDistance(lhs, seed) < WeightedDistance(rhs, seed);
    });
    return !solutions.empty();
}

}  // namespace holistic_motion::robotics
