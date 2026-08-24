#include "SRSKinematicsInternal.h"

#include <algorithm>
#include <cmath>

namespace holistic_motion::robotics::srs_detail {

std::pair<Eigen::Vector3d, double> FitAxisIntersection(
        const std::array<Eigen::Vector3d, 3>& points,
        const std::array<Eigen::Vector3d, 3>& axes) {
    Eigen::Matrix3d system = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d direction = axes[i].normalized();
        const Eigen::Matrix3d projector =
                Eigen::Matrix3d::Identity() - direction * direction.transpose();
        system += projector;
        rhs += projector * points[i];
    }
    const Eigen::Vector3d center =
            system.completeOrthogonalDecomposition().solve(rhs);
    double squared_error = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d direction = axes[i].normalized();
        squared_error +=
                ((center - points[i]).cross(direction)).squaredNorm();
    }
    return {center, std::sqrt(squared_error / points.size())};
}

bool ChainCentres(const SRSKinematics& solver,
                  const Eigen::VectorXd& joints,
                  const Eigen::Vector3d& shoulder,
                  Eigen::Vector3d& elbow,
                  Eigen::Vector3d& wrist,
                  Eigen::Vector3d& elbow_axis) {
    std::vector<SE3d> poses;
    if (!solver.GetAllFK(joints, poses) || poses.size() < 8) return false;
    const auto nodes = solver.GetJointNode();
    elbow_axis = poses[3].GetRotation() * nodes[3].axis;
    if (elbow_axis.norm() < 1e-12) return false;
    elbow_axis.normalize();
    const Eigen::Vector3d elbow_origin = poses[3].GetTranslation();
    elbow = elbow_origin + elbow_axis * elbow_axis.dot(shoulder - elbow_origin);
    std::array<Eigen::Vector3d, 3> wrist_points;
    std::array<Eigen::Vector3d, 3> wrist_axes;
    for (int index = 4; index <= 6; ++index) {
        wrist_points[static_cast<std::size_t>(index - 4)] =
                poses[index].GetTranslation();
        wrist_axes[static_cast<std::size_t>(index - 4)] =
                poses[index].GetRotation() * nodes[index].axis;
    }
    wrist = FitAxisIntersection(wrist_points, wrist_axes).first;
    return elbow.allFinite() && wrist.allFinite();
}

namespace {

double SignedElbowAngle(const SRSKinematics& solver,
                        const Eigen::VectorXd& joints,
                        const Eigen::Vector3d& shoulder) {
    Eigen::Vector3d elbow, wrist, axis;
    if (!ChainCentres(solver, joints, shoulder, elbow, wrist, axis)) return 0.0;
    const Eigen::Vector3d upper = (elbow - shoulder).normalized();
    const Eigen::Vector3d lower = (wrist - elbow).normalized();
    return std::atan2(axis.dot(upper.cross(lower)), upper.dot(lower));
}

}  // namespace

}  // namespace holistic_motion::robotics::srs_detail

namespace holistic_motion::robotics {

SRSGeometryAnalysis SRSKinematics::AnalyzeGeometry(double axis_tolerance) const {
    if (geometry_analysis_cache_ &&
        geometry_analysis_cache_->first == axis_tolerance) {
        return geometry_analysis_cache_->second;
    }
    SRSGeometryAnalysis analysis;
    analysis.structurally_compatible = IsCompatible();
    if (!analysis.structurally_compatible ||
        !std::isfinite(axis_tolerance) || axis_tolerance <= 0.0) return analysis;
    std::array<Eigen::Vector3d, 7> points;
    std::array<Eigen::Vector3d, 7> axes;
    SE3d pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    for (std::size_t i = 0; i < 7; ++i) {
        pose = pose * joint_nodes_[i].origin_pose;
        points[i] = pose.GetTranslation();
        axes[i] = pose.GetRotation() * joint_nodes_[i].axis;
        if (!points[i].allFinite() || !axes[i].allFinite() ||
            axes[i].norm() < 1e-12) return analysis;
    }
    std::array<Eigen::Vector3d, 3> shoulder_points{
            points[0], points[1], points[2]};
    std::array<Eigen::Vector3d, 3> shoulder_axes{
            axes[0], axes[1], axes[2]};
    std::tie(analysis.shoulder_center, analysis.shoulder_axis_residual) =
            srs_detail::FitAxisIntersection(shoulder_points, shoulder_axes);
    analysis.shoulder_orthogonality_residual = std::max(
            std::abs(axes[0].normalized().dot(axes[1].normalized())),
            std::abs(axes[1].normalized().dot(axes[2].normalized())));
    std::array<Eigen::Vector3d, 3> wrist_points{
            points[4], points[5], points[6]};
    std::array<Eigen::Vector3d, 3> wrist_axes{
            axes[4], axes[5], axes[6]};
    std::tie(analysis.wrist_center, analysis.wrist_axis_residual) =
            srs_detail::FitAxisIntersection(wrist_points, wrist_axes);
    analysis.wrist_orthogonality_residual = std::max(
            std::abs(axes[4].normalized().dot(axes[5].normalized())),
            std::abs(axes[5].normalized().dot(axes[6].normalized())));
    const Eigen::Vector3d elbow_axis = axes[3].normalized();
    const Eigen::Vector3d shoulder_to_elbow = points[3] + elbow_axis *
            elbow_axis.dot(analysis.shoulder_center - points[3]);
    const Eigen::Vector3d wrist_to_elbow = points[3] + elbow_axis *
            elbow_axis.dot(analysis.wrist_center - points[3]);
    analysis.upper_arm_length =
            (shoulder_to_elbow - analysis.shoulder_center).norm();
    analysis.forearm_length =
            (wrist_to_elbow - analysis.wrist_center).norm();
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(7);
    analysis.elbow_angle_offset = srs_detail::SignedElbowAngle(
            *this, zero, analysis.shoulder_center);
    Eigen::VectorXd perturbed = zero;
    perturbed[3] = 1e-4;
    const double perturbed_angle = srs_detail::SignedElbowAngle(
            *this, perturbed, analysis.shoulder_center);
    const double derivative = std::remainder(
            perturbed_angle - analysis.elbow_angle_offset, 2.0 * M_PI) / 1e-4;
    analysis.elbow_angle_direction = derivative < 0.0 ? -1.0 : 1.0;
    analysis.closed_form_compatible =
            analysis.shoulder_axis_residual <= axis_tolerance &&
            analysis.wrist_axis_residual <= axis_tolerance &&
            analysis.shoulder_orthogonality_residual <= axis_tolerance &&
            analysis.wrist_orthogonality_residual <= axis_tolerance &&
            analysis.upper_arm_length > axis_tolerance &&
            analysis.forearm_length > axis_tolerance;
    geometry_analysis_cache_ = std::make_pair(axis_tolerance, analysis);
    return analysis;
}

}  // namespace holistic_motion::robotics
