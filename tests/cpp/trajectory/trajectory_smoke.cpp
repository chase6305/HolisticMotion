#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>

#include "holistic_motion/trajectory/PathBezierCurve.h"
#include "holistic_motion/trajectory/Polynomial.h"
#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "holistic_motion/trajectory/TrajectoryTrapezoidal.h"

using namespace holistic_motion::robotics;

template <typename Trajectory>
bool DerivativesAreFiniteAndBounded(Trajectory& trajectory,
                                    double limit) {
    constexpr int samples = 2001;
    for (int sample = 0; sample < samples; ++sample) {
        const double t = trajectory.GetDuration() * sample / (samples - 1.0);
        const auto state = trajectory.GetState(t);
        if (!state.position.Coeffs().allFinite()) return false;
        for (const auto& derivative : {
                     state.velocity.Coeffs(), state.acceleration.Coeffs(),
                     state.jerk.Coeffs()}) {
            if (!derivative.allFinite() ||
                derivative.cwiseAbs().maxCoeff() > limit) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    const TrajectoryConstraints null_joint_constraints(
            std::vector<std::shared_ptr<Joint>>(1));
    if (null_joint_constraints.IsValid()) return 1;

    Eigen::Vector4d coefficients;
    coefficients << 1.0, 2.0, 3.0, 4.0;
    auto polynomial = std::make_shared<Polynomial>(coefficients);
    if (polynomial->GetDegree() != 3 ||
        std::abs(polynomial->ComputePolyValueAtS(1.0) - 10.0) > 1e-12 ||
        std::abs(polynomial->ComputePolyValueAtS(1.0, 1) - 20.0) > 1e-12 ||
        std::abs(polynomial->ComputePolyValueAtS(1.0, 2) - 30.0) > 1e-12 ||
        std::abs(polynomial->ComputePolyValueAtS(1.0, 3) - 24.0) > 1e-12) {
        return 1;
    }
    PSpline spline;
    spline.PushBack(polynomial, 2.0);
    const double query_time = 0.75;
    if (std::abs(spline.ComputeValueAtS(query_time) -
                 polynomial->ComputePolyValueAtS(query_time)) > 1e-12 ||
        query_time != 0.75) {
        return 2;
    }
    const auto jet = spline.ComputeJetAtS(query_time);
    for (unsigned order = 0; order < jet.size(); ++order) {
        if (std::abs(jet[order] -
                     polynomial->ComputePolyValueAtS(query_time, order)) >
            1e-12) {
            return 2;
        }
    }

    const std::vector<Rn<double, 7>> empty_waypoints;
    auto invalid_path = std::make_shared<PathBezierCurve<Rn<double, 7>>>(
            empty_waypoints, 5, false, 0.005);
    if (invalid_path->IsValid() || invalid_path->GetType() != PathType::NoBlend ||
        invalid_path->GetNumOfPathSegments() != 0) {
        return 3;
    }
    const auto invalid_constraints = std::make_shared<TrajectoryConstraints>(
            Eigen::VectorXd::Ones(7), Eigen::VectorXd::Ones(7),
            Eigen::VectorXd::Ones(7));
    TrajectoryDoubleS<Rn<double, 7>> invalid_trajectory(
            invalid_path, invalid_constraints);
    if (invalid_trajectory.IsValid() || invalid_trajectory.GetDuration() != 0.0) {
        return 4;
    }

    using JointGroup = Rn<double, 7>;
    std::array<JointGroup, 3> degenerate_controls;
    for (auto& control : degenerate_controls) control.Coeffs().setZero();
    const PathSegBezierCurve5th<JointGroup> degenerate_segment(
            degenerate_controls, 0.0, 1.0, 0.0, 1.0, 0.0, false);
    if (degenerate_segment.IsValid()) return 5;

    std::vector<JointGroup> joints(4);
    for (auto& waypoint : joints) waypoint.Coeffs().setZero();
    joints[1].Coeffs().template head<2>() << 0.2, -0.1;
    joints[2].Coeffs().template head<2>() << 0.4, 0.1;
    joints[3].Coeffs().template head<2>() << 0.6, 0.0;
    const Eigen::VectorXd joint_limits = Eigen::VectorXd::Ones(7);
    auto joint_path = std::make_shared<PathBezierCurve<JointGroup>>(
            joints, 5, false, 0.005);
    auto joint_constraints = std::make_shared<TrajectoryConstraints>(
            joint_limits, joint_limits, joint_limits);
    TrajectoryDoubleS<JointGroup> joint_trajectory(
            joint_path, joint_constraints);
    if (!joint_trajectory.IsValid() ||
        !DerivativesAreFiniteAndBounded(joint_trajectory, 1.0)) {
        return 5;
    }
    const auto constraint_report = joint_trajectory.GetConstraintReport();
    if (!constraint_report.within_limits ||
        !constraint_report.velocity_continuous ||
        !constraint_report.acceleration_continuous ||
        constraint_report.maximum_utilization > 1.0 + 1e-12 ||
        constraint_report.peak_velocity.size() != 7) {
        return 5;
    }
    try {
        joint_trajectory.GetConstraintReport(1);
        return 5;
    } catch (const std::invalid_argument&) {
    }
    TrajectoryDoubleS<JointGroup> invalid_velocity_trajectory(
            joint_path, joint_constraints, -0.1, 0.0);
    if (invalid_velocity_trajectory.IsValid()) return 6;
    try {
        joint_trajectory.GetPosition(
                std::numeric_limits<double>::quiet_NaN());
        return 7;
    } catch (const std::invalid_argument&) {
    }
    const double duration = joint_trajectory.GetDuration();
    if (!joint_trajectory.SetMinimumDuration(2.0 * duration) ||
        std::abs(joint_trajectory.GetDuration() - 2.0 * duration) > 1e-9) {
        return 8;
    }
    TrajectoryTrapezoidal<JointGroup> trapezium_trajectory(
            joint_path, joint_constraints);
    if (!trapezium_trajectory.IsValid() ||
        !DerivativesAreFiniteAndBounded(trapezium_trajectory, 1.0)) {
        return 9;
    }

    std::vector<SE3d> poses(2);
    Eigen::Matrix4d end_pose = Eigen::Matrix4d::Identity();
    end_pose(0, 3) = 0.2;
    poses[1] = SE3d(end_pose);
    const Eigen::VectorXd cartesian_limits = Eigen::VectorXd::Ones(6);
    auto cartesian_path = std::make_shared<PathBezierCurve<SE3d>>(
            poses, 5, true, 0.0);
    auto cartesian_constraints = std::make_shared<TrajectoryConstraints>(
            cartesian_limits, cartesian_limits, cartesian_limits);
    TrajectoryDoubleS<SE3d> cartesian_trajectory(
            cartesian_path, cartesian_constraints);
    return cartesian_trajectory.IsValid() &&
                           DerivativesAreFiniteAndBounded(
                                   cartesian_trajectory, 1.0)
                   ? 0
                   : 10;
}
