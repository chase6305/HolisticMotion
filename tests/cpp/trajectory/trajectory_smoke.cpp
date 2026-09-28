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

template <int DoF>
bool LowDimensionalCartesianBlendIsValid() {
    using Group = Rn<double, DoF>;
    std::array<Group, 3> controls;
    for (int i = 0; i < 3; ++i) {
        controls[i].Coeffs().setZero();
        controls[i].Coeffs()[0] = 0.1 * i;
    }
    const PathSegBezierCurve5th<Group> segment(
            controls, 0.0, 1.0, 0.0, 1.0, 0.0, true);
    if (!segment.IsValid() || std::abs(segment.GetLength() - 0.2) > 1e-12)
        return false;
    for (int i = 0; i <= 10; ++i) {
        const double s = segment.GetLength() * i / 10.0;
        const auto position = segment.GetConfig(s).Coeffs().eval();
        if (!position.allFinite() || std::abs(position[0] - s) > 1e-12)
            return false;
    }
    return true;
}

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

bool PathQueriesRespectValidation() {
    using Group = Rn<double, 2>;
    std::vector<Group> points(3);
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 1.0, 0.0;
    points[2].Coeffs() << 1.0, 1.0;
    PathBezierCurve<Group> path(points, 5, false, 0.0);
    if (!path.IsValid() || path.GetNumOfPathSegments() != 2) return false;
    const auto query = [](const PathBase<Group>& target, int kind, double s) {
        switch (kind) {
            case 0:
                target.GetPathSegmentAtS(s);
                break;
            case 1:
                target.GetConfig(s);
                break;
            case 2:
                target.GetTangent(s);
                break;
            case 3:
                target.GetCurvature(s);
                break;
            case 4:
                target.GetTorsion(s);
                break;
        }
    };
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
        for (int kind = 0; kind < 5; ++kind) {
            try {
                query(path, kind, invalid);
                return false;
            } catch (const std::invalid_argument&) {
            }
        }
    }
    const auto first = path.GetPathSegmentByIndex(0);
    const auto last = path.GetPathSegmentByIndex(1);
    if (path.GetPathSegmentAtS(-1.0) != first ||
        path.GetPathSegmentAtS(3.0) != last ||
        path.GetPathSegmentAtS(std::nextafter(1.0, 0.0)) != first ||
        path.GetPathSegmentAtS(1.0) != last ||
        path.GetPathSegmentByIndex(-1) != first ||
        path.GetPathSegmentByIndex(99) != last ||
        (path.GetConfig(-1.0) - points.front()).Coeffs().norm() != 0.0 ||
        (path.GetConfig(3.0) - points.back()).Coeffs().norm() != 0.0 ||
        path.GetTangent(1.0)[0] != 0.0 || path.GetTangent(1.0)[1] != 1.0)
        return false;

    struct InvalidatedPath : PathBezierCurve<Group> {
        explicit InvalidatedPath(const std::vector<Group>& points)
            : PathBezierCurve(points, 5, false, 0.0) {
            this->valid_ = false;
        }
    } invalidated(points);
    const PathBezierCurve<Group> empty(std::vector<Group>{}, 5, false, 0.0);
    // Failed construction can retain segments; neither invalid form is
    // queryable.
    for (const PathBase<Group>* invalid :
         {static_cast<const PathBase<Group>*>(&invalidated),
          static_cast<const PathBase<Group>*>(&empty)}) {
        if (invalid->GetPathSegmentAtS(0.5) ||
            invalid->GetPathSegmentByIndex(0))
            return false;
        for (int kind = 1; kind < 5; ++kind) {
            try {
                query(*invalid, kind, 0.5);
                return false;
            } catch (const std::logic_error&) {
            }
        }
    }
    return true;
}

bool CustomSegmentQueriesAreRespected() {
    using Group = Rn<double, 2>;
    struct CustomSegment : PathSegBezierCurve5th<Group> {
        static std::array<Group, 3> Points() {
            std::array<Group, 3> points;
            points[0].Coeffs() << 0.0, 0.0;
            points[1].Coeffs() << 2.0, 0.0;
            points[2].Coeffs() << 5.0, 0.0;
            return points;
        }
        CustomSegment() : PathSegBezierCurve5th(Points(), 0.0) {}
        Group GetConfig(double s) const override {
            auto result = PathSegBezierCurve5th::GetConfig(s);
            result.Coeffs()[1] += 7.0;
            return result;
        }
        Group::Tangent GetTangent(double s) const override {
            return 2.0 * PathSegBezierCurve5th::GetTangent(s);
        }
        Group::Tangent GetCurvature(double) const override {
            Group::Tangent result;
            result.Coeffs() << 3.0, 4.0;
            return result;
        }
        Group::Tangent GetTorsion(double) const override {
            Group::Tangent result;
            result.Coeffs() << 5.0, 6.0;
            return result;
        }
    };
    struct CustomPath : PathBase<Group> {
        CustomPath() {
            path_segments_.push_back(std::make_shared<CustomSegment>());
            length_ = path_segments_.front()->GetLength();
            valid_ = true;
        }
    };
    struct QueryProbe : TrajectoryBase<Group> {
        QueryProbe() {
            path_ = std::make_shared<CustomPath>();
            trajectory_pspline_ = std::make_shared<PSpline>();
            trajectory_pspline_->PushBack(
                std::make_shared<Polynomial>(Eigen::Vector4d(0.0, 1.0, 0.1, 0.01)),
                1.0);
            valid_ = true;
            SetMinimumDuration(1.7);
        }
    } trajectory;
    for (double time : {-1.0, 0.0, 0.1, 0.5, 1.0, 1.7, 2.0}) {
        const auto state = trajectory.GetState(time);
        if ((state.position - trajectory.GetPosition(time)).Coeffs().norm() > 1e-12 ||
            (state.velocity - trajectory.GetVelocity(time)).Coeffs().norm() > 1e-12 ||
            (state.acceleration - trajectory.GetAcceleration(time)).Coeffs().norm() >
                1e-12 ||
            (state.jerk - trajectory.GetJerk(time)).Coeffs().norm() > 1e-12)
            return false;
    }
    return true;
}

int main() {
    if (!CustomSegmentQueriesAreRespected())
        return 13;
    if (!PathQueriesRespectValidation()) return 12;
    if (!LowDimensionalCartesianBlendIsValid<1>() ||
        !LowDimensionalCartesianBlendIsValid<2>() ||
        !LowDimensionalCartesianBlendIsValid<3>())
        return 11;
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
    Eigen::Matrix4d middle_pose = Eigen::Matrix4d::Identity();
    middle_pose(0, 3) = 0.1;
    const std::array<SE3d, 3> cartesian_controls{
            poses.front(), SE3d(middle_pose), poses.back()};
    const PathSegLinear<SE3d> cartesian_line(
            {poses.front(), poses.back()}, 0.0, true);
    const PathSegBezierCurve2nd<SE3d> cartesian_quadratic(
            cartesian_controls, 0.0, true);
    const PathSegBezierCurve5th<SE3d> cartesian_quintic(
            cartesian_controls, 0.0, 1.0, 0.0, 1.0, 0.0, true);
    // Exercise each segment's SE3 tangent metric under Eigen assertions.
    for (const double length : {cartesian_line.GetLength(),
                                cartesian_quadratic.GetLength(),
                                cartesian_quintic.GetLength()}) {
        if (!std::isfinite(length) || std::abs(length - 0.2) > 1e-12)
            return 10;
    }
    const std::array<SE3d, 3> rotation_controls{
            SE3d(), SE3d(Eigen::Vector3d::Zero(), SO3d(0.0, 0.0, 0.1)),
            SE3d(Eigen::Vector3d::Zero(), SO3d(0.0, 0.0, 0.2))};
    const PathSegBezierCurve5th<SE3d> rotation_quintic(
            rotation_controls, 0.0, 1.0, 0.0, 1.0, 0.0, true);
    if (!rotation_quintic.IsValid() ||
        std::abs(rotation_quintic.GetLength() - 0.2) > 1e-12)
        return 12;
    for (int i = 0; i <= 10; ++i) {
        const double s = rotation_quintic.GetLength() * i / 10.0;
        const SE3d expected(Eigen::Vector3d::Zero(), SO3d(0.0, 0.0, s));
        if (!rotation_quintic.GetConfig(s).IsApprox(expected, 1e-10))
            return 12;
    }
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
