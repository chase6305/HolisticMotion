#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "holistic_motion/robot/Robot.h"
#include "holistic_motion/kinematics/OPWKinematics.h"
#include "holistic_motion/kinematics/URKinematics.h"
#include "holistic_motion/kinematics/srs/SRSKinematics.h"
#include "holistic_motion/kinematics/fep/FEPKinematics.h"
#include "holistic_motion/planning/NullSpacePlanner.h"
#include "holistic_motion/trajectory/PathBezierCurve.h"
#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "holistic_motion/trajectory/TrajectoryTrapezoidal.h"
#include "Bindings.h"

namespace py = pybind11;
using namespace holistic_motion::robotics;

template <int N>
class JointTrajectory {
public:
    using Group = Rn<double, N>;

    JointTrajectory(const Eigen::MatrixXd& waypoints,
                    const Eigen::VectorXd& max_velocity,
                    const Eigen::VectorXd& max_acceleration,
                    const Eigen::VectorXd& max_jerk,
                    double blend_tolerance,
                    double minimum_duration,
                    const std::string& profile) {
        if (waypoints.rows() < 2 || waypoints.cols() != N) {
            throw std::invalid_argument("waypoints must have shape (M, DOF), M >= 2");
        }
        if (!waypoints.allFinite()) {
            throw std::invalid_argument("waypoints must be finite");
        }
        bool has_motion = false;
        for (Eigen::Index row = 1; row < waypoints.rows(); ++row) {
            if ((waypoints.row(row) - waypoints.row(row - 1)).norm() >
                Epsilon) {
                has_motion = true;
                break;
            }
        }
        if (!has_motion) {
            throw std::invalid_argument(
                    "waypoints must contain at least two distinct positions");
        }
        if (max_velocity.size() != N || max_acceleration.size() != N ||
            max_jerk.size() != N) {
            throw std::invalid_argument("trajectory limits must match DOF");
        }
        if (!std::isfinite(blend_tolerance) || blend_tolerance < 0.0) {
            throw std::invalid_argument(
                    "blend_tolerance must be finite and non-negative");
        }
        if (!std::isfinite(minimum_duration) || minimum_duration < 0.0) {
            throw std::invalid_argument(
                    "minimum_duration must be finite and non-negative");
        }
        std::vector<Group> points(static_cast<std::size_t>(waypoints.rows()));
        for (Eigen::Index i = 0; i < waypoints.rows(); ++i) {
            points[static_cast<std::size_t>(i)].Coeffs() = waypoints.row(i);
        }
        path_ = std::make_shared<PathBezierCurve<Group>>(
                points, 5, false, blend_tolerance);
        auto constraints = std::make_shared<TrajectoryConstraints>(
                max_velocity, max_acceleration, max_jerk);
        if (!constraints->IsValid()) {
            throw std::invalid_argument(
                    "trajectory limits must be finite and positive");
        }
        if (profile == "double_s") {
            profile_ = profile;
            trajectory_ = std::make_shared<TrajectoryDoubleS<Group>>(
                    path_, constraints);
        } else if (profile == "trapezoidal" || profile == "trapezium") {
            profile_ = "trapezoidal";
            trajectory_ = std::make_shared<TrajectoryTrapezoidal<Group>>(
                    path_, constraints);
        } else {
            throw std::invalid_argument(
                    "profile must be 'double_s' or 'trapezoidal'");
        }
        if (!trajectory_->IsValid()) {
            throw std::runtime_error("failed to construct joint trajectory");
        }
        if (!trajectory_->SetMinimumDuration(minimum_duration)) {
            throw std::invalid_argument(
                    "minimum_duration cannot be represented by this trajectory");
        }
    }

    double Duration() const { return trajectory_->GetDuration(); }
    constexpr int Dof() const { return N; }
    double TimeScale() const { return trajectory_->GetTimeScale(); }
    double BlendTolerance() const {
        return path_->GetBlendTolerance();
    }
    const std::string& Profile() const { return profile_; }
    Eigen::VectorXd MaxVelocity() const {
        return trajectory_->GetMaxVelocity();
    }
    Eigen::VectorXd MaxAcceleration() const {
        return trajectory_->GetMaxAcceleration();
    }
    Eigen::VectorXd MaxJerk() const { return trajectory_->GetMaxJerk(); }
    double PathLength() const { return path_->GetLength(); }
    Eigen::MatrixXd Waypoints() const {
        const auto& waypoints = path_->GetWaypoints();
        Eigen::MatrixXd result(static_cast<Eigen::Index>(waypoints.size()), N);
        for (Eigen::Index index = 0; index < result.rows(); ++index) {
            result.row(index) =
                    waypoints[static_cast<std::size_t>(index)].Coeffs().transpose();
        }
        return result;
    }
    Eigen::VectorXd Breakpoints() const {
        const auto breakpoints = trajectory_->GetBreakpoints();
        Eigen::VectorXd result(static_cast<Eigen::Index>(breakpoints.size()));
        for (Eigen::Index index = 0; index < result.size(); ++index) {
            result[index] = breakpoints[static_cast<std::size_t>(index)];
        }
        return result;
    }
    Eigen::Matrix<double, N, 1> Position(double time) const {
        ValidateTime(time);
        return trajectory_->GetPosition(time).Coeffs();
    }
    Eigen::Matrix<double, N, 1> Velocity(double time) const {
        ValidateTime(time);
        return trajectory_->GetVelocity(time).Coeffs();
    }
    Eigen::Matrix<double, N, 1> Acceleration(double time) const {
        ValidateTime(time);
        return trajectory_->GetAcceleration(time).Coeffs();
    }
    Eigen::Matrix<double, N, 1> Jerk(double time) const {
        ValidateTime(time);
        return trajectory_->GetJerk(time).Coeffs();
    }
    void SetMinimumDuration(double duration) {
        if (!trajectory_->SetMinimumDuration(duration)) {
            throw std::invalid_argument(
                    "minimum_duration must be finite and non-negative");
        }
    }
    py::tuple Sample(const Eigen::VectorXd& times) const {
        if (!times.allFinite()) {
            throw std::invalid_argument("trajectory times must be finite");
        }
        Eigen::MatrixXd positions(times.size(), N);
        Eigen::MatrixXd velocities(times.size(), N);
        Eigen::MatrixXd accelerations(times.size(), N);
        Eigen::MatrixXd jerks(times.size(), N);
        {
            py::gil_scoped_release release;
            for (Eigen::Index index = 0; index < times.size(); ++index) {
                const auto state = trajectory_->GetState(times[index]);
                positions.row(index) = state.position.Coeffs().transpose();
                velocities.row(index) = state.velocity.Coeffs().transpose();
                accelerations.row(index) =
                        state.acceleration.Coeffs().transpose();
                jerks.row(index) = state.jerk.Coeffs().transpose();
            }
        }
        return py::make_tuple(positions, velocities, accelerations, jerks);
    }
    py::tuple SampleUniform(std::size_t samples) const {
        if (samples < 2) {
            throw std::invalid_argument(
                    "uniform trajectory sampling requires at least 2 samples");
        }
        if (samples > static_cast<std::size_t>(
                              (std::numeric_limits<Eigen::Index>::max)())) {
            throw std::invalid_argument("sample count is too large");
        }
        const Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(
                static_cast<Eigen::Index>(samples), 0.0, Duration());
        const py::tuple states = Sample(times);
        return py::make_tuple(
                times, states[0], states[1], states[2], states[3]);
    }
    py::tuple State(double time) const {
        ValidateTime(time);
        const auto state = trajectory_->GetState(time);
        return py::make_tuple(
                state.position.Coeffs(), state.velocity.Coeffs(),
                state.acceleration.Coeffs(), state.jerk.Coeffs());
    }
    py::dict ConstraintReport(std::size_t samples) const {
        const auto report = trajectory_->GetConstraintReport(samples);
        py::dict result;
        result["peak_velocity"] = report.peak_velocity;
        result["peak_acceleration"] = report.peak_acceleration;
        result["peak_jerk"] = report.peak_jerk;
        result["velocity_utilization"] = report.velocity_utilization;
        result["acceleration_utilization"] = report.acceleration_utilization;
        result["jerk_utilization"] = report.jerk_utilization;
        result["maximum_velocity_jump"] = report.maximum_velocity_jump;
        result["maximum_acceleration_jump"] =
                report.maximum_acceleration_jump;
        result["maximum_utilization"] = report.maximum_utilization;
        result["within_limits"] = report.within_limits;
        result["velocity_continuous"] = report.velocity_continuous;
        result["acceleration_continuous"] = report.acceleration_continuous;
        return result;
    }

private:
    static void ValidateTime(double time) {
        if (!std::isfinite(time)) {
            throw std::invalid_argument("trajectory time must be finite");
        }
    }

    std::shared_ptr<PathBezierCurve<Group>> path_;
    std::shared_ptr<TrajectoryBase<Group>> trajectory_;
    std::string profile_;
};

template <int N>
void BindJointTrajectory(py::module_& module, const char* name) {
    py::class_<JointTrajectory<N>>(
            module, name,
            "Fifth-order Bezier joint path with selectable Double-S or trapezoidal timing.")
            .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
                          const Eigen::VectorXd&, const Eigen::VectorXd&,
                          double, double, const std::string&>(),
                 py::arg("waypoints"), py::arg("max_velocity"),
                 py::arg("max_acceleration"), py::arg("max_jerk"),
                 py::arg("blend_tolerance") = 0.0,
                 py::arg("minimum_duration") = 0.0,
                 py::arg("profile") = "double_s")
            .def_property_readonly("duration", &JointTrajectory<N>::Duration)
            .def_property_readonly("dof", &JointTrajectory<N>::Dof)
            .def_property_readonly("time_scale", &JointTrajectory<N>::TimeScale)
            .def_property_readonly("blend_tolerance",
                                   &JointTrajectory<N>::BlendTolerance)
            .def_property_readonly("profile", &JointTrajectory<N>::Profile)
            .def_property_readonly("breakpoints", &JointTrajectory<N>::Breakpoints)
            .def_property_readonly("max_velocity",
                                   &JointTrajectory<N>::MaxVelocity)
            .def_property_readonly("max_acceleration",
                                   &JointTrajectory<N>::MaxAcceleration)
            .def_property_readonly("max_jerk", &JointTrajectory<N>::MaxJerk)
            .def_property_readonly("path_length", &JointTrajectory<N>::PathLength)
            .def_property_readonly("waypoints", &JointTrajectory<N>::Waypoints)
            .def("position", &JointTrajectory<N>::Position, py::arg("time"),
                 "Evaluate joint position; finite times are clamped to the trajectory.")
            .def("velocity", &JointTrajectory<N>::Velocity, py::arg("time"),
                 "Evaluate joint velocity at a finite time.")
            .def("acceleration", &JointTrajectory<N>::Acceleration,
                 py::arg("time"),
                 "Evaluate joint acceleration at a finite time.")
            .def("jerk", &JointTrajectory<N>::Jerk, py::arg("time"),
                 "Evaluate joint jerk at a finite time.")
            .def("set_minimum_duration",
                 &JointTrajectory<N>::SetMinimumDuration,
                 py::arg("duration"),
                 "Slow the trajectory to at least duration seconds; never speeds it up.")
            .def("sample", &JointTrajectory<N>::Sample, py::arg("times"),
                 "Return position, velocity, acceleration, and jerk matrices for 1-D times.")
            .def("sample_uniform", &JointTrajectory<N>::SampleUniform,
                 py::arg("samples") = 1001,
                 "Uniformly sample the inclusive [0, duration] interval and "
                 "return time plus four state matrices.")
            .def("state", &JointTrajectory<N>::State, py::arg("time"),
                 "Return position, velocity, acceleration, and jerk at one time.")
            .def("constraint_report", &JointTrajectory<N>::ConstraintReport,
                 py::arg("samples") = 2001,
                 "Return sampled per-joint peaks, limit utilization, and "
                 "breakpoint continuity diagnostics.");
}

template <int N>
py::object MakeJointTrajectory(
        const Eigen::MatrixXd& waypoints,
        const Eigen::VectorXd& max_velocity,
        const Eigen::VectorXd& max_acceleration,
        const Eigen::VectorXd& max_jerk,
        double blend_tolerance,
        double minimum_duration,
        const std::string& profile) {
    return py::cast(std::make_unique<JointTrajectory<N>>(
            waypoints, max_velocity, max_acceleration, max_jerk,
            blend_tolerance, minimum_duration, profile));
}

py::object MakeDynamicJointTrajectory(
        const Eigen::MatrixXd& waypoints,
        const Eigen::VectorXd& max_velocity,
        const Eigen::VectorXd& max_acceleration,
        const Eigen::VectorXd& max_jerk,
        double blend_tolerance,
        double minimum_duration,
        const std::string& profile) {
    switch (waypoints.cols()) {
        case 1: return MakeJointTrajectory<1>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 2: return MakeJointTrajectory<2>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 3: return MakeJointTrajectory<3>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 4: return MakeJointTrajectory<4>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 5: return MakeJointTrajectory<5>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 6: return MakeJointTrajectory<6>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 7: return MakeJointTrajectory<7>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 8: return MakeJointTrajectory<8>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 9: return MakeJointTrajectory<9>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 10: return MakeJointTrajectory<10>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 11: return MakeJointTrajectory<11>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 12: return MakeJointTrajectory<12>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 13: return MakeJointTrajectory<13>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 14: return MakeJointTrajectory<14>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 15: return MakeJointTrajectory<15>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 16: return MakeJointTrajectory<16>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 17: return MakeJointTrajectory<17>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 18: return MakeJointTrajectory<18>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 19: return MakeJointTrajectory<19>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 20: return MakeJointTrajectory<20>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 21: return MakeJointTrajectory<21>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 22: return MakeJointTrajectory<22>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 23: return MakeJointTrajectory<23>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 24: return MakeJointTrajectory<24>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 25: return MakeJointTrajectory<25>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 26: return MakeJointTrajectory<26>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 27: return MakeJointTrajectory<27>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 28: return MakeJointTrajectory<28>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 29: return MakeJointTrajectory<29>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 30: return MakeJointTrajectory<30>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 31: return MakeJointTrajectory<31>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        case 32: return MakeJointTrajectory<32>(waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance, minimum_duration, profile);
        default:
            throw std::invalid_argument(
                    "joint trajectory supports waypoint DOF from 1 through 32");
    }
}

class CartesianLineTrajectory {
public:
    CartesianLineTrajectory(
            const Eigen::Matrix4d& start,
            const Eigen::Matrix4d& end,
            const Eigen::VectorXd& max_velocity,
            const Eigen::VectorXd& max_acceleration,
            const Eigen::VectorXd& max_jerk,
            double minimum_duration,
            const std::string& profile) {
        ValidateTransform(start, "start");
        ValidateTransform(end, "end");
        if (max_velocity.size() != 6 || max_acceleration.size() != 6 ||
            max_jerk.size() != 6) {
            throw std::invalid_argument(
                    "Cartesian limits must contain 6 values");
        }
        const std::vector<SE3d> points{SE3d(start), SE3d(end)};
        path_ = std::make_shared<PathBezierCurve<SE3d>>(
                points, 5, true, 0.0);
        auto constraints = std::make_shared<TrajectoryConstraints>(
                max_velocity, max_acceleration, max_jerk);
        if (!constraints->IsValid()) {
            throw std::invalid_argument(
                    "Cartesian limits must be finite and positive");
        }
        if (profile == "double_s") {
            profile_ = profile;
            trajectory_ = std::make_shared<TrajectoryDoubleS<SE3d>>(
                    path_, constraints);
        } else if (profile == "trapezoidal" || profile == "trapezium") {
            profile_ = "trapezoidal";
            trajectory_ = std::make_shared<TrajectoryTrapezoidal<SE3d>>(
                    path_, constraints);
        } else {
            throw std::invalid_argument(
                    "profile must be 'double_s' or 'trapezoidal'");
        }
        if (!trajectory_->IsValid()) {
            throw std::runtime_error(
                    "failed to construct Cartesian line trajectory");
        }
        if (!trajectory_->SetMinimumDuration(minimum_duration)) {
            throw std::invalid_argument(
                    "minimum_duration must be finite and non-negative");
        }
    }

    double Duration() const { return trajectory_->GetDuration(); }
    const std::string& Profile() const { return profile_; }
    Eigen::Matrix4d Position(double time) const {
        ValidateTime(time);
        return trajectory_->GetPosition(time).GetTransform();
    }
    py::tuple State(double time) const {
        ValidateTime(time);
        const auto state = trajectory_->GetState(time);
        return py::make_tuple(
                state.position.GetTransform(), state.velocity.Coeffs(),
                state.acceleration.Coeffs(), state.jerk.Coeffs());
    }
    py::tuple SampleUniform(std::size_t samples) const {
        if (samples < 2) {
            throw std::invalid_argument(
                    "uniform trajectory sampling requires at least 2 samples");
        }
        const Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(
                static_cast<Eigen::Index>(samples), 0.0, Duration());
        py::array_t<double> poses(py::array::ShapeContainer{
                static_cast<py::ssize_t>(samples),
                static_cast<py::ssize_t>(4),
                static_cast<py::ssize_t>(4)});
        Eigen::MatrixXd velocity(samples, 6), acceleration(samples, 6),
                jerk(samples, 6);
        auto pose_view = poses.mutable_unchecked<3>();
        for (std::size_t index = 0; index < samples; ++index) {
            const auto state = trajectory_->GetState(times[index]);
            const Eigen::Matrix4d pose = state.position.GetTransform();
            for (int row = 0; row < 4; ++row)
                for (int column = 0; column < 4; ++column)
                    pose_view(index, row, column) = pose(row, column);
            velocity.row(index) = state.velocity.Coeffs().transpose();
            acceleration.row(index) =
                    state.acceleration.Coeffs().transpose();
            jerk.row(index) = state.jerk.Coeffs().transpose();
        }
        return py::make_tuple(times, poses, velocity, acceleration, jerk);
    }
    py::dict ConstraintReport(std::size_t samples) const {
        const auto report = trajectory_->GetConstraintReport(samples);
        py::dict result;
        result["peak_velocity"] = report.peak_velocity;
        result["peak_acceleration"] = report.peak_acceleration;
        result["peak_jerk"] = report.peak_jerk;
        result["maximum_utilization"] = report.maximum_utilization;
        result["within_limits"] = report.within_limits;
        result["velocity_continuous"] = report.velocity_continuous;
        result["acceleration_continuous"] = report.acceleration_continuous;
        return result;
    }

private:
    static void ValidateTime(double time) {
        if (!std::isfinite(time))
            throw std::invalid_argument("trajectory time must be finite");
    }
    static void ValidateTransform(
            const Eigen::Matrix4d& transform, const char* name) {
        const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
        if (!transform.allFinite() ||
            !transform.row(3).isApprox(
                    Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9) ||
            !(rotation.transpose() * rotation).isApprox(
                    Eigen::Matrix3d::Identity(), 1e-7) ||
            std::abs(rotation.determinant() - 1.0) > 1e-7) {
            throw std::invalid_argument(
                    std::string(name) + " must be a finite rigid transform");
        }
    }

    std::shared_ptr<PathBezierCurve<SE3d>> path_;
    std::shared_ptr<TrajectoryBase<SE3d>> trajectory_;
    std::string profile_;
};

PYBIND11_MODULE(_holistic_motion, module) {
    module.doc() = "Robot models, kinematics, manifolds, and trajectories";

    holistic_motion::python::BindCollision(module);
    holistic_motion::python::BindSamplingPlanning(module);

    py::class_<SE3d>(module, "SE3")
            .def(py::init<>())
            .def(py::init<const Eigen::Matrix4d&>())
            .def("matrix", &SE3d::GetTransform)
            .def("inverse", &SE3d::Inverse);

    py::enum_<JointType>(module, "JointType")
            .value("UNKNOWN", JointType::UNKNOWN)
            .value("REVOLUTE", JointType::REVOLUTE)
            .value("PRISMATIC", JointType::PRISMATIC)
            .value("CONTINUOUS", JointType::CONTINUOUS)
            .value("FLOATING", JointType::FLOATING)
            .value("PLANAR", JointType::PLANAR)
            .value("FIXED", JointType::FIXED);

    py::class_<JointLimit, std::shared_ptr<JointLimit>>(module, "JointLimit")
            .def_readonly("lower", &JointLimit::lower_limit)
            .def_readonly("upper", &JointLimit::upper_limit)
            .def_readonly("max_velocity", &JointLimit::max_velocity)
            .def_readonly("max_acceleration", &JointLimit::max_acceleration)
            .def_readonly("max_jerk", &JointLimit::max_jerk)
            .def_readonly("max_effort", &JointLimit::max_effort);

    py::class_<Joint, std::shared_ptr<Joint>>(module, "Joint")
            .def_readonly("name", &Joint::name)
            .def_readonly("parent_link", &Joint::parent_link)
            .def_readonly("child_link", &Joint::child_link)
            .def_readonly("joint_type", &Joint::joint_type)
            .def_readonly("axis", &Joint::axis)
            .def_readonly("limit", &Joint::limit)
            .def_readonly("mimic_joint", &Joint::mimic_joint)
            .def_readonly("mimic_multiplier", &Joint::mimic_multiplier)
            .def_readonly("mimic_offset", &Joint::mimic_offset)
            .def_property_readonly("origin", [](const Joint& self) {
                return self.origin_pose.GetTransform();
            });

    py::class_<Inertial>(module, "Inertial")
            .def_readonly("present", &Inertial::present)
            .def_readonly("mass", &Inertial::mass)
            .def_readonly("inertia", &Inertial::inertia)
            .def_property_readonly("origin", [](const Inertial& self) {
                return self.origin.GetTransform();
            });

    py::enum_<GeometryType>(module, "GeometryType")
            .value("MESH", GeometryType::MESH)
            .value("BOX", GeometryType::BOX)
            .value("CYLINDER", GeometryType::CYLINDER)
            .value("SPHERE", GeometryType::SPHERE);

    py::class_<VisualGeometry>(module, "VisualGeometry")
            .def_readonly("name", &VisualGeometry::name)
            .def_readonly("type", &VisualGeometry::type)
            .def_readonly("mesh_path", &VisualGeometry::mesh_path)
            .def_readonly("material_name", &VisualGeometry::material_name)
            .def_readonly("texture_path", &VisualGeometry::texture_path)
            .def_readonly("color", &VisualGeometry::color)
            .def_readonly("has_color", &VisualGeometry::has_color)
            .def_readonly("scale", &VisualGeometry::scale)
            .def_readonly("size", &VisualGeometry::size)
            .def_readonly("radius", &VisualGeometry::radius)
            .def_readonly("length", &VisualGeometry::length)
            .def_property_readonly("origin", [](const VisualGeometry& self) {
                return self.origin.GetTransform();
            });

    py::class_<Link, std::shared_ptr<Link>>(module, "Link")
            .def_readonly("name", &Link::name)
            .def_readonly("inertial", &Link::inertial)
            .def_readonly("visuals", &Link::visuals)
            .def_readonly("parent_joint", &Link::parent_joint)
            .def_readonly("child_joints", &Link::child_joints);

    py::class_<KinematicsBase, std::shared_ptr<KinematicsBase>>(
            module, "AnalyticKinematics")
            .def("dof", &KinematicsBase::GetDOF)
            .def("forward", [](const KinematicsBase& self,
                                const Eigen::VectorXd& joints) {
                SE3d pose;
                if (!self.GetFK(joints, pose)) {
                    throw py::value_error("invalid analytic FK joint vector");
                }
                return pose.GetTransform();
            }, py::arg("joints"))
            .def("solve_all", [](const KinematicsBase& self,
                                  const Eigen::Matrix4d& target,
                                  Eigen::VectorXd seed) {
                IkRtn solutions;
                std::vector<double> distances;
                if (!self.GetIK(SE3d(target), solutions, seed, distances)) {
                    return std::vector<Eigen::VectorXd>{};
                }
                return solutions.ik_joints;
            }, py::arg("target"), py::arg("seed"))
            .def("inverse", [](const KinematicsBase& self,
                                const Eigen::Matrix4d& target,
                                Eigen::VectorXd seed) {
                IkRtn solutions;
                double distance = 0.0;
                if (!self.GetNearestIK(SE3d(target), solutions, seed, distance) ||
                    solutions.ik_joints.empty()) {
                    throw py::value_error("analytic inverse kinematics failed");
                }
                return solutions.ik_joints.front();
            }, py::arg("target"), py::arg("seed"));

    py::class_<OPWParameters>(module, "OPWParameters")
            .def(py::init<>())
            .def_readwrite("a1", &OPWParameters::a1)
            .def_readwrite("a2", &OPWParameters::a2)
            .def_readwrite("b", &OPWParameters::b)
            .def_readwrite("c1", &OPWParameters::c1)
            .def_readwrite("c2", &OPWParameters::c2)
            .def_readwrite("c3", &OPWParameters::c3)
            .def_readwrite("c4", &OPWParameters::c4)
            .def_readwrite("offsets", &OPWParameters::offsets)
            .def_readwrite("rotation_directions",
                           &OPWParameters::rotation_directions)
            .def_property("base_transform",
                    [](const OPWParameters& self) {
                        return self.T_b_ob.GetTransform();
                    }, [](OPWParameters& self, const Eigen::Matrix4d& value) {
                        self.T_b_ob = SE3d(value);
                    })
            .def_property("tool_transform",
                    [](const OPWParameters& self) {
                        return self.T_oe_e.GetTransform();
                    }, [](OPWParameters& self, const Eigen::Matrix4d& value) {
                        self.T_oe_e = SE3d(value);
                    });

    py::class_<URParameters>(module, "URParameters")
            .def(py::init<>())
            .def_readwrite("d1", &URParameters::d1)
            .def_readwrite("a2", &URParameters::a2)
            .def_readwrite("a3", &URParameters::a3)
            .def_readwrite("d4", &URParameters::d4)
            .def_readwrite("d5", &URParameters::d5)
            .def_readwrite("d6", &URParameters::d6)
            .def_readwrite("offsets", &URParameters::offsets)
            .def_readwrite("rotation_directions",
                           &URParameters::rotation_directions)
            .def_property("base_transform",
                    [](const URParameters& self) {
                        return self.T_b_ob.GetTransform();
                    }, [](URParameters& self, const Eigen::Matrix4d& value) {
                        self.T_b_ob = SE3d(value);
                    })
            .def_property("tool_transform",
                    [](const URParameters& self) {
                        return self.T_oe_e.GetTransform();
                    }, [](URParameters& self, const Eigen::Matrix4d& value) {
                        self.T_oe_e = SE3d(value);
                    });

    auto valid_directions = [](const auto& params) {
        return std::all_of(
                params.rotation_directions.begin(),
                params.rotation_directions.end(),
                [](signed char direction) {
                    return direction == 1 || direction == -1;
                });
    };
    auto valid_offsets = [](const auto& params) {
        return std::all_of(params.offsets.begin(), params.offsets.end(),
                           [](double value) { return std::isfinite(value); });
    };
    auto make_joint_nodes = [](const Eigen::VectorXd& lower,
                               const Eigen::VectorXd& upper) {
        if (lower.size() != 6 || upper.size() != 6 || !lower.allFinite() ||
            !upper.allFinite() || (lower.array() > upper.array()).any()) {
            throw py::value_error("joint limits must be finite length-6 arrays");
        }
        std::vector<JointNode> nodes;
        for (int index = 0; index < 6; ++index) {
            nodes.emplace_back(SE3d(), Eigen::Vector3d::UnitZ(),
                               JointType::REVOLUTE, lower[index], upper[index]);
        }
        return nodes;
    };
    py::class_<OPWKinematics, KinematicsBase,
               std::shared_ptr<OPWKinematics>>(module, "OPWKinematics")
            .def(py::init([make_joint_nodes, valid_directions, valid_offsets](
                                             const OPWParameters& params,
                                             const Eigen::VectorXd& lower,
                                             const Eigen::VectorXd& upper) {
                const bool geometry_valid =
                        std::isfinite(params.a1) && std::isfinite(params.a2) &&
                        std::isfinite(params.b) && std::isfinite(params.c1) &&
                        std::isfinite(params.c2) && std::isfinite(params.c3) &&
                        std::isfinite(params.c4) &&
                        std::abs(params.c2) > 1e-12 &&
                        std::hypot(params.a2, params.c3) > 1e-12;
                if (!geometry_valid || !valid_directions(params) ||
                    !valid_offsets(params)) {
                    throw py::value_error(
                            "invalid OPW geometry, offsets, or axis directions");
                }
                auto solver = std::make_shared<OPWKinematics>(
                        params, make_joint_nodes(lower, upper));
                solver->DisableRobotConfig();
                return solver;
            }), py::arg("parameters"), py::arg("lower_limits"),
                py::arg("upper_limits"));
    py::class_<URKinematics, KinematicsBase,
               std::shared_ptr<URKinematics>>(module, "URKinematics")
            .def(py::init([make_joint_nodes, valid_directions, valid_offsets](
                                             const URParameters& params,
                                             const Eigen::VectorXd& lower,
                                             const Eigen::VectorXd& upper) {
                const bool geometry_valid =
                        std::isfinite(params.d1) && std::isfinite(params.a2) &&
                        std::isfinite(params.a3) && std::isfinite(params.d4) &&
                        std::isfinite(params.d5) && std::isfinite(params.d6) &&
                        std::abs(params.a2) > 1e-12 &&
                        std::abs(params.a3) > 1e-12;
                if (!geometry_valid || !valid_directions(params) ||
                    !valid_offsets(params)) {
                    throw py::value_error(
                            "invalid UR geometry, offsets, or axis directions");
                }
                auto solver = std::make_shared<URKinematics>(
                        params, make_joint_nodes(lower, upper));
                solver->DisableRobotConfig();
                return solver;
            }), py::arg("parameters"), py::arg("lower_limits"),
                py::arg("upper_limits"));

    py::class_<NumericalKinematics, KinematicsBase,
               std::shared_ptr<NumericalKinematics>>(
            module, "NumericalKinematics")
            .def_property(
                    "home_joints", &NumericalKinematics::GetHomeJoints,
                    [](NumericalKinematics& self,
                       const Eigen::VectorXd& joints) {
                        if (!self.SetHomeJoints(joints)) {
                            throw py::value_error(
                                    "home_joints must match DOF and be finite");
                        }
                    })
            .def_property_readonly("joint_limits",
                    [](const NumericalKinematics& self) {
                        Eigen::VectorXd upper;
                        Eigen::VectorXd lower;
                        self.GetJointLimits(upper, lower);
                        return py::make_tuple(lower, upper);
                    })
            .def("is_reachable", [](const NumericalKinematics& self,
                                     const Eigen::Matrix4d& target) {
                if (!target.allFinite()) return false;
                return self.IsReachable(SE3d(target));
            }, py::arg("target"))
            .def("forward", [](const NumericalKinematics& self,
                               const Eigen::VectorXd& joints) {
                SE3d pose;
                if (!self.GetFK(joints, pose)) {
                    throw py::value_error("joint vector does not match robot DOF");
                }
                return pose.GetTransform();
            })
            .def("forward_all", [](const NumericalKinematics& self,
                                   const Eigen::VectorXd& joints) {
                std::vector<SE3d> poses;
                if (!self.GetAllFK(joints, poses)) {
                    throw py::value_error("joint vector does not match robot DOF");
                }
                std::vector<Eigen::Matrix4d> matrices;
                matrices.reserve(poses.size());
                for (const auto& pose : poses) {
                    matrices.push_back(pose.GetTransform());
                }
                return matrices;
            }, py::arg("joints"))
            .def("inverse", [](const NumericalKinematics& self,
                               const Eigen::Matrix4d& target,
                               Eigen::VectorXd seed) {
                IkRtn solutions;
                double distance = 0.0;
                if (!self.GetNearestIK(SE3d(target), solutions, seed, distance) ||
                    solutions.ik_joints.empty()) {
                    throw py::value_error("inverse kinematics did not converge");
                }
                return solutions.ik_joints.front();
            }, py::arg("target"), py::arg("seed"))
            .def("jacobian", [](const NumericalKinematics& self,
                                const Eigen::VectorXd& joints) {
                Eigen::MatrixXd jacobian;
                if (!self.GetJacobian(joints, jacobian)) {
                    throw py::value_error("joint vector does not match robot DOF");
                }
                return jacobian;
            }, py::arg("joints"))
            .def("set_tcp", [](NumericalKinematics& self,
                               const Eigen::Matrix4d& pose) {
                return self.SetTCP(SE3d(pose));
            }, py::arg("pose"))
            .def("set_user_frame", [](NumericalKinematics& self,
                                      const Eigen::Matrix4d& pose) {
                return self.SetUserFrame(SE3d(pose));
            }, py::arg("pose"));

    py::enum_<SRSSolveMethod>(module, "SRSSolveMethod")
            .value("SEEDED_NUMERICAL", SRSSolveMethod::SEEDED_NUMERICAL)
            .value("CONFIGURATION", SRSSolveMethod::CONFIGURATION)
            .value("ALL_CONFIGURATIONS", SRSSolveMethod::ALL_CONFIGURATIONS)
            .value("NEAREST_REDUNDANCY", SRSSolveMethod::NEAREST_REDUNDANCY);

    py::class_<SRSConfiguration>(module, "SRSConfiguration")
            .def(py::init<>())
            .def_readwrite("shoulder", &SRSConfiguration::shoulder)
            .def_readwrite("elbow", &SRSConfiguration::elbow)
            .def_readwrite("wrist", &SRSConfiguration::wrist)
            .def_readwrite("redundancy", &SRSConfiguration::redundancy);

    py::class_<SRSGeometryAnalysis>(module, "SRSGeometryAnalysis")
            .def_readonly("structurally_compatible",
                          &SRSGeometryAnalysis::structurally_compatible)
            .def_readonly("closed_form_compatible",
                          &SRSGeometryAnalysis::closed_form_compatible)
            .def_readonly("shoulder_center",
                          &SRSGeometryAnalysis::shoulder_center)
            .def_readonly("wrist_center", &SRSGeometryAnalysis::wrist_center)
            .def_readonly("upper_arm_length",
                          &SRSGeometryAnalysis::upper_arm_length)
            .def_readonly("forearm_length",
                          &SRSGeometryAnalysis::forearm_length)
            .def_readonly("shoulder_axis_residual",
                          &SRSGeometryAnalysis::shoulder_axis_residual)
            .def_readonly("wrist_axis_residual",
                          &SRSGeometryAnalysis::wrist_axis_residual)
            .def_readonly("shoulder_orthogonality_residual",
                          &SRSGeometryAnalysis::shoulder_orthogonality_residual)
            .def_readonly("wrist_orthogonality_residual",
                          &SRSGeometryAnalysis::wrist_orthogonality_residual)
            .def_readonly("elbow_angle_offset",
                          &SRSGeometryAnalysis::elbow_angle_offset)
            .def_readonly("elbow_angle_direction",
                          &SRSGeometryAnalysis::elbow_angle_direction);

    py::class_<SRSKinematics, NumericalKinematics,
               std::shared_ptr<SRSKinematics>>(module, "SRSKinematics")
            .def_property_readonly("compatible", &SRSKinematics::IsCompatible)
            .def("analyze_geometry", &SRSKinematics::AnalyzeGeometry,
                 py::arg("axis_tolerance") = 1e-4)
            .def("configuration", &SRSKinematics::GetConfiguration,
                 py::arg("joints"))
            .def("solve", [](const SRSKinematics& self,
                              const Eigen::Matrix4d& target,
                              const Eigen::VectorXd& seed,
                              SRSSolveMethod method) {
                std::vector<Eigen::VectorXd> solutions;
                if (!self.Solve(SE3d(target), seed, method, solutions)) {
                    throw py::value_error("SRS inverse kinematics did not converge");
                }
                return solutions;
            }, py::arg("target"), py::arg("seed"),
               py::arg("method") = SRSSolveMethod::NEAREST_REDUNDANCY)
            .def("solve_configuration", [](const SRSKinematics& self,
                                             const Eigen::Matrix4d& target,
                                             const SRSConfiguration& config,
                                             const Eigen::VectorXd& seed) {
                Eigen::VectorXd solution;
                if (!self.SolveConfiguration(SE3d(target), config, seed,
                                             solution)) {
                    throw py::value_error("SRS configuration is infeasible");
                }
                return solution;
            }, py::arg("target"), py::arg("configuration"), py::arg("seed"))
            .def("analytic_elbow_seed", [](const SRSKinematics& self,
                                            const Eigen::Matrix4d& target,
                                            const SRSConfiguration& config,
                                            const Eigen::VectorXd& seed) {
                Eigen::VectorXd result;
                if (!self.GetAnalyticElbowSeed(SE3d(target), config, seed,
                                               result)) {
                    throw py::value_error("target is outside the SRS workspace");
                }
                return result;
            }, py::arg("target"), py::arg("configuration"), py::arg("seed"))
            .def("analytic_solution", [](const SRSKinematics& self,
                                          const Eigen::Matrix4d& target,
                                          const SRSConfiguration& config,
                                          const Eigen::VectorXd& seed) {
                Eigen::VectorXd result;
                if (!self.GetAnalyticSolution(SE3d(target), config, seed,
                                              result)) {
                    throw py::value_error("no feasible closed-form SRS solution");
                }
                return result;
            }, py::arg("target"), py::arg("configuration"), py::arg("seed"))
            .def("null_space_velocity", [](const SRSKinematics& self,
                                            const Eigen::VectorXd& joints,
                                            const Eigen::VectorXd& direction) {
                Eigen::VectorXd velocity;
                if (!self.GetNullSpaceVelocity(joints, direction, velocity)) {
                    throw py::value_error("expected a compatible 7R chain");
                }
                return velocity;
            }, py::arg("joints"), py::arg("preferred_direction"));

    py::enum_<FEPSolveMethod>(module, "FEPSolveMethod")
            .value("SEEDED_NUMERICAL", FEPSolveMethod::SEEDED_NUMERICAL)
            .value("CONFIGURATION", FEPSolveMethod::CONFIGURATION)
            .value("ALL_CONFIGURATIONS", FEPSolveMethod::ALL_CONFIGURATIONS)
            .value("NEAREST_REDUNDANCY", FEPSolveMethod::NEAREST_REDUNDANCY)
            .value("COMPATIBILITY", FEPSolveMethod::COMPATIBILITY);

    py::enum_<FEPBackend>(module, "FEPBackend")
            .value("AUTO", FEPBackend::AUTO)
            .value("CPU", FEPBackend::CPU)
            .value("CUDA", FEPBackend::CUDA);

    py::class_<FEPConfiguration>(module, "FEPConfiguration")
            .def(py::init<>())
            .def_readwrite("shoulder", &FEPConfiguration::shoulder)
            .def_readwrite("elbow", &FEPConfiguration::elbow)
            .def_readwrite("wrist", &FEPConfiguration::wrist)
            .def_readwrite("redundancy", &FEPConfiguration::redundancy);

    py::class_<FEPKinematics, NumericalKinematics,
               std::shared_ptr<FEPKinematics>>(module, "FEPKinematics")
            .def_property_readonly("compatible", &FEPKinematics::IsCompatible)
            .def_property_readonly_static(
                    "cuda_available", [](py::object) {
                        return FEPKinematics::HasCudaBackend();
                    })
            .def_property_readonly_static(
                    "cuda_compiled", [](py::object) {
                        return FEPKinematics::IsCudaCompiled();
                    })
            .def("resolve_backend", &FEPKinematics::ResolveBackend,
                 py::arg("backend"), py::arg("batch_size"))
            .def("forward_batch", [](const FEPKinematics& self,
                                      const Eigen::MatrixXd& joints,
                                      FEPBackend backend) {
                std::vector<Eigen::Matrix4d> poses;
                if (!self.ForwardBatch(joints, backend, poses)) {
                    throw py::value_error(
                            "batch FK requires finite in-limit (N, 7) joints "
                            "and an available requested backend");
                }
                py::array_t<double> result(py::array::ShapeContainer{
                        static_cast<py::ssize_t>(poses.size()),
                        static_cast<py::ssize_t>(4),
                        static_cast<py::ssize_t>(4)});
                auto values = result.mutable_unchecked<3>();
                for (py::ssize_t pose = 0;
                     pose < static_cast<py::ssize_t>(poses.size()); ++pose)
                    for (py::ssize_t row = 0; row < 4; ++row)
                        for (py::ssize_t col = 0; col < 4; ++col)
                            values(pose, row, col) =
                                    poses[static_cast<std::size_t>(pose)](row, col);
                return result;
            }, py::arg("joints"), py::arg("backend") = FEPBackend::AUTO)
            .def("configuration", &FEPKinematics::GetConfiguration,
                 py::arg("joints"))
            .def("solve", [](const FEPKinematics& self,
                              const Eigen::Matrix4d& target,
                              const Eigen::VectorXd& seed,
                              FEPSolveMethod method) {
                std::vector<Eigen::VectorXd> solutions;
                if (!self.Solve(SE3d(target), seed, method, solutions)) {
                    throw py::value_error("FEP inverse kinematics did not converge");
                }
                return solutions;
            }, py::arg("target"), py::arg("seed"),
               py::arg("method") = FEPSolveMethod::NEAREST_REDUNDANCY)
            .def("solve_configuration", [](const FEPKinematics& self,
                                             const Eigen::Matrix4d& target,
                                             const FEPConfiguration& config,
                                             const Eigen::VectorXd& seed) {
                Eigen::VectorXd solution;
                if (!self.SolveConfiguration(SE3d(target), config, seed,
                                             solution)) {
                    throw py::value_error("FEP configuration is infeasible");
                }
                return solution;
            }, py::arg("target"), py::arg("configuration"), py::arg("seed"));

    py::class_<planning::NullSpacePlanner>(module, "NullSpacePlanner")
            .def(py::init<std::shared_ptr<SRSKinematics>>(),
                 py::arg("kinematics"))
            .def("plan", [](const planning::NullSpacePlanner& self,
                            const Eigen::VectorXd& start,
                            const Eigen::VectorXd& direction,
                            int steps, double step_size) {
                std::vector<Eigen::VectorXd> path;
                if (!self.Plan(start, direction, steps, step_size, path)) {
                    throw py::value_error("null-space planning failed");
                }
                return path;
            }, py::arg("start"), py::arg("preferred_direction"),
               py::arg("steps"), py::arg("step_size") = 0.05);

    py::class_<Robot, std::shared_ptr<Robot>>(module, "Robot")
            .def(py::init<const std::string&, bool>(), py::arg("urdf_path"),
                 py::arg("load_visuals") = false)
            .def_property_readonly("name", &Robot::GetName)
            .def_property_readonly("urdf_path", &Robot::GetURDFPath)
            .def_property_readonly("root_link_name", &Robot::GetRootLinkName)
            .def_property_readonly("dof", &Robot::GetDoF)
            .def_property_readonly("model_dof", &Robot::GetModelDoF)
            .def_property_readonly("joints", &Robot::GetJoints)
            .def_property_readonly("actuated_joints", &Robot::GetActuatedJoints)
            .def_property_readonly("all_actuated_joints",
                                   &Robot::GetAllActuatedJoints)
            .def_property_readonly("links", &Robot::GetLinks)
            .def("get_link", &Robot::GetLink, py::arg("name"))
            .def("get_joint", &Robot::GetJoint, py::arg("name"))
            .def("create_kinematics", &Robot::CreateKinematics,
                 py::arg("base_link"), py::arg("tip_link"))
            .def("create_fep_kinematics", &Robot::CreateFEPKinematics,
                 py::arg("base_link"), py::arg("tip_link"))
            .def_property_readonly("kinematics", &Robot::GetKinematics)
            .def_property_readonly("has_kinematics", &Robot::HasKinematics)
            .def_property_readonly("visuals_loaded", &Robot::HasVisualsLoaded)
            .def("load_visuals", &Robot::LoadVisuals);

    py::class_<TrajectoryConstraints, std::shared_ptr<TrajectoryConstraints>>(
            module, "TrajectoryConstraints")
            .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&,
                          const Eigen::VectorXd&>(),
                 py::arg("max_velocity"), py::arg("max_acceleration"),
                 py::arg("max_jerk"))
            .def_property_readonly("valid", &TrajectoryConstraints::IsValid);

    BindJointTrajectory<1>(module, "JointTrajectory1");
    BindJointTrajectory<2>(module, "JointTrajectory2");
    BindJointTrajectory<3>(module, "JointTrajectory3");
    BindJointTrajectory<4>(module, "JointTrajectory4");
    BindJointTrajectory<5>(module, "JointTrajectory5");
    BindJointTrajectory<6>(module, "JointTrajectory6");
    BindJointTrajectory<7>(module, "JointTrajectory7");
    BindJointTrajectory<8>(module, "JointTrajectory8");
    BindJointTrajectory<9>(module, "JointTrajectory9");
    BindJointTrajectory<10>(module, "JointTrajectory10");
    BindJointTrajectory<11>(module, "JointTrajectory11");
    BindJointTrajectory<12>(module, "JointTrajectory12");
    BindJointTrajectory<13>(module, "JointTrajectory13");
    BindJointTrajectory<14>(module, "JointTrajectory14");
    BindJointTrajectory<15>(module, "JointTrajectory15");
    BindJointTrajectory<16>(module, "JointTrajectory16");
    BindJointTrajectory<17>(module, "JointTrajectory17");
    BindJointTrajectory<18>(module, "JointTrajectory18");
    BindJointTrajectory<19>(module, "JointTrajectory19");
    BindJointTrajectory<20>(module, "JointTrajectory20");
    BindJointTrajectory<21>(module, "JointTrajectory21");
    BindJointTrajectory<22>(module, "JointTrajectory22");
    BindJointTrajectory<23>(module, "JointTrajectory23");
    BindJointTrajectory<24>(module, "JointTrajectory24");
    BindJointTrajectory<25>(module, "JointTrajectory25");
    BindJointTrajectory<26>(module, "JointTrajectory26");
    BindJointTrajectory<27>(module, "JointTrajectory27");
    BindJointTrajectory<28>(module, "JointTrajectory28");
    BindJointTrajectory<29>(module, "JointTrajectory29");
    BindJointTrajectory<30>(module, "JointTrajectory30");
    BindJointTrajectory<31>(module, "JointTrajectory31");
    BindJointTrajectory<32>(module, "JointTrajectory32");
    module.def(
            "JointTrajectory", &MakeDynamicJointTrajectory,
            py::arg("waypoints"), py::arg("max_velocity"),
            py::arg("max_acceleration"), py::arg("max_jerk"),
            py::arg("blend_tolerance") = 0.0,
            py::arg("minimum_duration") = 0.0,
            py::arg("profile") = "double_s",
            "Construct a 1-32 DoF joint/configuration trajectory from its waypoint columns.");
    module.def(
            "RnTrajectory", &MakeDynamicJointTrajectory,
            py::arg("waypoints"), py::arg("max_velocity"),
            py::arg("max_acceleration"), py::arg("max_jerk"),
            py::arg("blend_tolerance") = 0.0,
            py::arg("minimum_duration") = 0.0,
            py::arg("profile") = "double_s",
            "Construct an R^n trajectory and infer n (1-32) from the waypoint columns.");
    module.def(
            "BaseTrajectory", &MakeJointTrajectory<3>,
            py::arg("waypoints"), py::arg("max_velocity"),
            py::arg("max_acceleration"), py::arg("max_jerk"),
            py::arg("blend_tolerance") = 0.0,
            py::arg("minimum_duration") = 0.0,
            py::arg("profile") = "double_s",
            "Construct an [x, y, yaw] mobile-base trajectory.");
    py::class_<CartesianLineTrajectory>(module, "CartesianLineTrajectory")
            .def(py::init<const Eigen::Matrix4d&, const Eigen::Matrix4d&,
                          const Eigen::VectorXd&, const Eigen::VectorXd&,
                          const Eigen::VectorXd&, double,
                          const std::string&>(),
                 py::arg("start"), py::arg("end"),
                 py::arg("max_velocity"), py::arg("max_acceleration"),
                 py::arg("max_jerk"), py::arg("minimum_duration") = 0.0,
                 py::arg("profile") = "double_s")
            .def_property_readonly("duration", &CartesianLineTrajectory::Duration)
            .def_property_readonly("profile", &CartesianLineTrajectory::Profile)
            .def("position", &CartesianLineTrajectory::Position, py::arg("time"))
            .def("state", &CartesianLineTrajectory::State, py::arg("time"))
            .def("sample_uniform", &CartesianLineTrajectory::SampleUniform,
                 py::arg("samples") = 1001)
            .def("constraint_report", &CartesianLineTrajectory::ConstraintReport,
                 py::arg("samples") = 2001);
}
