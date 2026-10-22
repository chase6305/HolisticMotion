#pragma once

#include "TrajectoryBindingHelpers.h"

namespace holistic_motion::python::trajectory_detail {

template <int N> class JointTrajectory {
    using SampleArray = py::array_t<double, py::array::f_style>;
    using SampleArrays = std::array<SampleArray, 4>;

  public:
    using Group = Rn<double, N>;

    JointTrajectory(const Eigen::MatrixXd &waypoints,
                    const Eigen::VectorXd &max_velocity,
                    const Eigen::VectorXd &max_acceleration,
                    const Eigen::VectorXd &max_jerk, double blend_tolerance,
                    double minimum_duration, const std::string &profile) {
        if (waypoints.rows() < 2 || waypoints.cols() != N) {
            throw std::invalid_argument(
                "waypoints must have shape (M, DOF), M >= 2");
        }
        if (!waypoints.allFinite()) {
            throw std::invalid_argument("waypoints must be finite");
        }
        bool has_motion = false;
        for (Eigen::Index row = 1; row < waypoints.rows(); ++row) {
            if ((waypoints.row(row).array() != waypoints.row(row - 1).array())
                    .any()) {
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
        path_ = std::make_shared<PathBezierCurve<Group>>(points, 5, false,
                                                         blend_tolerance);
        auto constraints = std::make_shared<TrajectoryConstraints>(
            max_velocity, max_acceleration, max_jerk);
        if (!constraints->IsValid()) {
            throw std::invalid_argument(
                "trajectory limits must be finite and positive");
        }
        if (profile == "double_s") {
            profile_ = profile;
            trajectory_ =
                std::make_shared<TrajectoryDoubleS<Group>>(path_, constraints);
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
    double BlendTolerance() const { return path_->GetBlendTolerance(); }
    const std::string &Profile() const { return profile_; }
    Eigen::VectorXd MaxVelocity() const {
        return trajectory_->GetMaxVelocity();
    }
    Eigen::VectorXd MaxAcceleration() const {
        return trajectory_->GetMaxAcceleration();
    }
    Eigen::VectorXd MaxJerk() const { return trajectory_->GetMaxJerk(); }
    double PathLength() const { return path_->GetLength(); }
    Eigen::MatrixXd Waypoints() const {
        const auto &waypoints = path_->GetWaypoints();
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
    py::tuple Sample(const Eigen::VectorXd &times) const {
        const auto states = SampleValues(times);
        return py::make_tuple(states[0], states[1], states[2], states[3]);
    }
    py::tuple SampleUniform(std::size_t samples) const {
        ValidateUniformSampleCount(samples, N);
        auto times = NewDoubleArray<1>({static_cast<Py_intptr_t>(samples)},
                                       {sizeof(double)});
        Eigen::Map<Eigen::VectorXd> time_values(times.mutable_data(), samples);
        time_values = Eigen::VectorXd::LinSpaced(
            static_cast<Eigen::Index>(samples), 0.0, Duration());
        const auto states = SampleValues(time_values);
        return py::make_tuple(times, states[0], states[1], states[2],
                              states[3]);
    }
    py::tuple State(double time) const {
        ValidateTime(time);
        const auto state = trajectory_->GetState(time);
        return py::make_tuple(CopyVector(state.position.Coeffs()),
                              CopyVector(state.velocity.Coeffs()),
                              CopyVector(state.acceleration.Coeffs()),
                              CopyVector(state.jerk.Coeffs()));
    }
    py::dict ConstraintReport(std::size_t samples,
                              const ReportKeys &keys) const {
        const auto report = trajectory_->GetConstraintReport(samples);
        py::dict result;
        result[keys.peak_velocity] = CopyVector(report.peak_velocity);
        result[keys.peak_acceleration] = CopyVector(report.peak_acceleration);
        result[keys.peak_jerk] = CopyVector(report.peak_jerk);
        result[keys.velocity_utilization] =
            CopyVector(report.velocity_utilization);
        result[keys.acceleration_utilization] =
            CopyVector(report.acceleration_utilization);
        result[keys.jerk_utilization] = CopyVector(report.jerk_utilization);
        result[keys.maximum_velocity_jump] =
            CopyVector(report.maximum_velocity_jump);
        result[keys.maximum_acceleration_jump] =
            CopyVector(report.maximum_acceleration_jump);
        result[keys.maximum_utilization] = report.maximum_utilization;
        result[keys.within_limits] = report.within_limits;
        result[keys.velocity_continuous] = report.velocity_continuous;
        result[keys.acceleration_continuous] = report.acceleration_continuous;
        return result;
    }

  private:
    SampleArrays
    SampleValues(const Eigen::Ref<const Eigen::VectorXd> &times) const {
        if (!times.allFinite()) {
            throw std::invalid_argument("trajectory times must be finite");
        }
        // Fill NumPy-owned column-major buffers directly. Returning temporary
        // Eigen matrices would copy every sample again while holding the GIL.
        const std::array<Py_intptr_t, 2> shape{times.size(), N};
        const std::array<Py_intptr_t, 2> strides{
            sizeof(double),
            static_cast<Py_intptr_t>(sizeof(double)) * times.size()};
        const auto allocate = [&] {
            return NewDoubleArray<2, py::array::f_style>(shape, strides);
        };
        SampleArray positions = allocate(), velocities = allocate(),
                    accelerations = allocate(), jerks = allocate();
        Eigen::Map<Eigen::MatrixXd> position_values(positions.mutable_data(),
                                                    times.size(), N);
        Eigen::Map<Eigen::MatrixXd> velocity_values(velocities.mutable_data(),
                                                    times.size(), N);
        Eigen::Map<Eigen::MatrixXd> acceleration_values(
            accelerations.mutable_data(), times.size(), N);
        Eigen::Map<Eigen::MatrixXd> jerk_values(jerks.mutable_data(),
                                                times.size(), N);
        {
            py::gil_scoped_release release;
            detail::TrajectoryStateSampler<Group> sampler(*trajectory_);
            for (Eigen::Index index = 0; index < times.size(); ++index) {
                const auto state = sampler.GetState(times[index]);
                position_values.row(index) =
                    state.position.Coeffs().transpose();
                velocity_values.row(index) =
                    state.velocity.Coeffs().transpose();
                acceleration_values.row(index) =
                    state.acceleration.Coeffs().transpose();
                jerk_values.row(index) = state.jerk.Coeffs().transpose();
            }
        }
        return {std::move(positions), std::move(velocities),
                std::move(accelerations), std::move(jerks)};
    }
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
void BindJointTrajectory(py::module_ &module, const char *name,
                         const std::shared_ptr<const ReportKeys> &report_keys) {
    py::class_<JointTrajectory<N>>(module, name,
                                   "Fifth-order Bezier joint path with "
                                   "selectable Double-S or trapezoidal timing.")
        .def(py::init([](const py::object &waypoints,
                         const py::object &max_velocity,
                         const py::object &max_acceleration,
                         const py::object &max_jerk,
                         const py::object &blend_tolerance,
                         const py::object &minimum_duration,
                         const std::string &profile) {
                 return std::make_unique<JointTrajectory<N>>(
                     CheckedRealEigen<Eigen::MatrixXd>(waypoints, "waypoints"),
                     CheckedRealEigen<Eigen::VectorXd>(max_velocity,
                                                       "max_velocity"),
                     CheckedRealEigen<Eigen::VectorXd>(max_acceleration,
                                                       "max_acceleration"),
                     CheckedRealEigen<Eigen::VectorXd>(max_jerk, "max_jerk"),
                     CheckedRealEigen<double>(blend_tolerance,
                                              "blend_tolerance"),
                     CheckedRealEigen<double>(minimum_duration,
                                              "minimum_duration"),
                     profile);
             }),
             py::arg("waypoints"), py::arg("max_velocity"),
             py::arg("max_acceleration"), py::arg("max_jerk"),
             py::arg("blend_tolerance") = 0.0,
             py::arg("minimum_duration") = 0.0, py::arg("profile") = "double_s")
        .def_property_readonly("duration", &JointTrajectory<N>::Duration)
        .def_property_readonly("dof", &JointTrajectory<N>::Dof)
        .def_property_readonly("time_scale", &JointTrajectory<N>::TimeScale)
        .def_property_readonly("blend_tolerance",
                               &JointTrajectory<N>::BlendTolerance)
        .def_property_readonly("profile", &JointTrajectory<N>::Profile)
        .def_property_readonly("breakpoints", &JointTrajectory<N>::Breakpoints)
        .def_property_readonly("max_velocity", &JointTrajectory<N>::MaxVelocity)
        .def_property_readonly("max_acceleration",
                               &JointTrajectory<N>::MaxAcceleration)
        .def_property_readonly("max_jerk", &JointTrajectory<N>::MaxJerk)
        .def_property_readonly("path_length", &JointTrajectory<N>::PathLength)
        .def_property_readonly("waypoints", &JointTrajectory<N>::Waypoints)
        .def("position", WithRealScalar(&JointTrajectory<N>::Position, "time"),
             py::arg("time"),
             "Evaluate joint position; finite times are clamped to the "
             "trajectory.")
        .def("velocity", WithRealScalar(&JointTrajectory<N>::Velocity, "time"),
             py::arg("time"), "Evaluate joint velocity at a finite time.")
        .def("acceleration",
             WithRealScalar(&JointTrajectory<N>::Acceleration, "time"),
             py::arg("time"), "Evaluate joint acceleration at a finite time.")
        .def("jerk", WithRealScalar(&JointTrajectory<N>::Jerk, "time"),
             py::arg("time"), "Evaluate joint jerk at a finite time.")
        .def(
            "set_minimum_duration",
            WithRealScalar(&JointTrajectory<N>::SetMinimumDuration, "duration"),
            py::arg("duration"),
            "Slow the trajectory to at least duration seconds; never speeds "
            "it up.")
        .def(
            "sample",
            [](const JointTrajectory<N> &trajectory, const py::object &times) {
                return trajectory.Sample(
                    CheckedRealEigen<Eigen::VectorXd>(times, "times"));
            },
            py::arg("times"),
            "Return position, velocity, acceleration, and jerk matrices for "
            "1-D times.")
        .def("sample_uniform",
             WithRealScalar(&JointTrajectory<N>::SampleUniform, "samples"),
             py::arg("samples") = 1001,
             "Uniformly sample the inclusive [0, duration] interval and "
             "return time plus four state matrices.")
        .def("state", WithRealScalar(&JointTrajectory<N>::State, "time"),
             py::arg("time"),
             "Return position, velocity, acceleration, and jerk at one time.")
        .def(
            "constraint_report",
            [report_keys](const JointTrajectory<N> &trajectory,
                          const py::object &samples) {
                return trajectory.ConstraintReport(
                    CheckedRealEigen<std::size_t>(samples, "samples"),
                    *report_keys);
            },
            py::arg("samples") = 2001,
            "Return sampled per-joint peaks, limit utilization, and "
            "breakpoint continuity diagnostics.");
}

template <int N>
py::object MakeJointTrajectory(const Eigen::MatrixXd &waypoints,
                               const Eigen::VectorXd &max_velocity,
                               const Eigen::VectorXd &max_acceleration,
                               const Eigen::VectorXd &max_jerk,
                               double blend_tolerance, double minimum_duration,
                               const std::string &profile) {
    return py::cast(std::make_unique<JointTrajectory<N>>(
        waypoints, max_velocity, max_acceleration, max_jerk, blend_tolerance,
        minimum_duration, profile));
}

template <int Begin, std::size_t... Offsets>
JointTrajectoryFactoryBlock
BindJointTrajectoryRange(py::module_ &module,
                         const std::shared_ptr<const ReportKeys> &keys,
                         std::index_sequence<Offsets...>) {
    (BindJointTrajectory<Begin + Offsets>(
         module, ("JointTrajectory" + std::to_string(Begin + Offsets)).c_str(),
         keys),
     ...);
    return {&MakeJointTrajectory<Begin + Offsets>...};
}

} // namespace holistic_motion::python::trajectory_detail
