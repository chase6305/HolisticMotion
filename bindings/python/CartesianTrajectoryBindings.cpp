#include "TrajectoryBindingHelpers.h"

namespace holistic_motion::python::trajectory_detail {

class CartesianLineTrajectory {
  public:
    CartesianLineTrajectory(const Eigen::Matrix4d &start,
                            const Eigen::Matrix4d &end,
                            const Eigen::VectorXd &max_velocity,
                            const Eigen::VectorXd &max_acceleration,
                            const Eigen::VectorXd &max_jerk,
                            double minimum_duration,
                            const std::string &profile) {
        ValidateTransform(start, "start");
        ValidateTransform(end, "end");
        if (max_velocity.size() != 6 || max_acceleration.size() != 6 ||
            max_jerk.size() != 6) {
            throw std::invalid_argument(
                "Cartesian limits must contain 6 values");
        }
        const std::vector<SE3d> points{SE3d(start), SE3d(end)};
        path_ = std::make_shared<PathBezierCurve<SE3d>>(points, 5, true, 0.0);
        auto constraints = std::make_shared<TrajectoryConstraints>(
            max_velocity, max_acceleration, max_jerk);
        if (!constraints->IsValid()) {
            throw std::invalid_argument(
                "Cartesian limits must be finite and positive");
        }
        if (profile == "double_s") {
            profile_ = profile;
            trajectory_ =
                std::make_shared<TrajectoryDoubleS<SE3d>>(path_, constraints);
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
    const std::string &Profile() const { return profile_; }
    Eigen::Matrix4d Position(double time) const {
        ValidateTime(time);
        return trajectory_->GetPosition(time).GetTransform();
    }
    py::tuple State(double time) const {
        ValidateTime(time);
        const auto state = trajectory_->GetState(time);
        return py::make_tuple(state.position.GetTransform(),
                              CopyVector(state.velocity.Coeffs()),
                              CopyVector(state.acceleration.Coeffs()),
                              CopyVector(state.jerk.Coeffs()));
    }
    py::tuple SampleUniform(std::size_t samples) const {
        ValidateUniformSampleCount(samples, 16);
        auto times = NewDoubleArray<1>({static_cast<Py_intptr_t>(samples)},
                                       {sizeof(double)});
        Eigen::Map<Eigen::VectorXd> time_values(times.mutable_data(), samples);
        time_values = Eigen::VectorXd::LinSpaced(
            static_cast<Eigen::Index>(samples), 0.0, Duration());
        auto poses = NewDoubleArray<3>(
            {static_cast<Py_intptr_t>(samples), 4, 4},
            {16 * sizeof(double), 4 * sizeof(double), sizeof(double)});
        using SampleArray = py::array_t<double, py::array::f_style>;
        const std::array<Py_intptr_t, 2> shape{
            static_cast<Py_intptr_t>(samples), 6};
        const std::array<Py_intptr_t, 2> strides{
            sizeof(double), static_cast<Py_intptr_t>(sizeof(double) * samples)};
        const auto allocate = [&] {
            return NewDoubleArray<2, py::array::f_style>(shape, strides);
        };
        SampleArray velocity = allocate(), acceleration = allocate(),
                    jerk = allocate();
        Eigen::Map<Eigen::MatrixXd> velocity_values(velocity.mutable_data(),
                                                    samples, 6);
        Eigen::Map<Eigen::MatrixXd> acceleration_values(
            acceleration.mutable_data(), samples, 6);
        Eigen::Map<Eigen::MatrixXd> jerk_values(jerk.mutable_data(), samples,
                                                6);
        auto pose_view = poses.mutable_unchecked<3>();
        {
            py::gil_scoped_release release;
            detail::TrajectoryStateSampler<SE3d> sampler(*trajectory_);
            for (std::size_t index = 0; index < samples; ++index) {
                const auto state = sampler.GetState(time_values[index]);
                const Eigen::Matrix4d pose = state.position.GetTransform();
                for (int row = 0; row < 4; ++row)
                    for (int column = 0; column < 4; ++column)
                        pose_view(index, row, column) = pose(row, column);
                velocity_values.row(index) =
                    state.velocity.Coeffs().transpose();
                acceleration_values.row(index) =
                    state.acceleration.Coeffs().transpose();
                jerk_values.row(index) = state.jerk.Coeffs().transpose();
            }
        }
        return py::make_tuple(times, poses, velocity, acceleration, jerk);
    }
    py::dict ConstraintReport(std::size_t samples,
                              const ReportKeys &keys) const {
        const auto report = trajectory_->GetConstraintReport(samples);
        py::dict result;
        result[keys.peak_velocity] = CopyVector(report.peak_velocity);
        result[keys.peak_acceleration] = CopyVector(report.peak_acceleration);
        result[keys.peak_jerk] = CopyVector(report.peak_jerk);
        result[keys.maximum_utilization] = report.maximum_utilization;
        result[keys.within_limits] = report.within_limits;
        result[keys.velocity_continuous] = report.velocity_continuous;
        result[keys.acceleration_continuous] = report.acceleration_continuous;
        return result;
    }

  private:
    static void ValidateTime(double time) {
        if (!std::isfinite(time))
            throw std::invalid_argument("trajectory time must be finite");
    }
    static void ValidateTransform(const Eigen::Matrix4d &transform,
                                  const char *name) {
        const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
        if (!transform.allFinite() ||
            !transform.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0),
                                       1e-9) ||
            !(rotation.transpose() * rotation)
                 .isApprox(Eigen::Matrix3d::Identity(), 1e-7) ||
            std::abs(rotation.determinant() - 1.0) > 1e-7) {
            throw std::invalid_argument(std::string(name) +
                                        " must be a finite rigid transform");
        }
    }

    std::shared_ptr<PathBezierCurve<SE3d>> path_;
    std::shared_ptr<TrajectoryBase<SE3d>> trajectory_;
    std::string profile_;
};

void BindCartesianTrajectory(
    py::module_ &module, const std::shared_ptr<const ReportKeys> &report_keys) {
    py::class_<CartesianLineTrajectory>(module, "CartesianLineTrajectory")
        .def(py::init([](const py::object &start, const py::object &end,
                         const py::object &max_velocity,
                         const py::object &max_acceleration,
                         const py::object &max_jerk,
                         const py::object &minimum_duration,
                         const std::string &profile) {
                 return std::make_unique<CartesianLineTrajectory>(
                     CheckedRealEigen<Eigen::Matrix4d>(start, "start"),
                     CheckedRealEigen<Eigen::Matrix4d>(end, "end"),
                     CheckedRealEigen<Eigen::VectorXd>(max_velocity,
                                                       "max_velocity"),
                     CheckedRealEigen<Eigen::VectorXd>(max_acceleration,
                                                       "max_acceleration"),
                     CheckedRealEigen<Eigen::VectorXd>(max_jerk, "max_jerk"),
                     CheckedRealEigen<double>(minimum_duration,
                                              "minimum_duration"),
                     profile);
             }),
             py::arg("start"), py::arg("end"), py::arg("max_velocity"),
             py::arg("max_acceleration"), py::arg("max_jerk"),
             py::arg("minimum_duration") = 0.0, py::arg("profile") = "double_s")
        .def_property_readonly("duration", &CartesianLineTrajectory::Duration)
        .def_property_readonly("profile", &CartesianLineTrajectory::Profile)
        .def("position",
             WithRealScalar(&CartesianLineTrajectory::Position, "time"),
             py::arg("time"))
        .def("state", WithRealScalar(&CartesianLineTrajectory::State, "time"),
             py::arg("time"))
        .def("sample_uniform",
             WithRealScalar(&CartesianLineTrajectory::SampleUniform, "samples"),
             py::arg("samples") = 1001)
        .def(
            "constraint_report",
            [report_keys](const CartesianLineTrajectory &trajectory,
                          const py::object &samples) {
                return trajectory.ConstraintReport(
                    CheckedRealEigen<std::size_t>(samples, "samples"),
                    *report_keys);
            },
            py::arg("samples") = 2001);
}

} // namespace holistic_motion::python::trajectory_detail
