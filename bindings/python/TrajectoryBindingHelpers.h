#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../src/trajectory/TrajectoryStateSampler.h"
#include "Bindings.h"
#include "RealInputs.h"
#include "holistic_motion/trajectory/PathBezierCurve.h"
#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "holistic_motion/trajectory/TrajectoryTrapezoidal.h"

namespace holistic_motion::python::trajectory_detail {

namespace py = pybind11;
using namespace holistic_motion::robotics;

// NumPy copies these fixed-size descriptors and allocates its own writable
// data. Use pybind11's NumPy API loader so this small bridge needs no NumPy
// headers and follows its NumPy 1/2 compatibility handling. The ordinary array
// constructor materializes heap-backed shape and stride containers first.
template <std::size_t Dimensions, int Flags = py::array::forcecast>
py::array_t<double, Flags>
NewDoubleArray(const std::array<Py_intptr_t, Dimensions> &shape,
               const std::array<Py_intptr_t, Dimensions> &strides) {
    auto dtype = py::dtype::of<double>();
    auto &api = py::detail::npy_api::get();
    auto result = py::reinterpret_steal<py::array_t<double, Flags>>(
        // NewFromDescr consumes the dtype reference, including on failure.
        api.PyArray_NewFromDescr_(api.PyArray_Type_, dtype.release().ptr(),
                                  static_cast<int>(Dimensions), shape.data(),
                                  strides.data(), nullptr, 0, nullptr));
    if (!result)
        throw py::error_already_set();
    return result;
}

// Allocate the owning NumPy buffer directly; results remain independent
// and writable after the temporary C++ state or report is destroyed.
template <int Rows>
py::array_t<double> CopyVector(const Eigen::Matrix<double, Rows, 1> &values) {
    auto result = NewDoubleArray<1>({values.size()}, {sizeof(double)});
    std::copy_n(values.data(), values.size(), result.mutable_data());
    return result;
}

// Binding closures own these immutable keys for the module's lifetime. Keeping
// them out of static storage also keeps destruction under Python's GIL.
struct ReportKeys {
    const py::str peak_velocity{"peak_velocity"};
    const py::str peak_acceleration{"peak_acceleration"};
    const py::str peak_jerk{"peak_jerk"};
    const py::str velocity_utilization{"velocity_utilization"};
    const py::str acceleration_utilization{"acceleration_utilization"};
    const py::str jerk_utilization{"jerk_utilization"};
    const py::str maximum_velocity_jump{"maximum_velocity_jump"};
    const py::str maximum_acceleration_jump{"maximum_acceleration_jump"};
    const py::str maximum_utilization{"maximum_utilization"};
    const py::str within_limits{"within_limits"};
    const py::str velocity_continuous{"velocity_continuous"};
    const py::str acceleration_continuous{"acceleration_continuous"};
};

using holistic_motion::python::CheckedRealEigen;
using holistic_motion::python::ReadParameter;
using holistic_motion::python::RequireRealInput;
using holistic_motion::python::WriteRealParameter;

template <class Class, class Result, class Value>
auto WithRealScalar(Result (Class::*method)(Value) const, const char *name) {
    return [method, name](const Class &self, const py::object &input) {
        return (self.*method)(CheckedRealEigen<Value>(input, name));
    };
}

template <class Class, class Result, class Value>
auto WithRealScalar(Result (Class::*method)(Value), const char *name) {
    return [method, name](Class &self, const py::object &input) {
        return (self.*method)(CheckedRealEigen<Value>(input, name));
    };
}

inline void ValidateUniformSampleCount(std::size_t samples,
                                       std::size_t components) {
    if (samples < 2) {
        throw std::invalid_argument(
            "uniform trajectory sampling requires at least 2 samples");
    }
    // Both Eigen indexing and NumPy's signed byte counts must represent each
    // output array. Divide before multiplying dimensions or allocating times.
    const auto maximum = std::min(
        static_cast<std::size_t>((std::numeric_limits<Eigen::Index>::max)()),
        static_cast<std::size_t>((std::numeric_limits<py::ssize_t>::max)()));
    if (samples > maximum / sizeof(double) / components) {
        throw std::invalid_argument("sample count is too large");
    }
}

template <typename Factory> auto CheckedTrajectoryFactory(Factory factory) {
    return [factory](
               const py::object &waypoints, const py::object &max_velocity,
               const py::object &max_acceleration, const py::object &max_jerk,
               const py::object &blend_tolerance,
               const py::object &minimum_duration, const std::string &profile) {
        return factory(
            CheckedRealEigen<Eigen::MatrixXd>(waypoints, "waypoints"),
            CheckedRealEigen<Eigen::VectorXd>(max_velocity, "max_velocity"),
            CheckedRealEigen<Eigen::VectorXd>(max_acceleration,
                                              "max_acceleration"),
            CheckedRealEigen<Eigen::VectorXd>(max_jerk, "max_jerk"),
            CheckedRealEigen<double>(blend_tolerance, "blend_tolerance"),
            CheckedRealEigen<double>(minimum_duration, "minimum_duration"),
            profile);
    };
}

using JointTrajectoryFactory = py::object (*)(const Eigen::MatrixXd &,
                                              const Eigen::VectorXd &,
                                              const Eigen::VectorXd &,
                                              const Eigen::VectorXd &, double,
                                              double, const std::string &);
using JointTrajectoryFactoryBlock = std::array<JointTrajectoryFactory, 8>;

JointTrajectoryFactoryBlock
BindJointTrajectories1To8(py::module_ &,
                          const std::shared_ptr<const ReportKeys> &);
JointTrajectoryFactoryBlock
BindJointTrajectories9To16(py::module_ &,
                           const std::shared_ptr<const ReportKeys> &);
JointTrajectoryFactoryBlock
BindJointTrajectories17To24(py::module_ &,
                            const std::shared_ptr<const ReportKeys> &);
JointTrajectoryFactoryBlock
BindJointTrajectories25To32(py::module_ &,
                            const std::shared_ptr<const ReportKeys> &);
void BindCartesianTrajectory(py::module_ &,
                             const std::shared_ptr<const ReportKeys> &);

} // namespace holistic_motion::python::trajectory_detail
