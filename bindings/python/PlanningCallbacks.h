#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#include <pybind11/functional.h>

#include "RealInputs.h"
#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"

namespace holistic_motion::robotics::planning::detail {

// Keep Python ownership mechanics in the binding. The core only grants access
// to its callback wrappers; it does not depend on Python or its collector.
struct PlanningCallbackAccess {
    static const auto &Validator(const SamplingPlanner &value) {
        return value.validator_;
    }
    static const auto &Validator(const PathOptimizer &value) {
        return value.validator_;
    }
    static const auto &Cost(const PathOptimizer &value) { return value.state_cost_; }
    static const auto &Gradient(const PathOptimizer &value) {
        return value.state_cost_gradient_;
    }
    static void Clear(SamplingPlanner &value) { value.SetStateValidator({}); }
    static void Clear(PathOptimizer &value) {
        value.SetStateValidator({});
        value.SetStateCost({});
        value.SetStateCostGradient({});
    }
};

} // namespace holistic_motion::robotics::planning::detail

namespace holistic_motion::python {
namespace py = pybind11;

// Each C++ copy owns one Python reference, so tp_traverse can account for each
// edge exactly once. Request copies that are outside the Python object graph
// keep callbacks alive even if a reentrant setter replaces the original.
class PlanningPythonReference {
  public:
    explicit PlanningPythonReference(py::object value) : object_(std::move(value)) {}
    PlanningPythonReference(const PlanningPythonReference &other) {
        py::gil_scoped_acquire acquire;
        object_ = other.object_;
    }
    PlanningPythonReference(PlanningPythonReference &&other) noexcept
        : object_(std::move(other.object_)) {}
    ~PlanningPythonReference() {
        if (object_) {
            py::gil_scoped_acquire acquire;
            py::object release(std::move(object_));
        }
    }
    PyObject *Pointer() const noexcept { return object_.ptr(); }
    py::object Call(const Eigen::VectorXd &state) const { return object_(state); }

  private:
    py::object object_;
};

template <class Result> struct PlanningPythonCallback {
    PlanningPythonReference function;
    const char *label;
    Result operator()(const Eigen::VectorXd &state) const {
        py::gil_scoped_acquire acquire;
        if constexpr (std::is_same_v<Result, bool>)
            return function.Call(state).template cast<bool>();
        else
            return CheckedRealEigen<Result>(function.Call(state), label);
    }
};

inline robotics::planning::SamplingPlanner::StateValidator
PlanningValidator(const py::object &callback) {
    if (callback.is_none())
        return {};
    using Validator = robotics::planning::SamplingPlanner::StateValidator;
    using NativeFunction = bool (*)(const Eigen::VectorXd &);
    auto converted = callback.cast<Validator>();
    if (converted.target<NativeFunction>())
        return converted;
    return PlanningPythonCallback<bool>{
        PlanningPythonReference(callback.cast<py::function>()),
        "state_validator result"};
}

// Export the functor type's RTTI: factories live in the collision extension,
// while the core extension traverses these std::function targets during GC.
// Each copy owns its resource, including request copies outside Python's graph.
template <class Result> struct PYBIND11_EXPORT PlanningResourceCallback {
    // Destroy callback captures before releasing the model they may reference.
    PlanningPythonReference resource;
    std::function<Result(const Eigen::VectorXd &)> callback;
    Result operator()(const Eigen::VectorXd &state) const { return callback(state); }
};

template <class Result>
std::function<Result(const Eigen::VectorXd &)> RetainPlanningCallbackResource(
    const std::function<Result(const Eigen::VectorXd &)> &callback,
    const py::object &resource) {
    if (!callback)
        return {};
    return PlanningResourceCallback<Result>{PlanningPythonReference(resource),
                                            callback};
}

template <class Planner>
Planner RetainPlanningResource(Planner result, const py::object &resource) {
    using namespace robotics::planning;
    using Access = robotics::planning::detail::PlanningCallbackAccess;
    result.SetStateValidator(
        RetainPlanningCallbackResource(Access::Validator(result), resource));
    if constexpr (std::is_same_v<Planner, PathOptimizer>) {
        result.SetStateCost(
            RetainPlanningCallbackResource(Access::Cost(result), resource));
        result.SetStateCostGradient(
            RetainPlanningCallbackResource(Access::Gradient(result), resource));
    }
    return result;
}

template <class Result>
int VisitPlanningCallback(
    const std::function<Result(const Eigen::VectorXd &)> &callback, visitproc visit,
    void *arg) {
    if (const auto *value =
            callback.template target<PlanningPythonCallback<Result>>()) {
        PyObject *function = value->function.Pointer();
        Py_VISIT(function);
    } else if (const auto *value =
                   callback.template target<PlanningResourceCallback<Result>>()) {
        PyObject *resource = value->resource.Pointer();
        Py_VISIT(resource);
        return VisitPlanningCallback(value->callback, visit, arg);
    }
    return 0;
}

template <class Planner> Planner *OwnedPlanningValue(PyObject *self) {
    auto *instance = reinterpret_cast<py::detail::instance *>(self);
    if (!instance->owned)
        return nullptr;
    const auto *info = py::detail::get_type_info(typeid(Planner));
    if (!info)
        return nullptr;
    const auto value = instance->get_value_and_holder(info, false);
    if (!value.inst || !value.holder_constructed() || !value.instance_registered())
        return nullptr;
    return value.template value_ptr<Planner>();
}

template <class Planner> struct PlanningGarbageCollection {
    inline static traverseproc previous_traverse = nullptr;
    inline static inquiry previous_clear = nullptr;

    static int Traverse(PyObject *self, visitproc visit, void *arg) {
        if (previous_traverse) {
            const int status = previous_traverse(self, visit, arg);
            if (status)
                return status;
        }
        using namespace robotics::planning;
        using Access = robotics::planning::detail::PlanningCallbackAccess;
        // Python multiple inheritance can place both native bases in one
        // instance. Its primary base must trace both sets of callback edges.
        if (const auto *value = OwnedPlanningValue<SamplingPlanner>(self)) {
            const int status =
                VisitPlanningCallback(Access::Validator(*value), visit, arg);
            if (status)
                return status;
        }
        if (const auto *value = OwnedPlanningValue<PathOptimizer>(self)) {
            int status = VisitPlanningCallback(Access::Validator(*value), visit, arg);
            if (status)
                return status;
            status = VisitPlanningCallback(Access::Cost(*value), visit, arg);
            if (status)
                return status;
            status = VisitPlanningCallback(Access::Gradient(*value), visit, arg);
            if (status)
                return status;
        }
        return 0;
    }
    static int Clear(PyObject *self) {
        using namespace robotics::planning;
        using Access = robotics::planning::detail::PlanningCallbackAccess;
        if (auto *value = OwnedPlanningValue<SamplingPlanner>(self))
            Access::Clear(*value);
        if (auto *value = OwnedPlanningValue<PathOptimizer>(self))
            Access::Clear(*value);
        return previous_clear ? previous_clear(self) : 0;
    }
    static void Setup(PyHeapTypeObject *heap_type) {
        auto *type = &heap_type->ht_type;
        previous_traverse = type->tp_traverse;
        previous_clear = type->tp_clear;
        type->tp_traverse = Traverse;
        type->tp_clear = Clear;
    }
};

} // namespace holistic_motion::python
