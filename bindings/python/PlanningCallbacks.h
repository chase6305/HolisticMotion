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
class PlanningPythonFunction {
  public:
    explicit PlanningPythonFunction(py::function function)
        : function_(std::move(function)) {}
    PlanningPythonFunction(const PlanningPythonFunction &other) {
        py::gil_scoped_acquire acquire;
        function_ = other.function_;
    }
    PlanningPythonFunction(PlanningPythonFunction &&other) noexcept
        : function_(std::move(other.function_)) {}
    ~PlanningPythonFunction() {
        if (function_) {
            py::gil_scoped_acquire acquire;
            py::function release(std::move(function_));
        }
    }
    PyObject *Pointer() const noexcept { return function_.ptr(); }
    py::object Call(const Eigen::VectorXd &state) const { return function_(state); }

  private:
    py::function function_;
};

template <class Result> struct PlanningPythonCallback {
    PlanningPythonFunction function;
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
        PlanningPythonFunction(callback.cast<py::function>()),
        "state_validator result"};
}

template <class Result>
int VisitPlanningCallback(
    const std::function<Result(const Eigen::VectorXd &)> &callback, visitproc visit,
    void *arg) {
    if (const auto *value =
            callback.template target<PlanningPythonCallback<Result>>()) {
        PyObject *function = value->function.Pointer();
        Py_VISIT(function);
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
