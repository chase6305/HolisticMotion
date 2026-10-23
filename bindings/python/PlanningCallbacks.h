#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

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

inline std::vector<py::detail::type_info *>
PlanningTypeInfoWithoutCaching(PyTypeObject *type) {
    const auto &registered = py::detail::get_internals().registered_types_py;
    std::vector<py::detail::type_info *> result;
    std::vector<PyTypeObject *> pending{type};
    for (std::size_t index = 0; index < pending.size(); ++index) {
        auto *current = pending[index];
        const auto cached = registered.find(current);
        if (cached != registered.end() && !cached->second.empty()) {
            for (auto *info : cached->second) {
                if (std::find(result.begin(), result.end(), info) == result.end())
                    py::detail::all_type_info_add_base_most_derived_first(result,
                                                                        info);
            }
        } else if (current->tp_bases && PyTuple_Check(current->tp_bases)) {
            for (Py_ssize_t base = 0; base < PyTuple_GET_SIZE(current->tp_bases);
                 ++base)
                pending.push_back(reinterpret_cast<PyTypeObject *>(
                    PyTuple_GET_ITEM(current->tp_bases, base)));
        }
    }
    return result;
}

template <class Planner> Planner *OwnedPlanningValue(PyObject *self) {
    auto *instance = reinterpret_cast<py::detail::instance *>(self);
    if (!instance->owned)
        return nullptr;
    // Type weak references may have been cleared before GC calls tp_clear.
    // Do not recreate that cache (or Python references) inside a collector hook.
    // Resurrected owners still need their native edges traced on later cycles.
    const auto &types = py::detail::get_internals().registered_types_py;
    const auto cached = types.find(Py_TYPE(self));
    const auto fallback = cached == types.end() || cached->second.empty()
                              ? PlanningTypeInfoWithoutCaching(Py_TYPE(self))
                              : std::vector<py::detail::type_info *>{};
    const auto &metadata = fallback.empty() && cached != types.end()
                               ? cached->second
                               : fallback;
    const auto *info = py::detail::get_type_info(typeid(Planner));
    if (!info)
        return nullptr;
    std::size_t offset = 0;
    for (std::size_t index = 0; index < metadata.size(); ++index) {
        if (metadata[index] == info) {
            const py::detail::value_and_holder value(instance, info, offset, index);
            if (!value.holder_constructed() || !value.instance_registered())
                return nullptr;
            return value.template value_ptr<Planner>();
        }
        offset += 1 + metadata[index]->holder_size_in_ptrs;
    }
    return nullptr;
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
#if PY_VERSION_HEX >= 0x030B0000 && PY_VERSION_HEX < 0x030C0000 && !defined(PYPY_VERSION)
        // CPython 3.11's subtype_traverse visits managed dictionaries even
        // when the native base also visits them. Use a traditional dictionary
        // slot so Python subclasses do not report that ownership edge twice.
        if (PyType_HasFeature(type, Py_TPFLAGS_MANAGED_DICT)) {
            type->tp_flags &= ~Py_TPFLAGS_MANAGED_DICT;
            type->tp_dictoffset = type->tp_basicsize;
            type->tp_basicsize += static_cast<Py_ssize_t>(sizeof(PyObject *));
        }
#endif
        previous_traverse = type->tp_traverse;
        previous_clear = type->tp_clear;
        type->tp_traverse = Traverse;
        type->tp_clear = Clear;
    }
};

} // namespace holistic_motion::python
