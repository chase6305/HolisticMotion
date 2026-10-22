#include "Bindings.h"
#include "PlanningCollisionMethods.h"

#ifdef HOLISTICMOTION_HAS_COLLISION
#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"
#endif

namespace holistic_motion::python {

void BindPlanningCollisionLoader(pybind11::module_ &module) {
#ifdef HOLISTICMOTION_HAS_COLLISION
    namespace py = pybind11;
    using robotics::planning::PathOptimizer;
    using robotics::planning::SamplingPlanner;
    auto optimizer =
        py::reinterpret_borrow<py::class_<PathOptimizer>>(module.attr("PathOptimizer"));
    auto planner = py::reinterpret_borrow<py::class_<SamplingPlanner>>(
        module.attr("SamplingPlanner"));
    const auto deferred = [](const char *type, const char *method) {
        return [type, method](py::args args, py::kwargs kwargs) {
            const auto collision =
                py::module_::import("holistic_motion").attr("_load_collision")();
            return collision.attr(type).attr(method)(*args, **kwargs);
        };
    };
    for (const auto *method : kOptimizerCollisionMethods)
        optimizer.def_static(method, deferred("PathOptimizer", method));
    for (const auto *method : kPlannerCollisionMethods)
        planner.def_static(method, deferred("SamplingPlanner", method));
#else
    (void)module;
#endif
}

} // namespace holistic_motion::python
