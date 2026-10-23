#include "Bindings.h"
#include "PlanningCollisionMethods.h"
#include "utility/BuildIdentity.h"

#include <cstring>

// Queryable before PyInit registers any C++ types. The Python loader can reject
// older or mismatched binaries without passing C++ values between components.
extern "C" PYBIND11_EXPORT const char *HolisticMotionCollisionBuildIdentity() {
    return HOLISTICMOTION_NATIVE_BUILD_ID;
}

PYBIND11_MODULE(_holistic_motion_collision, module) {
    module.doc() = "Optional native collision queries and planning adapters";
    namespace py = pybind11;
    const auto core =
        py::module_::import("holistic_motion.runtime").attr("_native_module")();
    if (core.is_none())
        throw py::import_error("load holistic_motion before its collision extension");
    if (!py::hasattr(core, "_native_build_id") ||
        !py::isinstance<py::str>(core.attr("_native_build_id")) ||
        core.attr("_native_build_id").cast<std::string>() !=
            HOLISTICMOTION_NATIVE_BUILD_ID ||
        std::strcmp(holistic_motion::utility::detail::CoreBuildIdentity(),
                    HOLISTICMOTION_NATIVE_BUILD_ID) != 0 ||
        std::strcmp(
            holistic_motion::robotics::collision::detail::CollisionBuildIdentity(),
            HOLISTICMOTION_NATIVE_BUILD_ID) != 0)
        throw py::import_error(
            "HolisticMotion's core and collision extensions have different native "
            "build identities. Rebuild and install all native components together.");
    module.attr("_native_build_id") = HOLISTICMOTION_NATIVE_BUILD_ID;
    module.attr("SamplingPlanner") = core.attr("SamplingPlanner");
    module.attr("PathOptimizer") = core.attr("PathOptimizer");
    holistic_motion::python::BindCollision(module);
    // Remove catch-all deferred wrappers before registering typed overloads.
    // Leaving them first in an overload chain would recurse into the loader.
    for (const auto *name : holistic_motion::python::kOptimizerCollisionMethods)
        py::delattr(module.attr("PathOptimizer"), name);
    for (const auto *name : holistic_motion::python::kPlannerCollisionMethods)
        py::delattr(module.attr("SamplingPlanner"), name);
    holistic_motion::python::BindPlanningCollision(module);
}
