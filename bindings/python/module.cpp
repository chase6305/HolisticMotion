#include "Bindings.h"
#include "utility/BuildIdentity.h"

#include <cstring>

PYBIND11_MODULE(_holistic_motion, module) {
    module.doc() = "Robot models, kinematics, manifolds, and trajectories";
    if (std::strcmp(holistic_motion::utility::detail::CoreBuildIdentity(),
                    HOLISTICMOTION_NATIVE_BUILD_ID) != 0)
        throw pybind11::import_error(
            "HolisticMotion's core library and Python extension have different native "
            "build identities. Rebuild and install all native components together.");
    module.attr("_native_build_id") = HOLISTICMOTION_NATIVE_BUILD_ID;
    using namespace holistic_motion::python;
    BindCoordinates(module);
    BindRobotTypes(module);
    BindKinematics(module);
    BindSevenAxisKinematics(module);
    BindRobot(module);
    BindSamplingPlanning(module);
    BindPlanningCollisionLoader(module);
    BindTrajectories(module);
    BindLogging(module);
#ifdef HOLISTICMOTION_HAS_COLLISION
    module.attr("_collision_compiled") = true;
#else
    module.attr("_collision_compiled") = false;
#endif
}
