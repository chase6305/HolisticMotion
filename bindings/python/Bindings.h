#pragma once

#include <pybind11/pybind11.h>

namespace holistic_motion::python {

void BindCollision(pybind11::module_& module);
void BindCoordinates(pybind11::module_ &module);
void BindLogging(pybind11::module_& module);
void BindSamplingPlanning(pybind11::module_& module);
void BindPlanningCollision(pybind11::module_ &module);
void BindPlanningCollisionLoader(pybind11::module_ &module);
void BindRobotTypes(pybind11::module_ &module);
void BindRobot(pybind11::module_ &module);
void BindKinematics(pybind11::module_ &module);
void BindSevenAxisKinematics(pybind11::module_ &module);
void BindTrajectories(pybind11::module_ &module);

}  // namespace holistic_motion::python
