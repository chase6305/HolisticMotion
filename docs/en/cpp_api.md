# C++ API reference

<div class="language-switcher">English · <a href="../zh_CN/cpp_api.html">简体中文</a></div>

The declarations below are generated from public headers by Doxygen and
Breathe.

## Collision model

```{doxygenclass} holistic_motion::robotics::collision::CollisionModel
:members:
```

```{doxygenclass} holistic_motion::robotics::collision::SphereCollisionModel
:members:
```

## Robot model

```{doxygenclass} holistic_motion::robotics::Robot
:members:
```

## Kinematics capabilities

```{doxygenstruct} holistic_motion::robotics::KinematicsCapabilities
:members:
```

All built-in solvers expose `GetCapabilities()` through `KinematicsBase`.
The default IK algorithm describes the common `GetIK()` and `GetNearestIK()`
entry points; specialized SRS/FEP solve methods remain explicit.

```{doxygenstruct} holistic_motion::robotics::IKSolveReport
:members:
```

`KinematicsBase::SolveIK()` returns an owned report and preserves the caller's
seed. Failure statuses distinguish invalid inputs from unavailable solutions.

## Named joint coordinates

```{doxygenstruct} holistic_motion::robotics::JointCoordinate
:members:
```

```{doxygenclass} holistic_motion::robotics::JointPositionMap
:members:
```

## Sampling planner

```{doxygenclass} holistic_motion::robotics::planning::JointSpace
:members:
```

```{doxygenclass} holistic_motion::robotics::planning::SamplingPlanner
:members:
```

```{doxygenclass} holistic_motion::robotics::planning::PathOptimizer
:members:
```

Additional public classes are added to this generated reference as their header
comments pass the documentation warning policy.
