# Core concepts

<div class="language-switcher">English · <a href="../zh_CN/concepts.html">简体中文</a></div>

## Explicit model ownership

HolisticMotion does not bundle robot assets. Pass an absolute or application-
resolved URDF path to `Robot`, `CollisionModel`, and retargeting solvers.

## Core and optional components

The core `holistic_motion` library contains robot models, kinematics,
manifolds, null-space path generation, and trajectories. Collision queries live
in the optional `holistic_motion_collision` companion library so ordinary
kinematics users do not acquire Pinocchio and Coal unnecessarily.

## Coordinates

- Target poses are finite 4×4 homogeneous transforms.
- Retargeting targets are expressed in the URDF world frame.
- Joint configurations follow Pinocchio model order in the generic retargeting
  result.
- Applications should perform any controller-specific joint reordering at
  their integration boundary.

## Scope

Null-space path generation and pose retargeting are solver-adjacent algorithms.
General motion planning, controllers, calibration, implicit model downloads,
and mandatory visualization dependencies remain outside the core library.
