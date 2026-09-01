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

## Initialization and runtime phases

Keep model construction and configuration outside real-time loops. `Robot`,
native collision models, continuous IK trackers, and TOPPRA trajectories
resolve topology and allocate persistent state in their constructors. For
retargeting, call `prepare(mode)` after construction or during a non-real-time
mode transition; `solve()` and `step()` then reuse mode indices and numerical
workspaces. Sampling planners and path optimizers retain joint topology in the
planner object, while each `plan()` or `optimize()` call owns only request-local
search state. Geometry sphere fitting is an offline initialization operation;
the generated sphere model is the runtime artifact.

Stateful solvers and collision models reuse mutable workspaces and are not
safe for concurrent calls on the same instance. Use one prepared instance per
runtime worker when parallel queries are required.

## Seven-axis kinematics

`SRSKinematics` is selected for compatible seven-revolute spherical
shoulder/wrist chains. Its null-space projection rejects non-finite inputs and
uses a relative singular-value cutoff so near-singular configurations do not
amplify numerical noise. Ideal SRS models retain the direct closed-form result;
small URDF link offsets use that result as a branch-preserving seed for a
strict local correction. Equivalent revolute angles are selected directly
from the declared joint-limit interval rather than from a fixed wrap count.

For streaming targets, use the stateful Python tracker instead of selecting
each frame independently:

```python
from holistic_motion.kinematics import SRSContinuousTracker

tracker = SRSContinuousTracker(solver, initial_joints=q0)
result = tracker.solve(target_pose, dt=0.01)
q_command = result.joints
```

The tracker expands periodic angles near the predicted state, applies branch
hysteresis, filters candidates by pose and dynamic limits, and reports
singularity, branch-change, velocity, acceleration, and residual diagnostics.
The native `SRSKinematics` remains stateless and can still be used directly.
Use `solve_detailed(target, seed, method)` when the caller needs a non-throwing
status plus each solution's configuration, minimum Jacobian singular value,
per-joint limit margins, near-singularity flag, and limit-hit flag. The existing
`solve()` API remains the compact exception-based interface.
TCP and user-frame changes are explicit runtime configuration. `forward()` and
`solve()` keep using the robot base frame; `forward_user()`, `inverse_user()`,
`solve_user()`, and `solve_detailed_user()` apply the configured user transform
using the SDK-compatible `user_T_base * base_T_tcp` convention. Use the
`tcp`/`user_frame` properties and `clear_tcp()`/`clear_user_frame()` to inspect
or reset that state.

`FEPKinematics` supports offset seven-axis chains and batched FK. CPU is used
for small batches; `AUTO` selects CUDA only when a runtime device is available
and the batch is large enough to amortize transfers. Returned FEP IK solutions
are independently checked to 10 µm and 10 µrad after optional high-precision
refinement. Explicit `CUDA` requests fail instead of silently falling back.
Streaming offset-arm targets can use `FEPContinuousTracker` with the same
options and diagnostics as the SRS tracker. Both trackers solve the current
branch on every frame, refresh all branches periodically, and immediately
enumerate all candidates if the current branch fails or approaches a
singularity. `candidate_refresh_interval` controls that tradeoff.

## Scope

Null-space path generation and pose retargeting are solver-adjacent algorithms.
The optional collision-backed sampling planner provides a focused set of
joint-space RRT algorithms implemented by HolisticMotion. Controllers,
calibration, implicit model downloads, and mandatory visualization dependencies
remain outside the core library.
