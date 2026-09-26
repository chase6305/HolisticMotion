# Core concepts

<div class="language-switcher">English · <a href="../zh_CN/concepts.html">简体中文</a></div>

## Explicit model ownership

HolisticMotion does not bundle robot assets. Pass an absolute or application-
resolved URDF path to `Robot`, `CollisionModel`, and retargeting solvers.

The default robot chain is the supported root-to-leaf path with the most actuated
joints. Ties retain the first visited leaf, and trailing fixed transforms remain
part of the tool pose. Paths containing mimic or unsupported joints are excluded.
Explicit base-to-tip construction uses incremental name indexes within each call
for long paths in larger models, while short paths retain direct lookup. C++ edits
to public model names and parent relations are observed on the next call; duplicate
names retain first-match resolution. A cyclic parent traversal terminates with
`nullptr` (`None` in Python) rather than hanging.

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
amplify numerical noise. Projection uses the retained right singular vectors
directly, without forming a pseudoinverse. `NullSpacePlanner` requires its start
to satisfy both hardware and user limits and safely normalizes large finite
preferred directions before taking a step.
Ideal SRS models retain the direct closed-form result;
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
The high-precision pass inherits the current TCP, so its target and final check
use the same tool frame as the initial solve.
CPU batch FK prepares joint axes and the fixed terminal/TCP transform once per
call, then accumulates each pose without allocating per-row joint or pose lists.
Model and TCP changes take effect on the next call. Scalar FK uses axis-angle
rotations for oblique revolute axes, and C++ continuous joint nodes contribute
to both FK and the geometric Jacobian. Empty batches return shape `(0, 4, 4)`.
Streaming offset-arm targets can use `FEPContinuousTracker` with the same
options and diagnostics as the SRS tracker. Both trackers solve the current
branch on every frame, refresh all branches periodically, and immediately
enumerate all candidates if the current branch fails or approaches a
singularity. `candidate_refresh_interval` controls that tradeoff.

Numerical IK reuses its thin-SVD, Jacobian, and step buffers within each solve.
An already converged seed skips these buffers. The workspace is local to the
call and introduces no shared mutable solver cache. C++ tolerance, step-size,
and damping setters require finite positive values; invalid values leave the
previous setting intact. `GetIK()` clears solution and distance outputs before
validation, so a failed call does not retain distances from a previous solve.

C++ solver construction validates joint fields and coordinate layout before
allocating model-sized state. Numerical models reserve their last node for a
fixed tool transform (`UNKNOWN` is also accepted for compatibility); OPW and UR
require at least six coordinate slots, with only fixed/unknown nodes after them.
Invalid models throw `std::invalid_argument`. `NumericalKinematics::SetDOF()`
accepts only the existing model-derived dimension; changing it requires a new
solver. A numerical model with one fixed node has zero DOF: its Jacobian is
`6 x 0`, and IK returns one empty joint vector only when the target already
matches its pose within tolerance.

`SetJointNode()` rejects size changes, unsupported joint types, invalid fields,
or actuated nodes beyond the coordinate slots. Both hardware and user limit
updates must preserve a nonempty intersection. Rejected updates leave the
previous model or limits intact. C++ TCP and user frames require finite
coefficients and a unit quaternion. FK also checks arithmetic overflow from
finite joint inputs: failure clears the pose list and propagates as `ValueError`
through Python `forward()`, `forward_all()`, and `jacobian()`. Full model
validation happens at construction/update time; FK retains lightweight runtime
dimension and arithmetic checks.

`IkRtn::GetLimitsIK()` expands revolute/continuous coordinates inside inclusive
hardware/user limits. It computes the feasible turn interval directly, so work
no longer grows with the number of turns in an out-of-limit seed. Extreme seeds
use their sine/cosine phase when ordinary remainder reduction loses accuracy;
returned wraps must preserve that phase to `1e-10`. Nonperiodic coordinates are
only checked against their interval. An already in-limit configuration with no
other periodic representation skips turn enumeration and its workspace.

Expansion is limited to 65536 configurations by default, counted before any
later deduplication. The C++ overload `GetLimitsIK(nodes, max_solutions)` accepts
an explicit budget from 1 through `INT_MAX`. Exceeding the budget or an
unrepresentable turn range returns `false` with an empty result. No partial
solution set is reported. Numerical `solve_all()` inherits the default budget;
large multi-turn ranges can therefore fail even if individual joints have valid
representations. Analytic nearest-branch filtering uses its existing separate
`WrapToLimitsNear()` path.

## Scope

Null-space path generation and pose retargeting are solver-adjacent algorithms.
The optional collision-backed sampling planner provides a focused set of
joint-space RRT algorithms implemented by HolisticMotion. Controllers,
calibration, implicit model downloads, and mandatory visualization dependencies
remain outside the core library.
