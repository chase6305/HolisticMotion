# Pose retargeting

<div class="language-switcher">English · <a href="../zh_CN/retargeting.html">简体中文</a></div>

Install Pinocchio's Python runtime and import the toolkit:

```bash
python -m pip install '.[retargeting]'
```

```python
from holistic_motion.kit.retargeting import (
    CenterOfMassTask,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
    SupportPolygonTask,
    ZmpTask,
)

solver = PinkRetargetingSolver(
    "/path/to/robot.urdf",
    frames={
        "left_hand": "left_ee",
        "right_hand": "right_ee",
        "head": "head_ee",
    },
    joint_groups={
        "left_arm": ["left_j1", "left_j2", "left_j3"],
        "right_arm": ["right_j1", "right_j2", "right_j3"],
    },
    frame_tasks={
        "left_hand": FrameTask(position_cost=1.0, orientation_cost=0.2),
        "right_hand": FrameTask(position_cost=1.0, orientation_cost=0.2),
    },
    posture_task=PostureTask(cost=1e-3),
    center_of_mass_task=CenterOfMassTask(cost=[1.0, 1.0, 0.2]),
    support_polygon_task=SupportPolygonTask(
        [[-0.12, -0.08], [0.12, -0.08], [0.12, 0.08], [-0.12, 0.08]],
        cost=10.0,
        margin=0.01,
    ),
    zmp_task=ZmpTask(cost=[1.0, 1.0]),
)
solver.set_center_of_mass_target([0.0, 0.0, 0.85])
solver.set_zmp_target([0.0, 0.0])
solver.set_center_of_mass_acceleration([0.0, 0.0, 0.0])
```

## Optional dex-retargeting hand backend

`DexHandRetargetingSolver` integrates the actual dex-retargeting 0.5.0
`VectorOptimizer` as a separate hand solver. Install the extra on Python 3.9–3.12
(the upstream package excludes Python 3.13 and later):

```bash
python -m pip install '.[hand-retargeting]'
```

To use a local dex-retargeting checkout, install it first with
`python -m pip install /path/to/dex-retargeting`; the adapter targets version
0.5.0. The extra adds PyTorch explicitly because upstream imports it without
declaring it. CPU execution is sufficient; CUDA is not required. Ordinary
toolkit imports do not load dex-retargeting, NLopt, or PyTorch.

```python
import numpy as np
from holistic_motion.kit.retargeting import DexHandRetargetingSolver

hand = DexHandRetargetingSolver(
    "/path/to/hand.urdf",
    joint_names=["thumb_joint", "index_joint"],  # optimized independent joints
    origin_links=["palm", "palm"],
    task_links=["thumb_tip", "index_tip"],
    keypoint_pairs=[[0, 1], [0, 2]],  # rows correspond to the robot link pairs
    scaling=1.0,
    temporal_weight=4e-3,
    vector_tolerance=0.02,
    max_evaluations=100,
)
points = np.load("/path/to/keypoints.npy", allow_pickle=False)  # (N, 3)
result = hand.solve(points)
if result.success:
    hand_positions = dict(result.joint_positions)
else:
    print(result.termination_reason, result.message)
```

Replace the example names and index pairs with your hand's correspondences.
Keypoints must use the model's base axes and length unit (normally metres);
align human wrist orientation before solving. The objective matches endpoint
differences, so global translation cancels. `scaling` multiplies human vectors.
The regularizer penalizes displacement from the current seed; its scalar and
gradient are kept consistent in the adapter.
`objective_tolerance` defaults to `1e-9` (absolute objective change), reducing
premature stopping compared with upstream's `1e-6`. It is separate from the
maximum geometric error checked by `vector_tolerance`; `max_evaluations` still
bounds optimizer evaluations.

Between solves, `scaling`, `temporal_weight`, `huber_delta`, `vector_tolerance`,
and `objective_tolerance` may be updated on the solver. The next `solve()`
validates all five settings before synchronizing the optimizer's loss, gradient,
and stopping tolerance. Invalid settings raise before changing native settings
or warm-start history; correct the attributes before retrying. Keep settings
fixed while a solve is running.

`solve()` can run inside `torch.no_grad()` or `torch.inference_mode()`, for
example after keypoint inference. The solver can also be constructed inside
these contexts: cached index tensors are created as ordinary tensors so they
remain usable in later backward passes. The adapter enables autograd locally when
NLopt requests derivatives and skips graph construction for scalar-only
evaluations. The caller's gradient and inference modes are restored on return
or exception. This internal gradient is for hand optimization; the NumPy result
does not provide end-to-end gradients back to the keypoint model.

Pass a dedicated scalar-joint hand URDF, with bounded revolute/prismatic joints.
Floating, spherical, and continuous joints are unsupported. Direct mimic joints
are derived from their optimized source; do not include followers in
`joint_names`. Their limits constrain the source without upstream's extra
limit padding. Nested mimics and mimics driven by unoptimized joints are rejected.
Source bounds are intersected using the actual float64 multiply-then-add map,
including locked followers and very small nonzero multipliers. This avoids
rejecting feasible poses because inverse-bound arithmetic rounds or overflows.
An interval with no representable feasible source position is rejected.
An explicit `seed` must map every independent hand joint to its position;
unoptimized joints retain that value. Omitted seeds use the previous successful
configuration, initially a limit-projected neutral pose. `reset()` restores
neutral; `reset(configuration)` installs a validated named configuration.

`HandRetargetingResult` reports named full hand positions (including mimic and
unoptimized joints), maximum vector residual, objective, time, evaluation count,
native status, and termination reason. An input seed already within the vector
tolerance is held directly, with `converged`, zero evaluations, and native status
zero (no optimizer invocation). If all optimized joint bounds coincide, including
mimic-induced bounds, an unmet target returns `no_feasible_motion` with the same
diagnostics. Both paths skip Jacobians and preserve the supplied pose;
a successful hold becomes the next warm start. Subsequent moving frames resume
normal optimization. These diagnostics do not reuse a preceding native solve's
status or evaluation count.

When optimization is needed, success requires optimizer convergence and residual
tolerance. Budget exhaustion, invalid output, and native solver
failure are explicit failures: return positions and residual describe the input
seed, and the previous warm start is retained. Callback errors propagate.
Each native optimization releases its frame-specific callback on completion,
failure, or interruption, so the native optimizer does not keep the hand solver,
old targets, or exception tracebacks alive through that callback. Later moving
frames register a fresh objective; optimizer status and evaluation counts remain
available in the result. This is object-lifetime handling, not a guarantee that
Python or native allocator memory is immediately returned to the operating system.

Run `python examples/python/retargeting/dex_hand.py --help` for the CLI using
explicit URDF, vector correspondences, a keypoint `.npy`, and an optional seed
JSON object. Failed solves exit with status 1.

For torso/arm/hand retargeting, solve torso and arm poses with the existing
solvers and fingers with one hand solver per side. Merge accepted finger
positions into the full configuration by joint name, excluding any fixed wrist
entries from the hand result. Hand URDF names must match the full robot or use an
explicit caller-provided mapping. This backend supplies position retargeting;
apply trajectory limits and full-robot collision validation after composition.
It does not implement WBC, velocity/acceleration constraints, or collision checks.

## Modes

| Mode | Required targets | Active velocities |
|---|---|---|
| `left_arm` | `left_hand` | Left-arm group |
| `right_arm` | `right_hand` | Right-arm group |
| `dual_arm` | Both hands | Both arm groups |
| `left_leg` | `left_foot` | Left-leg group |
| `right_leg` | `right_foot` | Right-leg group |
| `dual_leg` | Both feet | Both leg groups |
| `whole_body` | Both hands and head | Entire model |
| `full_body` | Both hands, both feet, head, and pelvis | Entire model |
| `torso` | `torso` | Torso group |
| `torso_left_arm` | `left_hand` | Torso and left-arm groups |
| `torso_right_arm` | `right_hand` | Torso and right-arm groups |
| `torso_dual_arm` | Both hands | Torso and both arm groups |

Frame mappings and joint groups are validated lazily for the selected mode.
An arm-only application therefore does not need dummy foot or pelvis mappings.
For leg modes, add `left_leg` and `right_leg` joint groups plus `left_foot` and
`right_foot` frame mappings. `whole_body` retains its original hand/head API;
use `full_body` when feet and pelvis are explicit targets.

For the Pink and cuRobo-style solvers, `FrameTask` separates Cartesian position
error from the rotation-vector error. Target poses are expressed in the world
frame; per-axis costs apply along the current tracked frame's axes, matching
its local Jacobian. Setting `orientation_cost=0` makes position tracking
independent of the target rotation. Zero-cost axes do not block convergence,
but raw telemetry still includes them. `position_residual` and each target's
position residual report Euclidean distance; orientation residuals report
rotation angle in radians. The position term no longer uses the translational
part of the SE(3) logarithm, so objectives and candidate rankings can change
when position and orientation errors are both nonzero.

For unequal per-axis frame costs, the task Jacobian accounts for the rotation
of the current frame's position-error axes and the derivative of its rotation
vector. This keeps the update gradient consistent with the weighted objective.
Isotropic position and orientation blocks retain their existing velocity-task
update. A partial local-axis target can be satisfied by reorienting those axes
without reducing the full Cartesian distance; raw position telemetry continues
to report that distance.

Task `gain` scales the desired QP correction. The line-search and seed-ranking
`objective` applies each gain once: a tracking or posture term contributes
`0.5 * gain * ||weighted_error||²`, with collision cost added separately.
Different task gains therefore also affect the compromise between conflicting
goals. The QP Hessian and gain-scaled LM damping retain their existing behavior.
`residual` reports the norm of cost-weighted tracking errors without gains,
posture, or collision terms; task convergence still uses physical tolerances.
Compared with earlier gain handling, non-unit gains can change reported
objectives, residuals, accepted offline updates, and seed rankings. Defaults
with all gains equal to one are unchanged.

Setting every cost of a CoM, ZMP-tracking, or support-polygon task to zero
disables both its solve contribution and diagnostics; its tracking target is
no longer required. If no enabled task depends on CoM, neither CoM nor its
Jacobian is evaluated, so a purely kinematic model needs no mass information.
Disabled CoM/ZMP residuals are `NaN` and disabled support violation is `0`;
these defaults are not measurements or stability guarantees. Partially weighted
tasks still report full residual telemetry. An enabled ZMP support constraint
continues to require `ZmpTask` for its plane and gravity model even if point
tracking has zero cost; a disabled ZMP support task has no such dependency.

`CenterOfMassTask` adds a world-frame CoM residual and Pinocchio's analytic CoM
Jacobian to the same bounded QP. Configure `center_of_mass_tolerance` separately
from frame tolerances. The final value is exposed as
`result.center_of_mass_residual`. This provides weighted CoM tracking; a hard
support-polygon condition should not be inferred from a small CoM residual
alone.

`SupportPolygonTask` adds a differentiable penalty for CoM positions outside a
strictly convex XY support polygon. Clockwise vertices are normalized
automatically; duplicate, collinear, concave, or non-finite polygons are
rejected. `margin` shrinks the usable region, and
`result.support_polygon_violation` reports the largest edge violation. CoM and
support tasks share one center/Jacobian evaluation per state. This remains a
soft QP objective.
Convexity checks use unit edge directions and normalized relative coordinates,
so their tolerance is independent of length units. Geometry whose spans, edge
lengths or offsets cannot be represented as finite floats is rejected. Support
distances are projected from vertex-relative coordinates to reduce cancellation
near boundaries far from the world origin; this cannot recover precision already
lost in the supplied coordinates.
Set `reference="zmp"` to constrain the kinematic ZMP instead of the CoM. This
requires a `ZmpTask`; use zero ZMP tracking cost when only the polygon
inequality is desired, in which case no point ZMP target is required.

`ZmpTask` tracks the kinematic approximation
`zmp_xy = com_xy - (com_z - plane_height) * com_acceleration_xy / gravity`.
With the default zero acceleration it reduces to the quasi-static CoM projection. Set the XY
target and, when available, the world-frame CoM acceleration with the setters
shown above. `result.zmp_residual` reports the final error. It is a weighted
solver objective, not a contact-force, friction-cone, or full rigid-body
dynamics constraint. Convergence tolerances apply only to axes with a positive
task cost; while tracking is enabled, zero-cost axes remain visible in the
reported raw residuals but do not prevent success.

The supplied CoM acceleration is treated as constant during each nonlinear
solve. The analytic ZMP Jacobian therefore differentiates the CoM position and
height, not an acceleration model coupled to the configuration. Set
`plane_height` when the support plane is not at world `z=0`.

```python
solver.prepare("dual_arm")
result = solver.solve({
    "left_hand": left_pose,
    "right_hand": right_pose,
})

if result.success:
    send_joint_command(result.configuration)
```

Call `prepare(mode)` during application initialization or a non-real-time mode
transition. It validates frame and joint-group mappings and caches active
indices, velocity limits, and Pink numerical workspaces. `set_mode(mode)` keeps
the legacy lazy behavior; it is useful while assembling configuration, but its
first subsequent solve may perform mode preparation.

Targets, task costs, mode specifications, and result payloads own immutable
snapshots of caller-provided arrays and sequences. Copy
`result.configuration` before adapting it in application code. This prevents
UI or controller code from modifying solver history through a returned result.
Target matrices must be proper SE(3) transforms with a homogeneous last row
and a right-handed orthonormal rotation.

The previous result is reused as a warm start. Pass `seed=` to override it or
call `solver.reset()` to return to Pinocchio's neutral configuration.
The base `PinocchioRetargetingSolver` validates the result before updating that
history. Rejected non-finite diagnostics or aborted result construction retain
the previous warm start, even when an explicit seed was supplied. A normally
returned result that exhausts the iteration budget still becomes the next seed.
Residual norms use overflow-safe accumulation, so a large finite pose error can
still be returned as a failed solve diagnostic. If that residual is representable
but its squared objective is not, `result.objective` is `NaN`; the residual and
configuration remain valid, and the normally returned configuration becomes the
next warm start.

Configurations are normalized on the model manifold before coordinate limits
are applied. This preserves the orientation of scaled continuous-joint
sine/cosine pairs and floating-joint quaternions while still clamping scalar
joint positions. All-zero orientation coordinates, and values that cannot be
normalized numerically, raise `ValueError`. Invalid seeds, reset configurations,
or posture targets leave the previous configuration, velocity history, and
posture preference unchanged; input arrays are not modified.

## Torso-arm coordination and torso-first stages

The `torso_left_arm`, `torso_right_arm`, and `torso_dual_arm` modes activate
the caller's `torso` joint group together with the selected arm groups. They
require the corresponding hand targets and jointly solve shared torso variables
only once. The torso group may contain one or several joints. Unlisted joints
stay fixed; head and leg targets are not required. A `torso` frame mapping is
needed only for the separate `torso` mode, which aligns that frame using the
torso joint group alone.

```python
from holistic_motion.kit.retargeting import (
    FrameTask, PinkRetargetingSolver, PostureTask, solve_torso_first,
)

solver = PinkRetargetingSolver(
    "/path/to/robot.urdf",
    frames={"torso": "torso_link", "left_hand": "left_ee", "right_hand": "right_ee"},
    joint_groups={
        "torso": ["waist_yaw"],
        "left_arm": left_arm_joint_names,
        "right_arm": right_arm_joint_names,
    },
    frame_tasks={"torso": FrameTask(position_cost=0.0)},
    posture_task=PostureTask(cost=1e-3, joint_costs={"waist_yaw": 0.01}),
    joint_motion_costs={"waist_yaw": 0.1},
)
solver.set_posture_target(preferred_configuration)
solver.prepare("torso_dual_arm")
result = solver.solve(
    {"left_hand": left_pose, "right_hand": right_pose}, seed=current_configuration,
)
```

Replace joint/link names with those from the explicit URDF. Posture targets use
the full model configuration; frame targets are world-frame SE(3) transforms.
The illustrative weights need task-specific tuning: a strong torso posture
preference can compete with hand tracking.

`PostureTask.joint_costs` overrides the default `cost` by joint name, applying
to all tangent coordinates of each named joint, even when the default is zero.
`joint_motion_costs` adds `0.5 * sum((cost * delta_q) ** 2)` to each local QP,
with zero defaults for unspecified joints. Both mappings are copied and resolved
using `nv` joint indices, including models with `nq != nv`. Motion regularization
affects local updates, not the reported `objective` or multi-seed ranking. It
neither bounds total travel nor replaces velocity/acceleration limits. Posture
preferences are soft objectives and do not independently block convergence: if
tracked tasks are already within tolerance, the solver may return immediately.

For an explicit torso-first sequence:

```python
stages = solve_torso_first(
    solver,
    torso_target=desired_torso_pose,
    hand_targets={"left_hand": left_pose, "right_hand": right_pose},
    seed=current_configuration,
    arm_mode="dual_arm",  # Also supports left_arm and right_arm.
)
if stages.success:
    aligned_configuration = stages.torso.configuration
    manipulation_configuration = stages.arms.configuration
```

The first stage holds arm joints fixed, so hands move with the torso. The second
holds torso joints fixed while solving the selected arms. The two modes must
have disjoint active joint sets. If alignment fails, `stages.arms` is `None`;
arm failure preserves diagnostics from both stages. The original mode is always
restored. Normal returns retain the last attempted configuration as the warm
start; exceptions restore configuration, velocity history, and multi-seed index
and candidate-count diagnostics. This includes `KeyboardInterrupt` and
`SystemExit`: the original interruption propagates after rollback. The helper also
accepts the cuRobo-style solver, whose neutral and random seeds now change only
active joints.

`seed` must explicitly provide a valid configuration: `None` is rejected instead
of silently selecting a neutral pose. `TorsoFirstResult` requires a `torso`
first stage, an arm-only second stage, and matching configuration dimensions.
Ordinary non-convergence retains the last attempt's diagnostics; exceptions
restore the pre-call state.

These are offline IK waypoints. The caller must plan and validate both paths and
confirm physical torso arrival before executing the arm stage. This strategy
does not enforce world-frame hand holding during alignment, rigid dual-grasp
constraints, contact forces, or dynamic balance. For simultaneous hand
compensation, use a coordinated torso-arm mode with ongoing hand targets and
separately validate path constraints.

Run the same sequence against an explicit URDF with the classic command-line
example. Repeat each joint option to define multi-joint groups:

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/retargeting/classic_torso_first.py \
  --urdf /absolute/path/to/robot.urdf \
  --torso-frame torso_link --torso-joint waist_yaw \
  --left-frame left_tool --left-joint left_shoulder --left-joint left_elbow \
  --right-frame right_tool --right-joint right_shoulder --right-joint right_elbow \
  --joint-delta waist_yaw=-0.2 --joint-delta left_elbow=0.4
```

The default `dual_arm` mode can be replaced by `left_arm` or `right_arm`. The
demo derives reachable poses from neutral plus scalar joint deltas and rejects
deltas outside the URDF limits. Group defaults come from `--torso-delta` and
`--arm-delta`; repeat `--joint-delta JOINT=DELTA` to override individual
configured joints. Its output remains offline IK waypoints rather
than an executable, collision-checked path. It prints convergence diagnostics,
solve time, named active-joint values, and the complete configuration for each
stage so the result can be inspected before planning.

## Solver choices

- `PinocchioRetargetingSolver` provides compact damped least-squares IK.
- `PinkRetargetingSolver` assembles a Pink-style weighted task objective and
  adds anisotropic frame costs, posture regularization, LM damping, and
  velocity-limited integration.
- `CuroboRetargetingSolver` adds deterministic multi-seed refinement,
  manifold-aware seed perturbations, best-result selection, and optional early
  exit on the first converged seed.

The cuRobo-style solver accepts the same frames, joint groups, tasks, modes,
and targets as `PinkRetargetingSolver`:

```python
from holistic_motion.kit.retargeting import CuroboRetargetingSolver

solver = CuroboRetargetingSolver(
    "/path/to/robot.urdf",
    frames=frames,
    joint_groups=joint_groups,
    num_seeds=8,
    seed_spread=0.35,
    sampler_seed=451,
)
result = solver.solve(targets, seed=current_configuration)
```

`last_seed_index` and `last_num_seeds_evaluated` expose selection statistics.
Set `stop_on_success=True` for latency-sensitive loops; leave it disabled when
you want all distinct seeds ranked for solution quality. `num_seeds` is an
upper bound: duplicate seeds created by zero spread or limit projection are
discarded before refinement. Converged seeds are preferred first, then seeds
are ranked by the same weighted `objective` used during optimization rather
than by an unweighted collision distance.
Candidates are generated on demand in primary, neutral, then deterministic
random order. Early success skips generation as well as refinement of unused
alternatives. Exhaustive search retains the same order and ranking. If a later
candidate fails to generate or refine, configuration, velocity history, and
selection statistics are restored to their values before the solve.
This rollback also applies to `KeyboardInterrupt` and `SystemExit` in both
Pink and cuRobo-style `solve()`/`step()` calls. An interrupted cycle restores
the previous velocity history before propagating the interruption; a subsequent
call therefore does not inherit an unreturned update.
`step()` always uses only its physical primary seed: random alternative
configurations cannot satisfy a one-cycle acceleration bound relative to the
robot's current state. Multi-seed refinement remains available through
`solve()` when acceleration enforcement is disabled.

## Collision cost and gradient

`PinkRetargetingSolver` and `CuroboRetargetingSolver` accept optional
`collision_cost`, `collision_gradient`, and combined `collision_cost_gradient`
callbacks. The non-negative cost is
included in line-search acceptance and convergence, while its tangent-space
gradient directly changes the IK step. If the gradient callback is omitted, a
joint-limit-aware finite difference is used over active velocity coordinates.

Numerical differences skip coordinates with zero velocity limits or equal scalar
position bounds, even when other joints in the mode remain movable. Those fixed
components are zero in the internal numerical gradient; they require no sample
integration or cost query. Movable components retain the same sampling and
caching behavior. Analytic and combined callbacks still return a full `nv`
gradient. The `limit_hits` diagnostic can differ because it compares the bounded
update with the unconstrained one, whose fixed-coordinate gradient has changed.

When limit projection gives unequal sample distances, a three-point formula
uses the current cost and both samples to estimate the derivative at the current
configuration, rather than at the samples' midpoint. Equal distances retain the
central difference; exact bounds retain a one-sided difference and reuse the
current cost. This correction adds no callback evaluations.

Sample usability is scale-relative, without a fixed absolute joint-displacement
cutoff. If one positive sample distance is less than `sqrt(machine epsilon)`
times the other, the longer one-sided difference is used to avoid amplifying
cost roundoff. Otherwise both representably nonzero offsets are retained,
including small joint ranges. A sample that rounds back to the current
configuration reuses its cost.

The existing sphere model can provide a quadratic clearance penalty. Its
analytic gradient is expressed in the `nv`-dimensional tangent space, including
when a manifold model has `nq != nv`:

```python
clearance = 0.05

def collision_cost(q):
    distance = sphere_model.minimum_distance(q).distance
    return max(0.0, clearance - distance) ** 2

def collision_cost_gradient(q):
    result = sphere_model.minimum_distance_with_gradient(q)
    deficit = max(0.0, clearance - result.distance_result.distance)
    return deficit**2, -2.0 * deficit * result.gradient

solver = CuroboRetargetingSolver(
    urdf_path,
    frames=frames,
    joint_groups=joint_groups,
    collision_cost=collision_cost,
    collision_cost_gradient=collision_cost_gradient,
    collision_cost_weight=10.0,
    collision_tolerance=1e-8,
)
```

The scalar callback keeps objective-only line-search candidates inexpensive;
the combined callback obtains distance and gradient from one query when the QP
is built. Use `collision_gradient` instead when cost and gradient naturally
come from separate systems.

`RetargetingResult` reports the final weighted `objective`, `collision_cost`,
`collision_evaluations`, and `collision_gradient_evaluations`. This is a soft
differentiable penalty, not a hard continuous-motion collision guarantee;
validate the final command and the motion between commands when strict safety
is required.
Collision costs and gradients are cached by exact configuration across all
seeds in one `solve()` call, then discarded so a dynamic scene is observed by
the next call.

The Pink- and cuRobo-style implementations are maintained inside
HolisticMotion. They do not import upstream `pink`, `qpsolvers`, cuRobo,
Torch, or Warp at runtime. See
`examples/python/retargeting/pink_dual_arm.py` for a complete command-line
example.

When collision support is built with Conan while Python Pinocchio comes from
the `pin` wheel, keep their native libraries out of the same process. Use the
dedicated launcher, which loads the pure-Python toolkit without the compiled
HolisticMotion extension:

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/retargeting/pink_dual_arm.py --help
```

For interactive URDF retargeting with left-hand, right-hand, and head gizmos:

```bash
python3 examples/python/visualization/pink_robot_viser.py \
  --urdf /path/to/robot.urdf
```

The Viser panel exposes left-arm, right-arm, dual-arm, and whole-body modes,
continuous solving, target reset, residual, iteration count, and solve time.
The solver uses separate position and orientation tolerances, adaptive damping,
backtracking step acceptance, velocity clipping, and stagnation detection. The
panel reports per-target errors and the termination reason; unreachable targets
are solved once per gizmo event rather than retried every render frame.

For a real-time control loop, use one bounded QP step per cycle:

```python
result = solver.step(targets, seed=current_configuration)
```

`solve(..., enforce_acceleration=True)` has the same single-cycle semantics:
it performs at most one QP update, even if `max_iterations` or the constructor's
iteration budget is larger. The budget must still be a positive integer.
The returned displacement and stored commanded velocity therefore describe one
`integration_dt`, including when no explicit acceleration limits are configured.
Call `step()` again with the next measured configuration to advance another cycle.
For offline iterative convergence, use the default `enforce_acceleration=False`;
an offline result is not a single-cycle command.

When continuous or floating joints make `nq != nv`, scalar revolute and prismatic
position limits are still mapped from their configuration indices to the QP's
velocity variables. Coordinated IK can assign the remaining motion to the torso
when an arm approaches its limit. This mapping does not treat sine/cosine or
quaternion coordinates as scalar joint limits. Joint groups and named acceleration
limits reject the `universe` placeholder joint.

`step()` jointly bounds displacement using position and velocity limits. When
`acceleration_limits` are configured, it also limits the change from the
previous commanded velocity. The Viser demo exposes this behavior as the
`Single QP step` strategy; `Iterative solve` remains available for offline or
event-driven convergence.

For the Pink and cuRobo-style solvers, a zero velocity limit fixes that coordinate
at the supplied (normalized and limit-projected) seed. It is not treated as an
unbounded velocity. Multi-seed search excludes zero-speed coordinates from both
neutral and random candidates. Negative or NaN active velocity limits are
rejected during mode preparation or solving; positive infinity remains unbounded.
If preparation fails or is interrupted, the original mode is restored and a repeated attempt
revalidates the limits instead of using a partial cache.

If the joint bounds permit only zero displacement, the solver evaluates the
current residuals and scalar collision cost once, without Jacobians, collision
gradients, or QP updates. An unmet target returns `no_feasible_motion`; a mode
with no active DOFs retains `no_active_dofs`. Already satisfied tasks return
`converged`. All three cases retain the current configuration and report zero
accepted steps. A hold from `step()` clears commanded velocity history;
offline `solve()` leaves that history unchanged. A nonzero forced braking
displacement still uses the normal bounded update.

The box QP uses a feasible active-set update: it moves toward the reduced
minimizer only up to the first blocking joint bound, then resolves the remaining
coordinates. This avoids cycling caused by independently clipping coupled
coordinates. Equal lower and upper bounds stay fixed throughout the solve.

Offline `solve()` accepts `step_size > 1`, but caps the initial line-search
scale at the first joint displacement bound. One shared scale preserves the
QP search direction; backtracking halves that feasible scale. Amplification
remains available when the entire update fits within the bounds. Thus each
local iteration respects its position and velocity displacement box even
with a large configured step size. This does not make an offline multi-iteration
result a single-cycle command; `step()` retains its unscaled bounded update.

Even when the input pose already satisfies task tolerances, `step()` checks
whether zero displacement satisfies the active joints' velocity and acceleration
bounds. If stopping within one cycle is infeasible, it still solves a bounded
QP update. A feasible hold returns the same configuration and clears commanded
velocity history, so the next start does not reuse an old velocity.
`result.success` still describes task convergence at the returned pose, not
whether the robot has stopped. Mode switches freeze inactive joints; the caller
must arrange their deceleration transition. Offline `solve()` does not maintain
per-cycle velocity history; use `reset(configuration)` to clear that history
before starting a new sequence of steps.

```{warning}
Collision callbacks are optional. Without them, retargeting does not perform
collision checking. Even with a soft collision cost, use the optional
C++/Python `CollisionModel` component before accepting a command when your
application requires hard collision rejection.
```
