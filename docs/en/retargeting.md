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

Frame mappings and joint groups are validated lazily for the selected mode.
An arm-only application therefore does not need dummy foot or pelvis mappings.
For leg modes, add `left_leg` and `right_leg` joint groups plus `left_foot` and
`right_foot` frame mappings. `whole_body` retains its original hand/head API;
use `full_body` when feet and pelvis are explicit targets.

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
task cost; zero-cost axes remain visible in the reported raw residuals but do
not prevent success.

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

`step()` jointly bounds displacement using position and velocity limits. When
`acceleration_limits` are configured, it also limits the change from the
previous commanded velocity. The Viser demo exposes this behavior as the
`Single QP step` strategy; `Iterative solve` remains available for offline or
event-driven convergence.

```{warning}
Collision callbacks are optional. Without them, retargeting does not perform
collision checking. Even with a soft collision cost, use the optional
C++/Python `CollisionModel` component before accepting a command when your
application requires hard collision rejection.
```
