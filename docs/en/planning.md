# Sampling-based planning

<div class="language-switcher">English · <a href="../zh_CN/planning.html">简体中文</a></div>

HolisticMotion contains its own compact C++17 joint-space planning
implementation. It does not link, vendor, or import OMPL. The initial planner
set is deliberately limited to `RRT_CONNECT`, `RRT_STAR`, and
`INFORMED_RRT_STAR`.

## Responsibilities

The sampling planner produces a collision-free geometric path. Shortcutting
reduces unnecessary waypoints, and TOPPRA should subsequently apply velocity
and acceleration limits. Controllers remain outside this library.

`RRT_CONNECT` is the default for single-query robot planning. `RRT_STAR` and
`INFORMED_RRT_STAR` use the complete time budget to improve path cost, so their
reported planning time normally approaches `timeout_seconds` even after an
initial solution is available.

Before growing a tree, every algorithm validates the complete start-to-goal
edge. If it is valid, planning immediately returns the two-point shortest path;
the optimizing variants consume their time budget only when an obstacle blocks
that direct solution. HolisticMotion currently reports exact solutions only:
timeout and failure results do not contain an ambiguous partial path.

## Python example

```python
collision = hm.CollisionModel(urdf_path, [str(urdf_path.parent)])
joint_names = left_arm_joints + right_arm_joints
context = collision.configuration_from_joint_positions(start_positions)

planner = hm.SamplingPlanner.from_collision_joints(
    collision, joint_names, context, security_margin=0.005
)
options = hm.PlanningOptions()
options.algorithm = hm.SamplingAlgorithm.RRT_CONNECT
options.timeout_seconds = 2.0
options.extension_range = 0.2
options.edge_resolution = 0.03
options.random_seed = 42

result = planner.plan(start, goal, options)
if not result.success:
    raise RuntimeError(result.message)
```

`from_collision_joints` plans only the named scalar joints while restoring all
inactive and manifold joints from `context` for every Coal query. This is the
preferred dual-arm API. Use both arms in one active space so inter-arm
collisions are evaluated throughout every edge.

The generic constructor accepts a callable validator:

```python
planner = hm.SamplingPlanner(lower, upper, lambda q: is_valid(q))
```

This is useful for tests and non-Coal environments. The native collision
factory avoids crossing the Python boundary for each sampled configuration and
is substantially better for robot planning.

The optional `security_margin` rejects configurations whose collision meshes
are closer than the requested distance, not only configurations that already
penetrate. The Marvin viewer defaults to the complete inter-arm, per-arm self,
and arm-to-body policy and verifies every displayed path sample.

## Resolution and safety

The planner validates each edge at intervals no larger than
`edge_resolution` in any active joint. Smaller values improve collision
coverage but increase Coal queries. Endpoints and every shortcut are checked.
The final path is geometric and must still be validated by the application if
the environment changes after planning.

## Examples

Compare the algorithms without robot assets:

```bash
python3 examples/python/planning/rrt_variants.py
```

Plan and animate a collision-checked Marvin dual-arm path:

```bash
./scripts/run.sh python3 \
  examples/python/visualization/rrt_robot_viser.py \
  --urdf /absolute/path/to/robot_with_ee.urdf
```

Add `--validate-only` for a headless planning check. Robot assets remain
caller-provided and are not part of tests or the installed package.

The viewer draws the planned left/right end-effector paths in orange and
magenta. Green markers are starts, colored markers are goals, and cyan markers
track playback. Thin red curves show the colliding straight-line alternative;
both path layers can be toggled from the Viser panel.

The Marvin profile also defines mirrored boxes in the left and right arm
workspaces. They are added to `CollisionModel`, participate in every Coal state
and edge query, and are rendered as translucent boxes. Use `--no-obstacles` to compare
against the self/inter-arm/body-only case. Applications can manage boxes with
`add_box_obstacle(name, size, pose)` and `remove_obstacle(name)`.

The demo defaults to `--planning-space symmetric`: RRT searches a mirrored
7-DoF arm subspace and expands every checked state to the complete 14-DoF
robot. Thus inter-arm, arm-body, gripper, and environment collision checks
remain active while both paths stay balanced. Use `--planning-space coupled`
for independent or asymmetric arm motion. Planning uses 1 mm of additional
clearance to protect the requested 5 mm margin from edge discretization, and
the displayed 180-sample path is independently revalidated.

## Current limits

- Fixed-base scalar active joints are supported by the collision adapter.
- Nearest-neighbor lookup is currently linear and optimized for typical
  manipulator search-tree sizes.
- Collision validation is single-threaded.
- Dynamic obstacles, kinodynamic planning, floating-base planning, and
  controller execution are not included.
