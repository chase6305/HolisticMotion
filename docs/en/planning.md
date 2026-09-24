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

When no state validator is configured, edge sampling is skipped: bounded joint
interpolation remains inside its limits and continuous joints wrap normally.
`edge_resolution` then has no validation work to control, and `collision_checks`
remains zero. Configure a validator to check obstacles or additional constraints.

Joint weights control search distances, not a goal tolerance. A direct solution
preserves distinct endpoints even with very small weights. RRT-Connect joins
its trees only after a validated extension actually reaches the target state;
a small weighted gap cannot substitute for checking the connecting motion.

For mixed bounded/continuous spaces with up to eight coordinates, trees of at
least 128 states use an internal balanced spatial index. It splits only bounded
coordinates and evaluates candidates with the original weighted, wrapped
metric. Recent insertions are scanned until the next rebuild; ties and radius
results retain insertion order. Small trees, fully bounded or fully continuous
spaces, and higher dimensions retain linear searches. No external indexing
dependency or public configuration option is required.

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

## Feasibility-preserving path optimization

`PathOptimizer` provides a lightweight post-processing stage for a feasible
geometric path. It keeps both endpoints fixed and iteratively reduces a joint-
weighted combination of path length and second-difference smoothness. Each
candidate waypoint is accepted only when the objective decreases and both
adjacent edges pass the configured collision validator.

Input validation checks the original waypoint values, including each edge's
exact endpoint. Reconstructing an endpoint by interpolation can round it across
a validity boundary, so only interior samples are interpolated. This preserves
the existing callback order and avoids rechecking already validated endpoints
during candidate updates.

```python
optimizer = hm.PathOptimizer.from_collision_joints(
    collision,
    joint_names,
    context,
    security_margin=0.005,
    clearance=0.02,
)
optimization = hm.PathOptimizationOptions()
optimization.timeout_seconds = 0.5
optimization.smoothness_weight = 0.5
optimization.state_cost_weight = 2.0
optimized = optimizer.optimize(result.path, optimization)
if not optimized.success:
    raise RuntimeError(optimized.message)
path = optimized.path
```

`security_margin` is a hard feasibility boundary. `clearance` is a soft target:
the native adapter adds a squared hinge cost when the minimum collision
distance falls below it. The optimizer estimates the state-cost gradient with
central finite differences (`finite_difference_step`) and combines it with the
analytic geometric gradient. `state_cost_step_size` scales the clearance
gradient before the combined direction is normalized. Leaving `clearance=0` and
`state_cost_weight=0` disables this stage completely.
Forward and reverse waypoint sweeps alternate between iterations to reduce
the directional bias of in-place updates.

The generic API accepts the same mechanism through
`optimizer.set_state_cost(callable)`. Costs must be finite, non-negative, and
cheap enough for repeated evaluation. Statistics expose
`state_cost_evaluations` separately from collision validity checks, while
`line_search_evaluations` records the number of candidate step sizes tested.
`iterations` counts every optimization sweep that was entered, including the
final converged or timed-out sweep.

Applications that already have analytic, automatic-differentiation, or
kinematic Jacobian gradients can additionally call
`optimizer.set_state_cost_gradient(callable)`. The callback returns one finite
gradient value per active joint. This replaces the default central finite
difference and reduces gradient work from `2 * DoF` cost calls to one gradient
call per attempted waypoint update. `state_cost_gradient_evaluations` records
this path independently; omitting the callback preserves the finite-difference
fallback.
At a bounded joint limit, a finite-difference side that clamps to the current
state reuses the waypoint's cached cost. Unequal sample offsets use the
three-point derivative at the waypoint, so projection does not shift the
derivative to the samples' midpoint. A nearly collapsed side is omitted using
a relative offset comparison. Continuous-joint steps are capped at `pi/2`
to keep the samples on opposite sides of the waypoint; unrepresentable
numerical gradients raise `ValueError`.
Preconditioning preserves relative descent directions even when the reciprocal
of a finite positive joint weight would overflow. Large geometric gradients use
an extended-precision fallback when intermediate arithmetic is non-finite;
a combined objective gradient that still cannot be represented raises
`ValueError` before candidate callbacks run.

A timeout still returns the best feasible path found so far. Invalid input
paths are rejected rather than repaired, so sampling remains responsible for
finding the initial feasible homotopy. This follows cuRobo's useful separation
of seed generation, feasibility-aware optimization, and best-solution
tracking, without importing its Torch, Warp, or CUDA optimizer stack. TOPPRA
still performs the subsequent velocity and acceleration retiming.
The deadline is checked after every validation callback, including the last
sample of an edge. A timeout during input validation returns an empty path with
`success=False`; a candidate whose validation overruns the deadline is discarded
and the previously validated path is retained. Callbacks run synchronously and
cannot be interrupted, so a slow callback can exceed the requested wall time.
If the deadline expires before all initial state costs are available, the
validated input path is returned and both objective statistics are `NaN`
because a complete objective was never evaluated.

Waypoint updates use an incremental objective: moving one interior waypoint
recomputes only its two adjacent length terms and the at most three affected
second-difference terms. Full passes compute the initial and final reported
objectives; accepted updates maintain the running objective by local deltas.
Geometric bookkeeping therefore scales linearly with waypoint count per sweep
for a fixed line-search budget. Collision and state-cost callbacks add their own
costs.

A workspace local to each `optimize()` call reuses edge, acceleration, gradient,
direction, and candidate buffers. The two outer neighboring edges remain cached
during one waypoint's backtracking search and refresh before the next waypoint,
including reverse sweeps. Local geometry and gradient evaluation allocate no
Eigen heap storage after workspace/output initialization. The complete optimizer
still allocates its path, validation samples, and optional state-cost buffers.

Each waypoint update uses bounded backtracking. `line_search_steps` controls
how many step sizes are attempted, while `line_search_decay` scales each
retry. This recovers a smaller update when the initial step collides or
overshoots, without accepting an infeasible intermediate path.
Because state costs are non-negative, a candidate is discarded before its
callbacks when its zero-cost lower bound still cannot satisfy
`minimum_improvement`.
Zero gradients and updates that clamp back to the current joint state terminate
their waypoint search without evaluating redundant callbacks.
A validated two-point path returns immediately because it has no interior
waypoints or state costs to optimize.

The update direction is the analytic gradient of the enabled length and
second-difference terms, combined with the optional state-cost gradient.
Inverse joint weights precondition that gradient before its largest component
is normalized, so objective weights affect both the search direction and the
acceptance test.
Default inverse-range-squared weights must be finite and positive; numerically
unrepresentable joint ranges are rejected during construction.

## Resolution and safety

RRT* and Informed RRT* refresh the best goal cost after rewiring, including
improvements propagated through ancestor nodes. With a fixed random seed and
completed iteration budgets, further search preserves or improves the incumbent
path before optional shortcutting. A wall-clock deadline can change how much of
the search and postprocessing completes.

The planner validates each edge at intervals no larger than
`edge_resolution` in any active joint. Smaller values improve collision
coverage but increase Coal queries. Endpoints and every shortcut are checked.
Shortcuts also revalidate the retained edge fragments because splitting an edge
changes its collision sample grid. Optional path interpolation preserves every
original corner and subdivides the longest segments to approach even spacing.
`interpolation_points` is the target count when adding points; existing points
are never removed. New subsegments are checked before accepting the interpolated
path. If that check fails or the deadline expires, the original path is retained,
so the returned count can differ from the target. Discrete validation cannot
exclude collisions between samples; choose the resolution for the application.
The wrapped joint difference is computed once per edge and reused by all
samples, preserving the shortest arc for continuous joints.
Extremely small positive resolutions use saturating segment counts, so the
deadline remains authoritative without floating-point-to-integer overflow.
When no state validator is configured, edge sampling is skipped completely;
joint-limit feasibility follows from valid endpoints and convex interpolation.
In that mode `collision_checks` remains zero; otherwise it equals the number
of actual validator callback invocations.
The final path is geometric and must still be validated by the application if
the environment changes after planning.

## Examples

Run the classic self-contained plan-and-retime pipeline first:

```bash
./scripts/run.sh python3 \
  examples/python/planning/classic_plan_and_retime.py
./scripts/run.sh python3 \
  examples/python/planning/classic_plan_and_retime.py --plot
```

It plans a point robot around a wall with deterministic RRT-Connect, then
applies native Double-S timing and rechecks sampled positions and derivative
limits. It uses unit metric weights so the reported length is Euclidean, adds a
margin larger than the discrete edge resolution, and keeps zero blend tolerance
so timing cannot cut a checked corner. It requires no robot asset or optional
solver; only `--plot` requires Matplotlib. Pass `--profile trapezoidal` to
compare the second native timing profile. Before timing, a deterministic
farthest-visible pass removes redundant RRT points and rechecks each candidate
edge with the same margin; this reduces avoidable full stops while preserving
the checked polyline semantics.

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
- The path optimizer is a local CPU smoother, not a global trajectory optimizer.
- Dynamic obstacles, kinodynamic planning, floating-base planning, and
  controller execution are not included.
