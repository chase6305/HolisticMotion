# cuRoboV2-Fast integration proposal

<div class="language-switcher">English · <a href="../zh_CN/curobo_v2_fast.html">简体中文</a></div>

Status: design proposal, reviewed against HolisticMotion `da453ca` and upstream
cuRobo `b29a8c2ee6869585bb05b056b9ab53b9c0753106` on 2026-11-05.
This document adds no runtime integration. API names below describe inspected
upstream code; proposed HolisticMotion entry points are explicitly marked.

## Recommendation and scope

Integrate NVIDIA cuRobo as an explicitly selected, optional Python planning
backend first. Validate robot conversion, collision semantics, dynamics and
trajectory ownership before considering a C++ deployment component. Preserve
the existing CPU sampling planners and TOPPRA APIs.

The [cuRoboV2-Fast paper](https://arxiv.org/pdf/2611.06913v1) combines GPU RRT,
cubic B-spline trajectory optimization, differentiable inverse dynamics and
fixed-path TimeOpt. Its reported 4.4 ms mean with a 3 kg payload on RTX 5090 is
an upstream benchmark, not an expected HolisticMotion latency. TimeOpt changes
timing along a fixed spline under sampled velocity, acceleration, jerk and
torque constraints. These capabilities motivate an adapter; they do not make
the existing trajectory interfaces interchangeable.

Start with fixed-base robots, named scalar revolute/prismatic joints, a fixed
set of locked joints, one scene, one request at a time, and rest-to-rest joint
goals. Floating-base motion, coupled/mimic active coordinates, moving boundary
states and batched planning require separate qualification. CPU planners remain
available through their existing APIs; a failed GPU request must retain its
failure rather than silently substitute a path with different constraints.

## Existing capabilities and gaps

| HolisticMotion capability | Integration decision |
| --- | --- |
| `JointSpace` snapshots names, limits, weights and continuous coordinates | Use named scalar order as the adapter boundary; validate compatibility with cuRobo's active coordinates. |
| `SamplingPlanner` and `PathOptimizer` return geometric waypoints | Keep these results geometric. Spline fitting can change the path and requires new collision checks. |
| NumPy/Torch TOPPRA implementations | They currently constrain velocity and acceleration, not torque or jerk. Keep their contract and use them as a separate baseline. |
| Optional `CollisionModel` and `SphereCollisionModel` | Reuse caller-supplied geometry and pair selection where representable; reject unsupported conversions. |
| `kit.retargeting.CuroboRetargetingSolver` | A local multi-seed solver built on the Pinocchio retargeting implementation; it does not import NVIDIA cuRobo. Keep its API and distinguish it from the new backend. |
| Native FEP CUDA support | Does not establish cuRobo availability, device compatibility or shared model storage. Probe the optional backend independently. |

## Verified upstream entry points

The v2 API differs from v1. Do not build a new integration around the old
`curobo.wrap.reacher.motion_gen.MotionGen` interface. The
[upstream README](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/README.md)
describes the rewrite; use the inspected commit as the initial reproducibility
reference and record the installed version/commit in benchmark output.

| Task | Public API at the inspected commit |
| --- | --- |
| Configure planning | `curobo.motion_planner.MotionPlannerCfg.create(...)`; explicitly select `graph_planner_algorithm="rrt"` and `enable_timeopt=True`. The graph algorithm default is `"prm"`. |
| Construct and warm once | `MotionPlanner(config)` and `planner.warmup(enable_graph=True, ...)`. |
| Joint goals | `planner.plan_cspace(goal_state, current_state, ...)`, with named `JointState` inputs. Goal precedes start. |
| Pose goals, later | `planner.plan_pose(goal_tool_poses, current_state, ...)`, with named `GoalToolPose` tools. |
| Single successful result | Check `result is not None` and its success mask, then `result.get_interpolated_plan()`; retain returned joint order and `dt`. |
| Standalone fixed-path timing | `curobo.trajectory_timeopt.TrajectoryTimeOpt`, `TrajectoryTimeOptCfg`, `TrajectoryTimeOptResult`. |
| World and payload updates | `planner.update_world(...)`, `reset_seed()`, `update_link_inertial(...)` and `update_links_inertial(...)`; collision attachments are a separate operation. |

Sources: [planner configuration](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/_src/motion/motion_planner_cfg.py),
[planner implementation](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/_src/motion/motion_planner.py),
[public TimeOpt exports](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/trajectory_timeopt.py)
and [TimeOpt implementation](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/_src/motion/motion_trajectory_timeopt.py).
Private files here provide evidence; an adapter should import public modules.

## Adapter boundary

Proposed implementation location: one cohesive module,
`python/holistic_motion/kit/curobo_planner.py`, introduced in a follow-up PR.
Keep cuRobo/Torch/Warp imports inside explicit backend construction. Ordinary
package imports and CPU-only installations must continue to work without them.
Require callers to install a qualified CUDA-specific upstream environment;
there is no install command, dependency change or model download in this PR.
Upstream currently requires Python 3.10+, while HolisticMotion supports 3.9+.
That restriction belongs to the optional backend. Review upstream's transitive
dependencies, including Viser, without making them core requirements.

The first adapter should take caller-provided absolute URDF, mesh roots and
cuRobo robot configuration paths, a named `JointSpace`, and an explicit scene.
Validate that the configuration actually describes the same URDF/model and
frames; paths alone do not establish model identity. Use a qualified existing
cuRobo configuration first instead of adding an automatic URDF converter.

| Boundary | Required behavior |
| --- | --- |
| Joint order | Build a checked name permutation in each direction. Reject duplicates, missing names and unsupported active joints. Preserve locked-joint context. Never interpret full Pinocchio `nq` as scalar positions. |
| Continuous joints | Use the request's periodic topology and `JointSpace.UnwrapPath` / Python `unwrap_path` when exporting CPU paths. Qualify cuRobo's representative limits and winding behavior; reject unsupported cases rather than clipping across a seam. |
| Frames and units | Use meters, radians and seconds, transform goals/obstacles into the configured base frame, and convert quaternions explicitly to upstream `wxyz`. Validate the base and tool with FK comparisons. |
| Numeric conversion | Validate finite real inputs before conversion, then check that float32 CUDA values remain finite, within limits and within an agreed conversion tolerance. |
| Geometry | Translate supported link-local spheres, world cuboids and pair exclusions explicitly. Do not confuse link frames with joint frames. Match margins and enabled robot/world pairs. Reject scenes whose checks cannot be represented. |
| Dynamics | Require meaningful inertias, center-of-mass frames, gravity, velocity/acceleration/jerk limits and asymmetric effort bounds. Missing data must block a torque-limited mode; do not fill it with zero inertia or infinite effort. |
| Payload | Treat inertial updates and collision attachment as distinct required updates. Reserve attachment capacity, remove the world copy while attached, and restore it at the release pose on detach. |

Construction should compare FK at neutral and nontrivial configurations, joint
limits and representative self/world collision outcomes before planning. Sphere
approximations need reviewed coverage; equal URDFs do not imply equal collision
geometry. Coal mesh checks can be an independent qualification step when the
optional native collision component is installed. A CPU Python validator cannot
be inserted into a captured GPU collision loop; use it for final validation and
measure its additional cost.

## Trajectory and failure contract

Return a separate timed backend result rather than extending `PlanningResult`
with fields unrelated to geometric planning. The first implementation needs
only success/failure diagnostics and, on success, owned joint names, explicit
sample times, position, velocity and acceleration arrays. Include jerk/torque
when they were actually evaluated; retain upstream diagnostics rather than
inventing equivalent CPU iteration/collision counters.

For a single planning result, trim padding through the public helper, derive
times from its stored `dt`, and reorder every derivative consistently. For
standalone TimeOpt use its returned `time` array: the path sampling grid does
not imply a uniform time interval. Copy outputs before another solve or payload
update can overwrite reusable buffers. Test this ownership explicitly.

Do not refit the timed result through `ToppraTrajectory` or a generic interpolant:
that can change geometry, derivatives, timing and torque feasibility. Any
eventual trajectory wrapper must evaluate the qualified representation and
retain its owner, or state that its output is only sampled data.

No-result, unsuccessful mask, non-finite output, invalid horizon, wrong endpoint,
conversion mismatch or failed final checks must yield a failure with no usable
trajectory. Preserve upstream failure details; do not guess that every failure
is a timeout. An explicit CPU fallback may be offered later with its own result
and constraint contract.

## TimeOpt and deployment options

1. **First: complete GPU pipeline.** Let cuRobo own RRT initialization, spline
   optimization and timing. This validates upstream behavior without a second
   path representation at the solver boundary.
2. **Then: CPU planning plus GPU timing.** The standalone TimeOpt input is a
   fixed cubic B-spline control layout, start/goal and optional original
   `knot_dt`; it is not an arbitrary waypoint list or HolisticMotion natural
   cubic spline. Establish an exact conversion where possible, otherwise fit
   and explicitly validate the changed geometry before timing. Its current
   implementation requires CUDA float32, 1–64 DoF and 32 timing controls;
   capability checks must be tied to the qualified upstream version.
3. **Finally: optional native runtime.** Upstream already provides
   [C++ planning and TimeOpt APIs](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/cpp_runtime/README.md).
   Qualify Python first, then export matching model/layout/GPU artifacts and
   evaluate a separately enabled C++ component. This needs artifact lifecycle,
   packaging and ABI work; it is not just linking the existing FEP CUDA target.
   Keep controllers, calibration and visualization outside the core library.

Construct solvers and allocate scene/attachment capacities once. Warm all used
branches before measuring, including RRT fallback. Serialize scene, payload and
locked-state edits with solves; invalidate graph-search state after scene
changes and reinitialize/rewarm for incompatible layouts or capacities. Use
one independent workspace per concurrent request. Standalone TimeOpt must see
the same updated dynamics as the planner if a separate instance is used.

Upstream TimeOpt explicitly defines success as sampled feasibility. Cubic path
knots can have jerk discontinuities; inspect both sides. Dense independent
collision/dynamics checks improve qualification but do not establish a
continuous-time certificate. Fixed-path timing also does not make a path safe
against time-dependent obstacles; that would require a separate formulation.

## Follow-up PRs and acceptance evidence

| Stage | Deliverable and exit condition |
| --- | --- |
| 1: isolated adapter | Explicit optional import, qualified model/config, fixed-base rest-to-rest `plan_cspace`, owned timed outputs, clear unsupported-input failures; CPU imports and existing APIs still pass. |
| 2: scene and dynamics | Reproduce blocked routes, self-collision, frame/margin cases and attach/detach payload changes. Compare FK and inverse dynamics against independent Pinocchio calculations using the same inertias and conventions. |
| 3: timing and deployment | Qualify standalone path conversion and TimeOpt, then evaluate C++ artifacts only if deployment needs justify them. |

Use temporary URDFs for coordinate/frame/error tests. GPU integration tests must
take explicit external robot/config/mesh paths and record them; an unavailable
GPU or skipped test is not successful integration validation. Cover joint-name
permutations, continuous seams, float32 loss, invalid inertia/effort, start or
goal in collision, stationary paths, infeasible gravity/payload torque, stale
scene state and output ownership across repeated solves. Compare trajectory
derivatives and torque on independent grids, including knot sides and endpoints.

Benchmark CPU RRT-Connect plus existing TOPPRA, full cuRobo, and later the hybrid
route on the same requests, geometry, limits, margins and budgets. Report
success and rejected outputs alongside median/p95/p99 wall latency, trajectory
duration, minimum clearance and derivative/effort utilization. Separate cold
construction/compilation/capture, warmed GPU solve, transfers, final validation
and total adapter latency; synchronize CUDA for completed-work timings. Record
GPU/CPU, driver/CUDA, Python/Torch/Warp/cuRobo versions, seeds and configuration.
Match the constraints when comparing duration; the current TOPPRA baseline does
not enforce torque/jerk. Set performance targets only after this measurement.

This proposal was checked by source inspection and documentation builds only.
No cuRobo installation, GPU run, dynamics parity or speedup is claimed.
