# NVIDIA cuRobo v2 planning

<div class="language-switcher">English · <a href="../zh_CN/curobo_v2_fast.html">简体中文</a></div>

`holistic_motion.kit.curobo_planner` integrates the current NVIDIA cuRobo v2
`MotionPlanner`, with RRT, B-spline optimization and TimeOpt enabled. It supports
rest-to-rest joint goals, named tool-pose goals, FK and scene replacement. The
backend is optional and independent of the native collision extension. Dynamics
are explicitly enabled in every planning stage, including when the caller's
upstream example YAML has `load_dynamics: false`.

The adapter uses upstream multi-seed IK with LM seed generation disabled. On
the qualified commit, the LM CUDA graph retains historical grad-enabled inputs
across pose requests; this configuration keeps repeated planning memory bounded.

## Install the optional runtime

Use Python 3.10+ and an NVIDIA GPU with a compatible CUDA/PyTorch installation.
cuRobo uses its separately installed CUDA runtime; HolisticMotion's native
`--cuda` build option controls the FEP batch component.
Follow the [upstream installation guide](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/docs/getting-started/installation.rst)
for your CUDA version. For example, in a separate CUDA 13 environment:

```bash
python -m pip install 'nvidia-curobo[cu13-torch] @ git+https://github.com/NVlabs/curobo.git@b29a8c2ee6869585bb05b056b9ab53b9c0753106'
```

The adapter checks for the v2 public planner and TimeOpt API; v1 `MotionGen`
is not supported. Ordinary HolisticMotion imports do not load cuRobo, Torch,
Warp or CUDA. No runtime installation or robot download occurs automatically.
The existing `CuroboRetargetingSolver` remains a separate local multi-seed IK
implementation; it does not use this NVIDIA backend.

## Plan with caller-provided assets

Supply an explicit local cuRobo robot YAML containing the URDF and mesh paths,
base/tool frames, collision spheres, locked joints, inertias and dynamics limits.
Use absolute URDF/mesh paths in that configuration when assets live outside
cuRobo. The adapter does not convert a HolisticMotion collision model or infer
robot geometry. Robot assets are never included in HolisticMotion.

```python
from holistic_motion.kit.curobo_planner import CuroboPlanner

with CuroboPlanner(
    "/absolute/path/robot.yml",
    scene="/absolute/path/scene.yml",  # Omit for an empty world.
    collision_cache={"cuboid": 8},
) as planner:
    planner.warmup()
    result = planner.plan(q_start, q_goal)
    if not result.success:
        raise RuntimeError(result.message)
    trajectory = result.trajectory
    # Arrays own their data and survive another solve or planner.close().
    joint_names = trajectory.joint_names
    times = trajectory.time
    positions = trajectory.position
```

`q_start` and `q_goal` are scalar positions in `planner.joint_names` order,
not full Pinocchio `nq` configurations. Optional `joint_names=` may permute the
complete active joint set; locked joints stay at their configured values.
`lower_limits` and `upper_limits` return independent arrays in that same order.
Inputs must be finite and within bounds, and must lose at most `1e-6` during
float32 conversion. The adapter does not infer continuous-joint winding.

Pose goals use base-frame homogeneous matrices in meters. Every configured
tool must have a target; no quaternion convention needs to be guessed:

Within the same open planner context:

```python
targets = planner.forward_kinematics(q_goal)  # Or caller-supplied 4x4 matrices.
result = planner.plan_pose(q_start, targets)
```

`position_tolerance` (default 0.005 m) and `orientation_tolerance` (default
0.05 rad) are configured at construction. The returned endpoint is checked
again with FK. `plan()` separately checks joint endpoints with its
`endpoint_tolerance` (default `1e-3`).

Both planning methods accept `enable_graph_attempt=0` to use RRT from the first
attempt. The default `1` first tries direct optimization and enables RRT on the
second attempt. `max_attempts` bounds retries.

## Scenes, results and lifetime

`update_link_inertial(link_name, mass=..., com=..., inertia=...)` replaces a
link's complete inertial properties. COM is a link-frame [3] vector in meters;
inertia about the COM is `[ixx, iyy, izz, ixy, ixz, iyz]` in link axes, kg*m².
The adapter validates finite float32 values, non-negative mass and physically
consistent principal moments, updates upstream dynamics and clears RRT state.
This does not attach collision geometry. A backend inertial-update failure
blocks further planning and requires recreating the instance.

`update_link_collision_spheres(link_name, spheres)` replaces all collision
spheres on a link with caller-supplied `[x, y, z, radius]` rows, in meters in
that link's frame. Radii must be positive. Reserve enough slots in the robot
configuration (for example, `extra_collision_spheres` on an attached-object
link); unused slots are disabled. Passing `None` restores that link's original
configured spheres. The update is shared by IK, RRT and trajectory collision
checking and clears cached RRT state. It does not update mass or remove world
obstacles. Update those explicitly when carrying an object. A partial backend
sphere-update failure requires recreating the planner.

```python
planner.update_link_collision_spheres("attached_object", [[0, 0, 0.1, 0.05]])
# Plan with this caller-defined payload geometry, then restore the YAML model.
planner.update_link_collision_spheres("attached_object", None)
```

`scene=` and `update_world(scene)` accept a local scene YAML, upstream scene
dictionary or `curobo.scene.Scene`. Poses use the configured robot base frame
and upstream `xyz, wxyz` layout. `None` is an empty world; self-collision checking
remains enabled. Updates replace the entire world and reset RRT search state.
Reserve collision capacity at construction. A failed world update blocks
planning until a replacement update succeeds. Requests and updates on one
instance are serialized; use separate instances for concurrent GPU work.

Scene dictionary keys are `cuboid`, `sphere`, `capsule`, `cylinder`, `mesh` and
`voxel`. Unknown types, duplicate names, invalid dimensions and non-unit pose
quaternions are rejected before replacing the current world. A public `Scene`
must keep its `objects` and typed obstacle lists consistent. Spheres are
represented as analytic capsules with coincident endpoints because the
qualified upstream loader otherwise omits them. Reserve `collision_cache={"capsule": N}`
for later sphere or capsule replacements; no mesh approximation is introduced.

Complete world replacement also discards old Warp mesh geometry. This prevents
the upstream name-only cache from reusing an old shape under the same name or
retaining every removed mesh. Mesh replacement waits for device work before
releasing geometry; allocated collision storage and captures are retained.
To move an existing obstacle efficiently, use
`update_obstacle_pose(name, transform)` with a base-frame 4x4 matrix. This keeps
its geometry and resets RRT state. A failed pose update requires a successful
complete `update_world()` before planning resumes.

A successful `CuroboPlanningResult` contains a `CuroboTrajectory` with read-only,
owned NumPy `time`, `position`, `velocity`, `acceleration` and optional `jerk`
arrays. Rows follow the trimmed interpolation horizon, the returned timestep
and requested joint order; locked joints and padding are removed. Failed
requests contain a message, available upstream validation failures, and no trajectory. Invalid caller inputs raise
before solving. Results are samples, not an interpolating trajectory object.
Do not refit them through TOPPRA and assume their original constraints survive.

cuRobo's sampled collision/dynamics acceptance does not establish continuous-time
feasibility. The existing HolisticMotion TOPPRA constrains velocity and
acceleration; it is not a torque/jerk equivalent. Review collision-sphere coverage
and robot dynamics for the intended workload before using planned motions.
Controllers and hardware execution are outside this adapter.

The NumPy interface manages the backend's internal autograd context. It can be
constructed and called inside a caller's `torch.no_grad()` or
`torch.inference_mode()` block; the caller's state is restored after each call.

## Run the example and tests

```bash
HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python python examples/python/planning/curobo_plan.py \
  --robot-config /absolute/path/robot.yml \
  --start 0 0 0 --goal 0.1 0.2 0.3 --output /tmp/trajectory.npz
```

Supply the correct number of joints for your robot. This example warms the
planner, reports completed wall time and exports named samples on success.
No visualization dependency is needed.

The CPU contract suite is `tests/python/test_curobo_planner.py`. Real GPU tests
are explicitly enabled with caller-provided assets and JSON coordinate arrays:

```bash
export HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG=/absolute/path/robot.yml
export HOLISTICMOTION_TEST_CUROBO_START='[0, 0, 0]'
export HOLISTICMOTION_TEST_CUROBO_GOAL='[0.1, 0.2, 0.3]'
HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python python -m pytest tests/python/test_curobo_planner_gpu.py -q
```

Validation targets upstream commit `b29a8c2ee6869585bb05b056b9ab53b9c0753106`.
GPU tests cover joint and pose planning, output ownership, obstacle rejection
and recovery after world replacement. A missing configuration skips GPU tests;
a configured test must actually construct and run the backend.
