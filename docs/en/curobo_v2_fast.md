# NVIDIA cuRobo v2 planning

<div class="language-switcher">English · <a href="../zh_CN/curobo_v2_fast.html">简体中文</a></div>

`holistic_motion.kit.curobo_planner` integrates the current NVIDIA cuRobo v2
`MotionPlanner`, with RRT, B-spline optimization and TimeOpt enabled. It supports
rest-to-rest joint goals, named tool-pose goals, FK and scene replacement. The
backend is optional and independent of the native collision extension. Dynamics
are explicitly enabled in every planning stage, including when the caller's
upstream example YAML has `load_dynamics: false`.

## Install the optional runtime

Use Python 3.10+ and an NVIDIA GPU with a compatible CUDA/PyTorch installation.
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

## Scenes, results and lifetime

`update_link_inertial(link_name, mass=..., com=..., inertia=...)` replaces a
link's complete inertial properties. COM is a link-frame [3] vector in meters;
inertia about the COM is `[ixx, iyy, izz, ixy, ixz, iyz]` in link axes, kg*m².
The adapter validates finite float32 values, non-negative mass and physically
consistent principal moments, updates upstream dynamics and clears RRT state.
This does not attach collision geometry. A backend inertial-update failure
blocks further planning and requires recreating the instance.

`scene=` and `update_world(scene)` accept a local scene YAML, upstream scene
dictionary or `curobo.scene.Scene`. Poses use the configured robot base frame
and upstream `xyz, wxyz` layout. `None` is an empty world; self-collision checking
remains enabled. Updates replace the entire world and reset RRT search state.
Reserve collision capacity at construction. A failed world update blocks
planning until a replacement update succeeds. Requests and updates on one
instance are serialized; use separate instances for concurrent GPU work.

A successful `CuroboPlanningResult` contains a `CuroboTrajectory` with read-only,
owned NumPy `time`, `position`, `velocity`, `acceleration` and optional `jerk`
arrays. Rows follow the trimmed interpolation horizon, the returned timestep
and requested joint order; locked joints and padding are removed. Failed
requests contain a message and no trajectory. Invalid caller inputs raise
before solving. Results are samples, not an interpolating trajectory object.
Do not refit them through TOPPRA and assume their original constraints survive.

cuRobo's sampled collision/dynamics acceptance does not establish continuous-time
feasibility. The existing HolisticMotion TOPPRA constrains velocity and
acceleration; it is not a torque/jerk equivalent. Review collision-sphere coverage
and robot dynamics for the intended workload before using planned motions.
Controllers and hardware execution are outside this adapter.

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
