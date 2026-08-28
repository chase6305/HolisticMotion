# HolisticMotion

English | [简体中文](README.zh-CN.md)

HolisticMotion is a focused C++17 robotics library with Python bindings for
URDF robot models, kinematics, manifolds, solver-adjacent algorithms,
trajectory generation, optional Pinocchio/Coal collision queries, and pose
retargeting.

Robot assets are external. Every API that needs a model accepts an explicit
URDF path; the project never downloads models implicitly.

![HolisticMotion modular capabilities: robot model, kinematics, retargeting, trajectory, planning, and collision](docs/assets/holistic-motion-overview.png)

## Highlights

- C++17 library with pybind11 bindings.
- URDF parsing and numerical, OPW, UR, SRS, and FEP kinematics.
- Constraint-aware Double-S, trapezoidal, and TOPPRA path timing.
- Dependency-free RRT-Connect, RRT*, and Informed RRT* joint-space planning.
- Optional CUDA batch backends, enabled explicitly with `--cuda`.
- Collision queries backed by Conan-managed Pinocchio and Coal, enabled by default.
- Pinocchio-backed task-space retargeting with single-arm, dual-arm, and whole-body modes.
- English and Simplified Chinese documentation.

## Components

![HolisticMotion component layers: public APIs, peer core modules, and development tools](docs/assets/holistic-motion-components.png)

| Area | Main API | Notes |
| --- | --- | --- |
| Robot model and kinematics | `holistic_motion::Robot`, `holistic_motion.Robot` | URDF, FK/IK, OPW, UR, SRS, and FEP |
| Trajectory generation | `holistic_motion.trajectory` | Double-S, trapezoidal, and locally maintained TOPPRA implementation |
| Retargeting toolkit | `holistic_motion.kit.retargeting` | Pinocchio-backed single-arm, dual-arm, and whole-body modes |
| Collision | `CollisionModel`, `SphereCollisionModel` | Exact mesh queries through Pinocchio/Coal and lightweight sphere queries |
| Sampling planning | `holistic_motion.planning` | Locally implemented RRT-Connect, RRT*, and Informed RRT* |

## Quick start

```bash
./scripts/build.sh
source scripts/activate.sh
```

The default build includes Python bindings and the Conan-managed Pinocchio/Coal
collision component. Use `--cuda` to add the CUDA backend or `--no-collision`
for a lighter build; add `--tests` to run the test suites.
The activation step exposes both the local Python package and Conan-managed
shared libraries. For one command, use `./scripts/run.sh python3 ...` instead.

Run the collision demo against your own URDF and configuration:

```bash
python3 examples/python/collision/basic_query.py \
  --urdf /absolute/path/to/robot.urdf -q 0 0 0 0 0 0
```

Generate an editable collision-sphere model from URDF collision geometry:

```bash
./scripts/run.sh python3 examples/python/collision/fit_urdf_spheres.py \
  --urdf /absolute/path/to/robot.urdf \
  --output build/robot_spheres.json

./scripts/run.sh python3 examples/python/visualization/sphere_model_editor.py \
  --urdf /absolute/path/to/robot.urdf \
  --spheres build/robot_spheres.json
```

Sphere fitting is automatic, while the Viser editor lets users inspect and
adjust link-local spheres before using them for planning or batch collision
queries. See the [collision guide](docs/en/collision.md) for coverage metrics,
collision groups, and exact-versus-sphere query trade-offs.

## Python example

```python
import holistic_motion as hm

robot = hm.Robot("/absolute/path/to/robot.urdf")
q = [0.0] * robot.dof
pose = robot.kinematics.forward(q)
solution = robot.kinematics.inverse(pose, q)
```

Retime a waypoint path without an external TOPPRA installation:

```python
from holistic_motion.trajectory import ToppraTrajectory

trajectory = ToppraTrajectory(
    [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
    max_velocity=[1.0, 0.8],
    max_acceleration=[2.0, 1.5],
)
times, positions, velocities, accelerations = trajectory.sample_uniform(200)
```

Retargeting is available under `holistic_motion.kit.retargeting`:

```python
from holistic_motion.kit.retargeting import PinkRetargetingSolver

solver = PinkRetargetingSolver(
    "/absolute/path/to/robot.urdf",
    frames={"left_hand": "left_ee", "right_hand": "right_ee", "head": "head_ee"},
    joint_groups={
        "left_arm": ["left_j1", "left_j2"],
        "right_arm": ["right_j1", "right_j2"],
    },
)
solver.set_mode("dual_arm")
result = solver.solve({"left_hand": left_pose, "right_hand": right_pose})
```

Install its Pinocchio Python runtime with
`python -m pip install '.[retargeting]'`; upstream Pink is not installed.

Run the native dual-arm sampling planner and Viser animation:

```bash
./scripts/run.sh python3 examples/python/visualization/rrt_robot_viser.py \
  --urdf /absolute/path/to/robot_with_ee.urdf
```

The planner is implemented inside HolisticMotion and does not depend on OMPL.
See the [planning guide](docs/en/planning.md) for collision adapters and tuning.

## Documentation

```bash
python -m pip install '.[docs]'
./scripts/docs.sh
```

- English output: `docs/_build/html/en/index.html`
- 中文输出: `docs/_build/html/zh_CN/index.html`

See the [English documentation](docs/en/index.md) for installation, concepts,
tutorials, API references, architecture, and contribution guidance.

## Dependencies and attribution

| Project | Relationship | License |
| --- | --- | --- |
| [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | Conan-managed C++ collision/kinematics dependency when collision support is enabled | BSD 2-Clause |
| [Coal](https://github.com/humanoid-path-planner/coal) | Conan-managed narrow-phase collision dependency when collision support is enabled | BSD |
| [Pink](https://github.com/stephane-caron/pink) | Algorithm and API-design reference for the locally maintained task-based retargeting solver; not imported or vendored | Apache-2.0 |
| [cuRobo](https://github.com/NVlabs/curobo) | Design reference for collision spheres and batched queries; not imported or vendored, with no Torch/Warp dependency | Apache-2.0 |
| [TOPPRA](https://github.com/hungpham2511/toppra) | Algorithm reference for the locally maintained path parameterization implementation; not a runtime dependency | MIT |

Pinocchio and Coal are resolved by Conan at the versions declared in
[`conanfile.py`](conanfile.py); the current defaults are Pinocchio 3.8.0 and
Coal 3.0.2. Disable both with `./scripts/build.sh --no-collision`. Pink and
cuRobo source trees are not included in this repository. Detailed scope,
copyright, and license notices are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Validation

```bash
./scripts/build.sh --tests
conan create . --no-remote -o '&:with_python=False'
```
