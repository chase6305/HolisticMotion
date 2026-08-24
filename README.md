# HolisticMotion

English | [简体中文](README.zh-CN.md)

HolisticMotion is a focused C++17 robotics library with Python bindings for
URDF robot models, kinematics, manifolds, solver-adjacent algorithms,
trajectory generation, optional Pinocchio/Coal collision queries, and pose
retargeting.

Robot assets are external. Every API that needs a model accepts an explicit
URDF path; the project never downloads models implicitly.

## Highlights

- C++17 library with pybind11 bindings.
- URDF parsing and numerical, OPW, UR, SRS, and FEP kinematics.
- Constraint-aware Double-S, trapezoidal, and TOPPRA path timing.
- CUDA batch backends, enabled by default and easy to disable.
- Collision queries backed by Conan-managed Pinocchio and Coal, enabled by default.
- Pinocchio and Pink-style retargeting with single-arm, dual-arm, and whole-body modes.
- English and Simplified Chinese documentation.

## Quick start

```bash
./scripts/build.sh
source scripts/activate.sh
```

The default build includes Python bindings, CUDA, and the Conan-managed
Pinocchio/Coal collision component. Use `--no-cuda` or `--no-collision` on
machines that do not need those features; add `--tests` to run the test suites.
The activation step exposes both the local Python package and Conan-managed
shared libraries. For one command, use `./scripts/run.sh python3 ...` instead.

Run the collision demo against your own URDF and configuration:

```bash
python3 examples/python/collision/basic_query.py \
  --urdf /absolute/path/to/robot.urdf -q 0 0 0 0 0 0
```

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

Install the runtime dependency with `python -m pip install '.[retargeting]'`.

## Documentation

```bash
python -m pip install '.[docs]'
./scripts/docs.sh
```

- English output: `docs/_build/html/en/index.html`
- 中文输出: `docs/_build/html/zh_CN/index.html`

See the [English documentation](docs/en/index.md) for installation, concepts,
tutorials, API references, architecture, and contribution guidance.

## Validation

```bash
./scripts/build.sh --tests
conan create . --no-remote -o '&:with_python=False'
```
