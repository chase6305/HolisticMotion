# HolisticMotion

English | [简体中文](README.zh-CN.md)

A C++17 robotics library with Python bindings for robot kinematics, motion
planning, trajectory generation, collision queries, and pose retargeting.
Robot models are supplied by the caller through explicit URDF paths.

![HolisticMotion modules](docs/assets/holistic-motion-overview.png)

## Features

| Module | Capabilities |
| --- | --- |
| Kinematics | URDF models, FK/IK, numerical, OPW, UR, SRS, and FEP solvers |
| Planning | Null-space planning, RRT-Connect, RRT*, Informed RRT*, and feasible path optimization |
| Trajectories | Double-S, trapezoidal, TOPPRA, and optional differentiable PyTorch timing |
| Collision | Optional Pinocchio/Coal mesh queries and sphere models |
| Retargeting | Arm, leg, full-body, and optional hand retargeting |

CUDA batch backends and Viser examples are optional. Planning runs without
OMPL; the TOPPRA implementation is maintained in this repository.

## Quick start

Requires a C++17 compiler, CMake 3.22+, Conan 2, and Python 3.9+.
See [installation](docs/en/installation.md) for dependency setup and C++ usage.

```bash
./scripts/build.sh
source scripts/activate.sh
```

The default build includes Python bindings and collision support. Add
`--cuda` for CUDA, `--no-collision` for a lighter build, or `--tests` to run tests.
Use `./scripts/run.sh python3 ...` to run a single command with the local install.

```python
import holistic_motion as hm

robot = hm.Robot("/absolute/path/to/robot.urdf")
q = [0.0] * robot.dof
pose = robot.kinematics.forward(q)
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

Optional Python extras include `differentiable`, `retargeting`,
`hand-retargeting`, and `examples`. See the corresponding guides below for
installation and runtime requirements.

## Documentation and examples

- [Kinematics and core concepts](docs/en/concepts.md)
- [Planning](docs/en/planning.md) · [Trajectories](docs/en/trajectory.md)
- [Collision and sphere fitting](docs/en/collision.md)
- [Retargeting](docs/en/retargeting.md)
- [Runnable examples](examples/README.md) · [Full documentation](docs/en/index.md)

Build the English and Chinese documentation locally with
`python -m pip install '.[docs]'` followed by `./scripts/docs.sh`.

## Development

```bash
./scripts/build.sh --tests
conan create . --no-remote -o '&:with_python=False'
```

See [development](docs/en/development.md) for tests, benchmarks, and contribution
guidelines. Dependency licenses and algorithm references are recorded in
[third-party notices](THIRD_PARTY_NOTICES.md).
