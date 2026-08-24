# Pose retargeting

<div class="language-switcher">English · <a href="../zh_CN/retargeting.html">简体中文</a></div>

Install Pinocchio's Python runtime and import the toolkit:

```bash
python -m pip install '.[retargeting]'
```

```python
from holistic_motion.kit.retargeting import (
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
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
)
```

## Modes

| Mode | Required targets | Active velocities |
|---|---|---|
| `left_arm` | `left_hand` | Left-arm group |
| `right_arm` | `right_hand` | Right-arm group |
| `dual_arm` | Both hands | Both arm groups |
| `whole_body` | Both hands and head | Entire model |

```python
solver.set_mode("dual_arm")
result = solver.solve({
    "left_hand": left_pose,
    "right_hand": right_pose,
})

if result.success:
    send_joint_command(result.configuration)
```

The previous result is reused as a warm start. Pass `seed=` to override it or
call `solver.reset()` to return to Pinocchio's neutral configuration.

## Solver choices

- `PinocchioRetargetingSolver` provides compact damped least-squares IK.
- `PinkRetargetingSolver` assembles a Pink-style weighted task objective and
  adds anisotropic frame costs, posture regularization, LM damping, and
  velocity-limited integration.

The Pink-style implementation is maintained inside HolisticMotion. It does not
import the upstream `pink` package or `qpsolvers` at runtime. See
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
python3 examples/python/visualization/pink_robot_viser.py
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
The generic retargeting solver does not perform collision checking. Use the
optional C++/Python `CollisionModel` component before accepting a command when
your application requires collision rejection.
```
