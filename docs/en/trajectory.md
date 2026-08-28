# Trajectory timing

<div class="language-switcher">English · <a href="../zh_CN/trajectory.html">简体中文</a></div>

HolisticMotion provides geometric path generation and time parametrization as
separate operations. `ToppraTrajectory` retimes an existing joint waypoint
path while preserving its geometry.

```python
from holistic_motion.trajectory import ToppraTrajectory

trajectory = ToppraTrajectory(
    [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
    max_velocity=[1.0, 0.8],
    max_acceleration=[2.0, 1.5],
)
times, position, velocity, acceleration = trajectory.sample_uniform(200)
```

Waypoint, limit, and timing-result arrays are immutable owned snapshots.
`sample()` and `sample_uniform()` return new arrays that applications may
modify freely without changing the trajectory object.

The implementation uses squared path velocity as its state, propagates
controllable sets backward, then performs a time-optimal forward pass. It has
no runtime dependency on the upstream TOPPRA package or an external LP/QP
solver.

Run the example:

```bash
./scripts/run.sh python3 examples/python/trajectory/toppra_retiming.py
./scripts/run.sh python3 examples/python/trajectory/toppra_retiming.py --plot
```

For an interactive Viser view of the joint path, playback cursor, velocity and
acceleration traces, and live constraint utilization:

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/visualization/toppra_viser.py --autoplay --loop
```

To animate the TOPPRA result on the full Marvin URDF, including both arm paths
and all visual meshes:

```bash
./scripts/run.sh python3 \
  examples/python/visualization/toppra_robot_viser.py \
  --urdf /path/to/robot.urdf --autoplay --loop
```

Trajectory charts are separated into left- and right-arm tabs. Their compact
`J1`–`J7` legends use consistent colors and include a mapping table to the full
URDF joint names. Click a legend entry to isolate one curve.

Pass either `--asset-root` with a `--profile`, or an explicit `--urdf` path.
The compatibility spellings `--urdf-path` and `--urdf_path` are also accepted.
When launched directly from a built source checkout, the example automatically
enters `scripts/run.sh`.

The path interpolator is a joint-space natural cubic spline. Constraints are
enforced on its analytic derivatives at the reachability grid points; use a
denser `grid_size` for paths with high curvature and validate sampled output
before commanding hardware.
