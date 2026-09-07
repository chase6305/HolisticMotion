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

The implementation uses squared path velocity as its state, propagates both
bounds of controllable intervals backward, then selects the largest reachable
speed at each forward step. It has
no runtime dependency on the upstream TOPPRA package or an external LP/QP
solver.

`start_path_velocity` and `end_path_velocity` specify endpoint speeds of the
normalized chord-length parameter, both zero by default. Feasible results
preserve these values. Requests that are infeasible under the current grid's
constraints raise `ValueError`; no subsequent global time scaling changes the
requested endpoint speeds.

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

The path interpolator is a joint-space natural cubic spline. Each interval uses
analytic extrema of the path derivative to bound velocity, and Bernstein
coefficients of the quadratic acceleration polynomial to bound acceleration.
These bounds cover the whole interval and both sides of each knot. They are
conservative and do not guarantee the globally shortest continuous trajectory;
increasing `grid_size` can reduce that conservatism. `ToppraResult` also checks
that its speed, acceleration, and time arrays describe consistent motion.


The C++ `PathSegBezierCurve5th` segment uses the full tangent metric consistently
for its curve-length coefficients, including rotation for SE3 paths. The
Cartesian flag does not truncate tangent vectors to three entries. One- and
two-dimensional Rn segments therefore remain valid with that flag, and a straight
pure-rotation segment has the same path-length convention as a linear segment.
