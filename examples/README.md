# Examples

Examples are grouped first by language and then by robotics domain.

Start with the classic self-contained pipeline: a 2-D point robot uses
RRT-Connect to route around a wall, then applies native Double-S timing to the
collision-checked polyline:

```bash
./scripts/run.sh python3 \
  examples/python/planning/classic_plan_and_retime.py
./scripts/run.sh python3 \
  examples/python/planning/classic_plan_and_retime.py --plot
```

It needs no robot assets or optional solver. The plot flag requires Matplotlib.
Use `--profile trapezoidal` to compare timing profiles. The example deliberately
uses zero blend tolerance because smoothing a checked polyline requires another
collision check. A deterministic farthest-visible shortcut pass rechecks every
candidate edge with the same margin and typically reduces 17 planner points to
5 execution waypoints, avoiding unnecessary stops at tree expansion points.

The classic upper-body sequence aligns the torso first, then solves one or both
arms from the aligned configuration. Supply every asset and robot-specific name
explicitly; repeat a joint option for each joint in that group:

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/retargeting/classic_torso_first.py \
  --urdf /absolute/path/to/robot.urdf \
  --torso-frame torso_link --torso-joint waist_yaw \
  --left-frame left_tool --left-joint left_shoulder --left-joint left_elbow \
  --right-frame right_tool --right-joint right_shoulder --right-joint right_elbow \
  --joint-delta waist_yaw=-0.2 --joint-delta left_elbow=0.4
```

Use `--arm-mode left_arm` or `right_arm` for a single arm. The demo creates
reachable targets from the neutral configuration; its scalar joint deltas must
remain inside the URDF limits. `--torso-delta` and `--arm-delta` provide group
defaults; repeat `--joint-delta JOINT=DELTA` for robot-specific overrides. It
reports two offline IK waypoints, which still
need collision-checked planning before execution. The output includes each
stage's convergence diagnostics and solve time, named torso/arm joint values,
and the complete model configurations.

- `python/collision/`: collision queries, pair policies, group checks, and path scans.
- `python/visualization/`: interactive Viser applications.
- `python/kinematics/`: forward/inverse kinematics and CUDA benchmarks.
- `python/trajectory/`: trajectory generation and visualization.
- `python/retargeting/`: Pinocchio and Pink-style task-space IK examples.
- `python/planning/`: dependency-free RRT algorithm examples.
- `configs/`: robot-specific example profiles; library code never depends on them.

Legacy launchers under `python/examples/` remain available during the migration.

All Viser applications use the shared studio scene environment: a blurred HDRI
background, balanced environment/fill/key lighting, shadow-casting default
lights, and a solid metric floor grid. Robot scenes use a 4 m floor, compact
plots and editors use a 3 m floor, and mesh fitting places a scale-aware floor
just below the supplied geometry. New examples should call
`holistic_motion.visualization.viser.configure_scene()` after creating their
`ViserServer`; mesh viewers can use `configure_scene_from_bounds()` to center
and size the floor from finite world-space bounds.

TOPPRA timing can be inspected interactively with:

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/visualization/toppra_viser.py --autoplay --loop
```

The URDF-backed dual-arm version is:

```bash
./scripts/run.sh python3 \
  examples/python/visualization/toppra_robot_viser.py --autoplay --loop
```

Pink-style interactive retargeting is available with:

```bash
python3 examples/python/visualization/pink_robot_viser.py
```

Compare the native RRT variants and launch the collision-checked robot demo:

```bash
python3 examples/python/planning/rrt_variants.py
./scripts/run.sh python3 examples/python/visualization/rrt_robot_viser.py \
  --urdf /absolute/path/to/robot_with_ee.urdf
```

The robot demo uses balanced mirrored-arm planning by default. Pass
`--planning-space coupled` for a full independent 14-DoF search.
