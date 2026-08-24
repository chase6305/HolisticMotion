# Examples

Examples are grouped first by language and then by robotics domain.

- `python/collision/`: collision queries, pair policies, group checks, and path scans.
- `python/visualization/`: interactive Viser applications.
- `python/kinematics/`: forward/inverse kinematics and CUDA benchmarks.
- `python/trajectory/`: trajectory generation and visualization.
- `python/retargeting/`: Pinocchio and Pink-style task-space IK examples.
- `configs/`: robot-specific example profiles; library code never depends on them.

Legacy launchers under `python/examples/` remain available during the migration.

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
