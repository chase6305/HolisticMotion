# Compatibility launchers and legacy examples

New examples are grouped by domain under
[`examples/python/`](../../examples/python/). This directory retains both
compatibility launchers and implementations that have not moved yet.

The four collision launchers (`collision_demo.py`,
`collision_pair_management_demo.py`, `collision_path_scan_demo.py`, and
`dual_arm_collision_demo.py`) forward to the organized collision examples.
`viser_dual_arm_gizmo.py` forwards to the organized visualization example, and
`viser_w1_dual_arm.py` forwards through that compatibility launcher.

The remaining kinematics demos, `fep_cuda_benchmark.py`, the examples under
`trajectory/`, and the other Viser applications still contain implementations.
`_bootstrap.py` and `_viser_utils.py` provide shared support. Keep these files
until their callers and implementations have been migrated together.

Robot assets are never downloaded or inferred from a developer machine. Pass
`--urdf /absolute/path/to/robot.urdf` (or an explicitly supported
`--asset-root`) when launching an example.
