#!/usr/bin/env bash

# Test the optional Python robotics runtime in its own process.
set -euo pipefail

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
PYTHON_EXECUTABLE="$("${PYTHON_EXECUTABLE}" -I -S -c 'import sys; print(sys.executable)')"
export HOLISTICMOTION_PURE_PYTHON=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="${REPOSITORY_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPOSITORY_DIR}"

# CI must fail when its optional dependency was not installed, rather than
# report a successful run in which every numerical retargeting test skipped.
"${PYTHON_EXECUTABLE}" -c 'import pinocchio; print("Python Pinocchio:", pinocchio.__version__)'
exec "${PYTHON_EXECUTABLE}" -m pytest \
    tests/python/test_frame_task_errors.py \
    tests/python/test_immobile_retargeting.py \
    tests/python/test_fixed_collision_coordinates.py \
    tests/python/test_retargeting_modes.py \
    tests/python/test_collision_gradient.py \
    tests/python/test_anisotropic_frame_jacobian.py \
    tests/python/test_support_polygon_geometry.py \
    tests/python/test_retargeting_failure_history.py \
    tests/python/test_lazy_retargeting_seeds.py \
    tests/python/test_task_gains.py \
    tests/python/test_disabled_balance_tasks.py \
    tests/python/test_retargeting_step_scale.py \
    tests/python/test_retargeting_qp_scaling.py \
    tests/python/test_retargeting_box_qp.py \
    tests/python/test_upper_body_retargeting.py \
    tests/python/test_curobo_retargeting.py \
    tests/python/test_classic_torso_first_example.py \
    tests/python/test_pink_retargeting.py \
    tests/python/test_joint_position_map.py \
    tests/python/test_urdf_joint_axes.py \
    tests/python/test_runtime.py \
    "$@"
