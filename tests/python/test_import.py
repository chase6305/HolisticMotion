import os
import subprocess
import sys
from pathlib import Path


def test_import():
    import holistic_motion

    assert holistic_motion.__name__ == "holistic_motion"


def test_continuous_kinematics_api_imports():
    from holistic_motion.kinematics import (
        FEPContinuousOptions,
        FEPContinuousResult,
        FEPContinuousTracker,
        SRSContinuousOptions,
        SRSContinuousResult,
        SRSContinuousTracker,
    )

    assert FEPContinuousOptions.__name__ == "FEPContinuousOptions"
    assert FEPContinuousResult.__name__ == "FEPContinuousResult"
    assert FEPContinuousTracker.__name__ == "FEPContinuousTracker"
    assert SRSContinuousOptions.__name__ == "SRSContinuousOptions"
    assert SRSContinuousResult.__name__ == "SRSContinuousResult"
    assert SRSContinuousTracker.__name__ == "SRSContinuousTracker"


def test_pure_python_toolkit_import_does_not_load_native_extension():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "1"
    environment["PYTHONPATH"] = str(repository / "python")
    command = (
        "import holistic_motion as hm; "
        "from holistic_motion.kit import retargeting; "
        "from holistic_motion.trajectory import ToppraTrajectory; "
        "assert not hasattr(hm, 'Robot'); "
        "assert not hasattr(hm, 'os'); "
        "assert retargeting.RetargetingMode.DUAL_ARM.value == 'dual_arm'; "
        "assert retargeting.CuroboRetargetingSolver is not None; "
        "assert ToppraTrajectory is not None"
    )

    subprocess.run(
        [sys.executable, "-c", command],
        cwd=repository,
        env=environment,
        check=True,
    )


def test_numpy_trajectory_does_not_load_optional_torch():
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "python")
    command = """
import sys
from holistic_motion.trajectory import retime_path
retime_path([[0.0], [1.0]], [1.0], [2.0])
assert 'torch' not in sys.modules
sys.modules['torch'] = None
try:
    from holistic_motion.trajectory import retime_path_torch
except ImportError as error:
    assert 'differentiable' in str(error)
else:
    raise AssertionError('Torch import unexpectedly succeeded')
"""
    subprocess.run([sys.executable, "-c", command], env=environment, check=True)


def test_geometry_namespace_loads_only_when_accessed():
    environment = os.environ.copy()
    command = """
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
import holistic_motion as hm
assert 'holistic_motion.geometry' not in sys.modules
assert 'numpy' not in sys.modules
assert 'geometry' in dir(hm)
assert 'geometry' in hm.__all__
hm.runtime_info()
assert 'holistic_motion.geometry' not in sys.modules
assert 'numpy' not in sys.modules
with ThreadPoolExecutor(4) as workers:
    modules = list(workers.map(lambda _: hm.geometry, range(16)))
assert all(module is modules[0] for module in modules)
from holistic_motion import geometry
assert geometry is hm.geometry is importlib.import_module('holistic_motion.geometry')
assert callable(geometry.fit_spheres)
assert 'numpy' in sys.modules
assert not hm.runtime_info().collision_loaded
"""
    subprocess.run(
        [sys.executable, "-c", command],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_native_model_first_numeric_query_loads_numpy_without_geometry(tmp_path):
    import pytest
    from holistic_motion import runtime_info

    if not runtime_info().native_loaded:
        pytest.skip("requires the native core")
    urdf = tmp_path / "lazy_numpy.urdf"
    urdf.write_text(
        '<robot name="lazy_numpy"><link name="base"/><link name="tip"/>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="tip"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint></robot>'
    )
    command = """
import sys
import holistic_motion as hm
assert 'numpy' not in sys.modules
robot = hm.Robot(sys.argv[1])
assert robot.dof == 1
assert robot.kinematics.capabilities.dof == 1
assert 'numpy' not in sys.modules
pose = robot.kinematics.forward([0.3])
assert pose.shape == (4, 4) and abs(pose[0, 3] - 0.3) < 1e-14
assert 'numpy' in sys.modules
assert 'holistic_motion.geometry' not in sys.modules
assert not hm.runtime_info().collision_loaded
"""
    subprocess.run(
        [sys.executable, "-c", command, str(urdf)],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
