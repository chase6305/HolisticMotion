"""Pure toolkit installation must not need a C++ compiler or Conan runtime."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_cmake_toolkit_install_imports_without_native_extension(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is unavailable")
    repository = Path(__file__).resolve().parents[2]
    build = tmp_path / "configure"
    install = tmp_path / "install"
    environment = os.environ.copy()
    environment.pop("HOLISTICMOTION_PURE_PYTHON", None)
    environment.pop("CMAKE_TOOLCHAIN_FILE", None)
    subprocess.run(
        [
            cmake,
            "-S",
            str(repository),
            "-B",
            str(build),
            "-DHOLISTICMOTION_PYTHON_TOOLKIT_ONLY=ON",
            "-DCMAKE_CXX_COMPILER=/nonexistent/compiler-for-toolkit-test",
            "-DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_pinocchio=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_pybind11=ON",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [cmake, "--install", str(build), "--prefix", str(install)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (install / "holistic_motion" / "_build_config.py").is_file()
    assert not list(install.rglob("*.so"))
    assert not list(install.rglob("*.pyd"))
    environment["PYTHONPATH"] = str(install)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import holistic_motion as hm
from holistic_motion.trajectory import ToppraTrajectory
from holistic_motion.configuration import JointCoordinate, JointPositionMap
from holistic_motion.visualization.viser import configure_scene
assert hm.runtime_info().pure_python
assert hm.runtime_info().collision_compiled is False
assert not hm.runtime_info().native_loaded
assert not hasattr(hm, 'Robot')
assert 'pinocchio' not in sys.modules
assert 'torch' not in sys.modules
assert 'trimesh' not in sys.modules
assert 'viser' not in sys.modules
assert 'holistic_motion.visualization.viser.robot' not in sys.modules
assert callable(configure_scene)
assert ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0]).duration > 0
mapping = JointPositionMap(1, 1, [JointCoordinate('slide', 0, 1, 0, 1)])
assert mapping.configuration_with_joint_positions([0.0], [0.5])[0] == 0.5
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
