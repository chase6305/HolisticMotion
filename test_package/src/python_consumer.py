"""Import the installed Conan package through its generated run environment."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["HOLISTICMOTION_PURE_PYTHON"] = "0"

import numpy as np
import holistic_motion as hm
from holistic_motion.trajectory import ToppraTrajectory


prefix = Path(sys.argv[1]).resolve()
assert Path(hm.__file__).resolve().parent == prefix / "holistic_motion"
native = sys.modules["holistic_motion._holistic_motion"]
assert Path(native.__file__).resolve().parent == prefix / "holistic_motion"
info = hm.runtime_info()
assert info.native_loaded and not info.pure_python
assert info.collision_compiled == (sys.argv[2] == "1")
assert not info.collision_loaded
assert info.cuda_compiled == (sys.argv[3] == "1")
assert info.native_python_abi.startswith(sys.implementation.name)
if info.native_python_version:
    assert info.native_python_version.split(".")[:2] == [
        str(sys.version_info.major),
        str(sys.version_info.minor),
    ]
assert "pinocchio" not in sys.modules
assert "torch" not in sys.modules
with TemporaryDirectory(prefix="holistic-motion-python-consumer-") as folder:
    path = Path(folder) / "consumer.urdf"
    path.write_text(
        '<robot name="consumer"><link name="base"/><link name="tip"/>'
        '<joint name="turn" type="continuous"><parent link="base"/>'
        '<child link="tip"/><axis xyz="0 0 1.5"/>'
        '<limit velocity="1" effort="1"/></joint></robot>'
    )
    robot = hm.Robot(str(path))
    pose = robot.kinematics.forward([0.3])
    np.testing.assert_allclose(np.arctan2(pose[1, 0], pose[0, 0]), 0.3, atol=1e-14)
    assert robot.create_joint_position_map(["turn"]).joint_positions([0.3])[0] == 0.3
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0])
    assert trajectory.duration > 0.0
    np.testing.assert_allclose(trajectory.sample([trajectory.duration])[0], [[1.0]])
if info.collision_compiled:
    assert hm.CollisionModel is not None
    assert hm.runtime_info().collision_loaded
    collision = sys.modules["holistic_motion._holistic_motion_collision"]
    assert Path(collision.__file__).resolve().parent == prefix / "holistic_motion"
print(hm.runtime_info())
