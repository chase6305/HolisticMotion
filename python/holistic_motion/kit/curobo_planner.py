"""Optional NVIDIA cuRobo v2 motion planning with owned NumPy trajectories.

cuRobo, Torch and CUDA are loaded only when a planner is constructed. Robot
configuration and assets are supplied by the caller; nothing is downloaded.
This module is independent of HolisticMotion's native collision component.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from threading import Lock

import numpy as np

from .._validation import finite_scalar, real_array, validated_transform

__all__ = ["CuroboPlanner", "CuroboPlanningResult", "CuroboTrajectory"]


def _names(values):
    if isinstance(values, str):
        raise TypeError("joint_names must be a sequence of names")
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint_names must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("joint_names must be unique")
    return names


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    value = finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class CuroboTrajectory:
    """Owned sampled trajectory; rows follow ``time`` and ``joint_names``.

    Arrays are read-only snapshots, independent of the planner's next solve.
    No interpolation or continuous-time feasibility guarantee is implied.
    """

    joint_names: tuple
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray | None = None

    def __post_init__(self):
        names = _names(self.joint_names)
        times = real_array(self.time, name="time", copy=True)
        if (
            times.ndim != 1
            or times.size < 1
            or not np.isfinite(times).all()
            or times[0] != 0.0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("time must start at zero and increase strictly")
        times.setflags(write=False)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "time", times)
        for name in ("position", "velocity", "acceleration", "jerk"):
            value = getattr(self, name)
            if name == "jerk" and value is None:
                continue
            array = real_array(value, name=name, copy=True)
            if array.shape != (times.size, len(names)) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape [samples, joints]")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def duration(self) -> float:
        return float(self.time[-1])


@dataclass(frozen=True)
class CuroboPlanningResult:
    """A failed request has diagnostics and no consumable trajectory."""

    success: bool
    message: str
    trajectory: CuroboTrajectory | None = None


class CuroboPlanner:
    """Reusable, single-request NVIDIA cuRobo v2 planner.

    ``robot_config`` is an explicit local YAML path containing the caller's
    robot, URDF, collision spheres, locked joints and dynamics limits.
    ``joint_names`` may permute the complete active joint set; locked joints
    stay at their configured positions. Inputs are scalar positions, not full
    Pinocchio ``nq`` configurations. Planning starts and finishes at rest.

    ``scene`` accepts an explicit local scene path, a cuRobo scene dictionary,
    or a public ``curobo.scene.Scene`` instance. None means an empty world;
    self-collision checking remains enabled. Scene poses use the robot base
    frame, meters and ``wxyz`` quaternions. The backend always selects RRT and
    TimeOpt. Calls and updates on one instance are serialized.
    """

    def __init__(
        self,
        robot_config,
        *,
        scene=None,
        joint_names=None,
        device="cuda:0",
        interpolation_dt=0.01,
        collision_cache=None,
        use_cuda_graph=True,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
    ):
        config_path = Path(robot_config).expanduser().resolve(strict=True)
        if not config_path.is_file():
            raise ValueError("robot_config must be a local file")
        interpolation_dt = _positive(interpolation_dt, "interpolation_dt")
        self._position_tolerance = _positive(position_tolerance, "position_tolerance")
        self._orientation_tolerance = _positive(
            orientation_tolerance, "orientation_tolerance"
        )
        if not isinstance(use_cuda_graph, (bool, np.bool_)):
            raise TypeError("use_cuda_graph must be boolean")
        requested_names = None if joint_names is None else _names(joint_names)
        try:
            import torch
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            from curobo.scene import Scene
            from curobo.trajectory_timeopt import TrajectoryTimeOpt  # noqa: F401
            from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose
        except ImportError as error:
            raise ImportError(
                "CuroboPlanner requires NVIDIA cuRobo v2 with TimeOpt, Torch "
                "and a compatible CUDA environment. Install NVlabs/curobo "
                "separately; cuRobo v1 is not supported."
            ) from error
        selected_device = torch.device(device)
        if selected_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CuroboPlanner requires an available CUDA device")
        self._torch = torch
        self._JointState = JointState
        self._GoalToolPose = GoalToolPose
        self._Pose = Pose
        self._Scene = Scene
        self._lock = Lock()
        self._planner = None
        self._world_valid = True
        config = MotionPlannerCfg.create(
            robot=str(config_path),
            scene_model=self._scene(scene),
            collision_cache=deepcopy(collision_cache),
            device_cfg=DeviceCfg(device=selected_device, dtype=torch.float32),
            graph_planner_algorithm="rrt",
            enable_timeopt=True,
            self_collision_check=True,
            interpolation_dt=interpolation_dt,
            use_cuda_graph=bool(use_cuda_graph),
            position_tolerance=self._position_tolerance,
            orientation_tolerance=self._orientation_tolerance,
        )
        planner = MotionPlanner(config)
        try:
            self._backend_names = _names(planner.joint_names)
            self._tool_frames = tuple(planner.tool_frames)
            self._joint_names = requested_names or self._backend_names
            if set(self._joint_names) != set(self._backend_names):
                raise ValueError(
                    "joint_names must match the complete active cuRobo joint set"
                )
            self._to_backend = [
                self._joint_names.index(name) for name in self._backend_names
            ]
            limits = planner.kinematics.get_joint_limits()
            limit_names = _names(limits.joint_names)
            if set(limit_names) != set(self._backend_names):
                raise ValueError("cuRobo joint limits do not match its active joints")
            order = [limit_names.index(name) for name in self._joint_names]
            self._limits = self._numpy(limits.position)[:, order]
            if (
                self._limits.shape != (2, len(self._joint_names))
                or not np.isfinite(self._limits).all()
                or np.any(self._limits[0] >= self._limits[1])
            ):
                raise ValueError(
                    "cuRobo must provide finite ordered scalar joint limits"
                )
            self._device = planner.default_joint_state.position.device
        except BaseException:
            planner.destroy()
            raise
        self._planner = planner

    @property
    def joint_names(self) -> tuple:
        return self._joint_names

    @property
    def tool_frames(self) -> tuple:
        return self._tool_frames

    @property
    def lower_limits(self) -> np.ndarray:
        return self._limits[0].copy()

    @property
    def upper_limits(self) -> np.ndarray:
        return self._limits[1].copy()

    def _scene(self, scene):
        if scene is None:
            return self._Scene()
        if isinstance(scene, (str, Path)):
            import yaml

            path = Path(scene).expanduser().resolve(strict=True)
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(scene, Mapping):
                raise TypeError("scene file must contain a scene dictionary")
        if isinstance(scene, Mapping):
            return self._Scene.create(deepcopy(dict(scene)))
        if isinstance(scene, self._Scene):
            return deepcopy(scene)
        raise TypeError("scene must be a local path, dictionary or cuRobo Scene")

    @staticmethod
    def _numpy(tensor):
        return tensor.detach().cpu().numpy()

    def _open(self):
        if self._planner is None:
            raise RuntimeError("CuroboPlanner is closed")
        return self._planner

    def _planning_backend(self):
        planner = self._open()
        if not self._world_valid:
            raise RuntimeError(
                "A failed world update requires a successful update_world before planning"
            )
        return planner

    def warmup(self, num_iterations=2):
        """Compile and capture representative pose and RRT branches once."""
        num_iterations = _count(num_iterations, "num_iterations")
        with self._lock:
            self._planning_backend().warmup(
                enable_graph=True, num_warmup_iterations=num_iterations
            )

    def update_world(self, scene):
        """Replace all world obstacles, retaining allocated backend capacities."""
        scene = self._scene(scene)
        with self._lock:
            planner = self._open()
            self._world_valid = False
            try:
                planner.update_world(scene)
            finally:
                planner.reset_seed()
            self._world_valid = True

    def _position(self, value, name):
        position = real_array(value, name=name, copy=True)
        if (
            position.shape != (len(self._joint_names),)
            or not np.isfinite(position).all()
        ):
            raise ValueError(f"{name} must be a finite scalar joint vector")
        if np.any(position < self._limits[0]) or np.any(position > self._limits[1]):
            raise ValueError(f"{name} is outside the configured joint limits")
        with np.errstate(over="ignore", invalid="ignore"):
            converted = position.astype(np.float32)
        if not np.isfinite(converted).all() or np.any(
            np.abs(converted - position) > 1e-6
        ):
            raise ValueError(f"{name} loses more than 1e-6 during float32 conversion")
        if np.any(converted < self._limits[0]) or np.any(converted > self._limits[1]):
            raise ValueError(f"{name} rounds outside the configured joint limits")
        tensor = self._torch.as_tensor(
            converted[self._to_backend][None, :],
            dtype=self._torch.float32,
            device=self._device,
        )
        return position, self._JointState.from_position(
            tensor, joint_names=list(self._backend_names)
        )

    def plan(self, start, goal, *, max_attempts=4, endpoint_tolerance=1e-3):
        """Plan a rest-to-rest joint motion, preserving supplied scalar order.

        Invalid inputs raise before solving. Upstream planning failure or
        rejected output returns ``success=False`` and no trajectory. This is
        a sampled result; it is not a continuous collision/torque certificate.
        """
        max_attempts = _count(max_attempts, "max_attempts")
        endpoint_tolerance = _positive(endpoint_tolerance, "endpoint_tolerance")
        with self._lock:
            planner = self._planning_backend()
            start_position, current_state = self._position(start, "start")
            goal_position, goal_state = self._position(goal, "goal")
            result = planner.plan_cspace(
                goal_state,
                current_state,
                max_attempts=max_attempts,
                enable_graph_attempt=1,
            )
            return self._planning_result(
                result, start_position, goal_position, endpoint_tolerance
            )

    def forward_kinematics(self, position):
        """Return owned base-frame 4x4 transforms for the configured tools."""
        with self._lock:
            planner = self._open()
            _, state = self._position(position, "position")
            return self._tool_transforms(planner, state)

    def _tool_transforms(self, planner, state):
        poses = planner.compute_kinematics(state).tool_poses
        return {
            name: self._numpy(poses.get_link_pose(name).get_matrix())
            .reshape(4, 4)
            .copy()
            for name in self._tool_frames
        }

    def plan_pose(self, start, tool_poses, *, max_attempts=4):
        """Plan from rest to named base-frame tool targets given as 4x4 matrices.

        Supply every configured tool. Position/orientation tolerances come
        from construction and are checked again with FK on the returned end
        configuration. No external quaternion convention is assumed.
        """
        max_attempts = _count(max_attempts, "max_attempts")
        if not isinstance(tool_poses, Mapping) or set(tool_poses) != set(
            self._tool_frames
        ):
            raise ValueError(
                "tool_poses must map every configured tool to a 4x4 transform"
            )
        targets = {
            name: validated_transform(tool_poses[name], name=name)
            for name in self._tool_frames
        }
        with self._lock:
            planner = self._planning_backend()
            start_position, current_state = self._position(start, "start")
            poses = {}
            for name, transform in targets.items():
                converted = transform.astype(np.float32)
                if (
                    not np.isfinite(converted).all()
                    or np.max(np.abs(converted - transform)) > 1e-6
                ):
                    raise ValueError(
                        "tool pose loses more than 1e-6 during float32 conversion"
                    )
                poses[name] = self._Pose.from_matrix(
                    self._torch.as_tensor(
                        converted, device=self._device, dtype=self._torch.float32
                    )
                )
            goal = self._GoalToolPose.from_poses(poses)
            result = self._planning_result(
                planner.plan_pose(goal, current_state, max_attempts=max_attempts),
                start_position,
            )
            if not result.success:
                return result
            _, end = self._position(
                result.trajectory.position[-1], "trajectory endpoint"
            )
            actual = self._tool_transforms(planner, end)
            for name, target in targets.items():
                relative = actual[name][:3, :3].T @ target[:3, :3]
                sine = 0.5 * np.linalg.norm(
                    [
                        relative[2, 1] - relative[1, 2],
                        relative[0, 2] - relative[2, 0],
                        relative[1, 0] - relative[0, 1],
                    ]
                )
                angle = np.arctan2(sine, (np.trace(relative) - 1.0) * 0.5)
                distance = np.linalg.norm(actual[name][:3, 3] - target[:3, 3])
                if (
                    not np.isfinite(distance + angle)
                    or distance > self._position_tolerance
                    or angle > self._orientation_tolerance
                ):
                    return CuroboPlanningResult(
                        False, f"cuRobo trajectory misses tool target: {name}"
                    )
            return result

    def _planning_result(self, result, start, goal=None, endpoint_tolerance=1e-3):
        if result is None:
            return CuroboPlanningResult(False, "cuRobo returned no solution")
        success = result.success
        if success is None or success.numel() != 1 or success.dtype != self._torch.bool:
            return CuroboPlanningResult(
                False, "cuRobo returned an invalid success mask"
            )
        if not bool(success.item()):
            return CuroboPlanningResult(
                False, str(getattr(result, "status", None) or "cuRobo planning failed")
            )
        try:
            trajectory = self._trajectory(result.get_interpolated_plan())
            if np.max(np.abs(trajectory.position[0] - start)) > endpoint_tolerance or (
                goal is not None
                and np.max(np.abs(trajectory.position[-1] - goal)) > endpoint_tolerance
            ):
                raise ValueError("cuRobo trajectory endpoints do not match the request")
            if np.any(trajectory.position < self._limits[0] - 1e-6) or np.any(
                trajectory.position > self._limits[1] + 1e-6
            ):
                raise ValueError("cuRobo trajectory exceeds joint position limits")
        except (TypeError, ValueError) as error:
            return CuroboPlanningResult(False, str(error))
        return CuroboPlanningResult(True, "cuRobo planning succeeded", trajectory)

    def _trajectory(self, state):
        if state is None or state.dt is None:
            raise ValueError("cuRobo returned no timed trajectory")
        names = _names(state.joint_names)
        if not set(self._joint_names).issubset(names):
            raise ValueError("cuRobo trajectory is missing requested joints")
        order = [names.index(name) for name in self._joint_names]
        arrays = {}
        for name in ("position", "velocity", "acceleration", "jerk"):
            value = getattr(state, name, None)
            if name == "jerk" and value is None:
                arrays[name] = None
                continue
            if value is None:
                raise ValueError(f"cuRobo trajectory is missing {name}")
            array = self._numpy(value)
            if array.ndim in (3, 4) and all(size == 1 for size in array.shape[:-2]):
                array = array.reshape(array.shape[-2:])
            if array.ndim != 2 or array.shape[1] != len(names):
                raise ValueError(f"cuRobo trajectory has an invalid {name} shape")
            arrays[name] = array[:, order]
        dt = self._numpy(state.dt)
        if dt.size != 1:
            raise ValueError("cuRobo trajectory must have one interpolation timestep")
        dt = _positive(dt.item(), "cuRobo trajectory dt")
        times = np.arange(arrays["position"].shape[0], dtype=float) * dt
        return CuroboTrajectory(self._joint_names, times, **arrays)

    def close(self):
        """Release backend resources; already returned trajectories stay valid."""
        with self._lock:
            if self._planner is not None:
                planner = self._planner
                self._planner = None
                planner.destroy()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_args):
        self.close()
