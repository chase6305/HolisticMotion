"""Optional NVIDIA cuRobo v2 motion planning with owned NumPy trajectories.

cuRobo, Torch and CUDA are loaded only when a planner is constructed. Robot
configuration and assets are supplied by the caller; nothing is downloaded.
This module is independent of HolisticMotion's native collision component.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import wraps
from math import prod
from numbers import Integral
from pathlib import Path
from threading import RLock
from weakref import ref

import numpy as np

from .._validation import finite_scalar, real_array, validated_transform

__all__ = ["CuroboPlanner", "CuroboPlanningResult", "CuroboTrajectory"]

_POSITION_LIMIT_TOLERANCE = 1e-6
_LIVE_PLANNERS = []
_BACKEND_LOCK = RLock()


def _serialized_backend_call(method):
    # Upstream captures in CUDA's global mode. Allocating on another planner's
    # thread during a lazy capture is forbidden, even on an independent stream.
    @wraps(method)
    def call(*args, **kwargs):
        with _BACKEND_LOCK:
            return method(*args, **kwargs)

    return call


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


def _count(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _float32(value, name):
    value = np.asarray(value, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        converted = value.astype(np.float32)
        if not np.isfinite(converted).all() or np.any(np.abs(converted - value) > 1e-6):
            raise ValueError(f"{name} loses more than 1e-6 during float32 conversion")
    return converted


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
    TimeOpt. Backend operations are serialized within each process.
    """

    @_serialized_backend_call
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
            from curobo.scene import Capsule, Scene
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
        self._Capsule = Capsule
        self._planner = None
        self._world_valid = True
        self._robot_update_failed = None
        import yaml

        robot = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(robot, Mapping):
            raise TypeError("robot_config must contain a robot dictionary")
        robot = robot.get("robot_cfg", robot)
        if not isinstance(robot, Mapping):
            raise TypeError("robot_cfg must be a robot dictionary")
        robot = dict(robot)
        # The upstream example YAML disables dynamics. Enable them in every
        # planning stage so torque checks and later payload updates agree.
        robot["load_dynamics"] = True
        with torch.inference_mode(False), torch.enable_grad():
            device_cfg = DeviceCfg(device=selected_device, dtype=torch.float32)
            self._device = device_cfg.device
            config = MotionPlannerCfg.create(
                robot=robot,
                scene_model=self._scene(scene),
                collision_cache=deepcopy(collision_cache),
                device_cfg=device_cfg,
                graph_planner_algorithm="rrt",
                enable_timeopt=True,
                self_collision_check=True,
                interpolation_dt=interpolation_dt,
                use_cuda_graph=bool(use_cuda_graph),
                position_tolerance=self._position_tolerance,
                orientation_tolerance=self._orientation_tolerance,
            )
            # The qualified upstream LM seed executor retains a grad-enabled input
            # on each replay. Use its public multi-seed IK configuration so this
            # long-lived planner does not accumulate those CUDA allocations.
            config.ik_solver_config.use_lm_seed = False
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
            self._use_cuda_graph = bool(use_cuda_graph)
            self._collision_layout = tuple(
                (type(data).__name__, data.max_n)
                for data in planner.scene_collision_checker.data.get_valid_data()
            )
            # New Warp generic specializations can unload modules referenced
            # by another planner's PyTorch captures. Reject incompatible live
            # layouts before the new planner launches collision kernels.
            live = []
            for reference in _LIVE_PLANNERS:
                other = reference()
                if other is None:
                    continue
                live.append(reference)
                if (
                    self._collision_layout
                    and other._collision_layout
                    and self._collision_layout != other._collision_layout
                    and (self._use_cuda_graph or other._use_cuda_graph)
                ):
                    raise RuntimeError(
                        "Live cuRobo CUDA graph planners must share a collision cache layout. "
                        "Use matching caches, separate processes, or construct all planners "
                        "with use_cuda_graph=False."
                    )
            self._torch.cuda.synchronize(self._device)
            _LIVE_PLANNERS[:] = [*live, ref(self)]
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
        supported = ("cuboid", "sphere", "capsule", "cylinder", "mesh", "voxel")
        if scene is None:
            return self._Scene()
        if isinstance(scene, (str, Path)):
            import yaml

            path = Path(scene).expanduser().resolve(strict=True)
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(scene, Mapping):
                raise TypeError("scene file must contain a scene dictionary")
        if isinstance(scene, Mapping):
            unknown = set(scene) - set(supported)
            if unknown:
                raise ValueError(f"unsupported scene obstacle types: {list(unknown)}")
        elif not isinstance(scene, self._Scene):
            raise TypeError("scene must be a local path, dictionary or cuRobo Scene")
        scene = self._copy_scene(scene)
        obstacles = list(scene)
        typed = [obstacle for kind in supported for obstacle in getattr(scene, kind)]
        if {id(value) for value in obstacles} != {id(value) for value in typed}:
            raise ValueError("Scene objects must match its typed obstacle lists")
        names = set()
        for obstacle in obstacles:
            name = obstacle.name
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("scene obstacle names must be nonempty and unique")
            names.add(name)
            pose = getattr(obstacle, "pose", None)
            if pose is None:
                raise ValueError(f"{name} requires an explicit [xyz, wxyz] pose")
            pose = real_array(pose, name=f"{name} pose")
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError(f"{name} pose must be finite [xyz, wxyz]")
            if abs(np.linalg.norm(pose[3:]) - 1.0) > 1e-5:
                raise ValueError(f"{name} pose quaternion must have unit norm")
            _float32(pose, f"{name} pose")
            for field in ("dims", "base", "tip", "scale"):
                value = getattr(obstacle, field, None)
                if value is None:
                    continue
                value = real_array(value, name=f"{name} {field}")
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(
                        f"{name} {field} must be a finite vector of size 3"
                    )
                if field == "dims" and np.any(value <= 0.0):
                    raise ValueError(f"{name} dimensions must be positive")
                if field == "scale" and np.any(value == 0.0):
                    raise ValueError(f"{name} scale must be nonzero")
                converted = _float32(value, f"{name} {field}")
                if field in ("dims", "scale") and np.any(converted == 0.0):
                    raise ValueError(f"{name} {field} rounds to zero in float32")
            for field in ("radius", "height", "voxel_size"):
                value = getattr(obstacle, field, None)
                if value is not None:
                    value = _positive(value, f"{name} {field}")
                    if _float32(value, f"{name} {field}") <= 0.0:
                        raise ValueError(f"{name} {field} rounds to zero in float32")
        for grid in scene.voxel:
            shape = grid.get_grid_shape()[0]
            if any(size < 1 for size in shape):
                raise ValueError(
                    f"{grid.name} grid shape must have at least one cell per axis"
                )
            if grid.feature_tensor.numel() != prod(shape):
                raise ValueError(
                    f"{grid.name} ESDF feature count does not match grid shape {shape}"
                )
        if scene.sphere:
            # The upstream SceneData loader omits Sphere objects. A capsule
            # with coincident endpoints is an exact analytic sphere, avoiding
            # both silent obstacle omission and mesh approximation.
            capsules = list(scene.capsule)
            capsules.extend(
                self._Capsule(name=ball.name, pose=ball.pose, radius=ball.radius)
                for ball in scene.sphere
            )
            scene = self._Scene(
                cuboid=scene.cuboid,
                capsule=capsules,
                cylinder=scene.cylinder,
                mesh=scene.mesh,
                voxel=scene.voxel,
            )
        return scene

    def _copy_scene(self, scene):
        if isinstance(scene, Mapping):
            grids = (scene.get("voxel") or {}).items()
        else:
            grids = ((grid.name, grid) for grid in scene.voxel)
        memo = {}
        for name, grid in grids:
            features = (
                grid.get("feature_tensor")
                if isinstance(grid, Mapping)
                else grid.feature_tensor
            )
            if not self._torch.is_tensor(features) or not features.is_floating_point():
                raise TypeError(f"{name} ESDF features must be a floating tensor")
            if features.numel() == 0:
                raise ValueError(f"{name} ESDF features must be nonempty")
            minimum, maximum = (bound.item() for bound in self._torch.aminmax(features))
            limit = self._torch.finfo(self._torch.float16).max
            if not (-limit <= minimum <= maximum <= limit):
                raise ValueError(
                    f"{name} ESDF features must stay finite in float16 storage"
                )
            # Non-leaf tensors cannot be deepcopied. Copy their values directly
            # into the contiguous float16 storage required by the Warp loader.
            with self._torch.inference_mode(False), self._torch.no_grad():
                owned_features = features.detach().to(
                    device=self._device,
                    dtype=self._torch.float16,
                    copy=True,
                    memory_format=self._torch.contiguous_format,
                )
            if isinstance(grid, Mapping):
                metadata = dict(grid)
                metadata.pop("feature_tensor", None)
                owned_grid = deepcopy(metadata)
                owned_grid["feature_tensor"] = owned_features
            else:
                owned_grid = copy(grid)
                owned_grid.feature_tensor = None
                owned_grid = deepcopy(owned_grid)
                owned_grid.feature_tensor = owned_features
                owned_grid.feature_dtype = owned_features.dtype
            memo[id(grid)] = owned_grid
        copied = deepcopy(dict(scene) if isinstance(scene, Mapping) else scene, memo)
        if isinstance(copied, Mapping):
            copied = self._Scene.create(copied)
        return copied

    @staticmethod
    def _numpy(tensor):
        return tensor.detach().cpu().numpy()

    def _open(self):
        if self._planner is None:
            raise RuntimeError("CuroboPlanner is closed")
        return self._planner

    def _planning_backend(self):
        planner = self._open()
        if self._robot_update_failed is not None:
            raise RuntimeError(
                f"A failed {self._robot_update_failed} update requires recreating CuroboPlanner"
            )
        if not self._world_valid:
            raise RuntimeError(
                "A failed world update requires a successful update_world before planning"
            )
        return planner

    @_serialized_backend_call
    def warmup(self, num_iterations=2):
        """Compile and capture representative pose and RRT branches once."""
        num_iterations = _count(num_iterations, "num_iterations")
        with self._torch.inference_mode(False), self._torch.enable_grad():
            self._planning_backend().warmup(
                enable_graph=True, num_warmup_iterations=num_iterations
            )
            self._torch.cuda.synchronize(self._device)

    @_serialized_backend_call
    def update_world(self, scene):
        """Replace all world obstacles, retaining allocated backend capacities."""
        scene = self._scene(scene)
        with self._torch.inference_mode(False), self._torch.enable_grad():
            planner = self._open()
            self._world_valid = False
            try:
                meshes = planner.scene_collision_checker.data.meshes
                if meshes is not None and meshes.get_cached_mesh_names():
                    # The upstream cache keys geometry only by obstacle name
                    # and otherwise retains meshes removed from the world.
                    self._torch.cuda.synchronize(self._device)
                    meshes.clear(clear_warp_cache=True)
                planner.update_world(scene)
            finally:
                planner.reset_seed()
                self._torch.cuda.synchronize(self._device)
            self._world_valid = True

    @_serialized_backend_call
    def update_obstacle_pose(self, obstacle_name, transform):
        """Move an existing world obstacle without rebuilding its geometry.

        ``transform`` is a base-frame 4x4 matrix, in meters. RRT state is
        reset, while mesh geometry, storage capacity and captures are retained.
        A backend failure blocks planning until update_world succeeds.
        """
        if not isinstance(obstacle_name, str) or not obstacle_name:
            raise ValueError("obstacle_name must be a nonempty string")
        transform = _float32(
            validated_transform(transform, name="obstacle transform"),
            "obstacle transform",
        )
        with self._torch.inference_mode(False), self._torch.enable_grad():
            planner = self._planning_backend()
            checker = planner.scene_collision_checker
            if not checker.check_obstacle_exists(obstacle_name):
                raise ValueError(f"unknown cuRobo obstacle: {obstacle_name}")
            pose = self._Pose.from_matrix(
                self._torch.as_tensor(transform, device=self._device)
            )
            self._world_valid = False
            try:
                checker.update_obstacle_pose(obstacle_name, pose)
            finally:
                planner.reset_seed()
                self._torch.cuda.synchronize(self._device)
            self._world_valid = True

    @_serialized_backend_call
    def update_link_inertial(self, link_name, *, mass, com, inertia):
        """Replace a link's complete inertial properties and reset RRT state.

        Mass is in kg, COM is [3] in the link frame, and inertia about the COM
        is [ixx, iyy, izz, ixy, ixz, iyz] in kg*m^2, expressed in link axes.
        This updates dynamics only; it does not attach collision geometry.
        A backend update failure requires closing and recreating this instance.
        """
        if not isinstance(link_name, str) or not link_name:
            raise ValueError("link_name must be a nonempty string")
        if isinstance(mass, (bool, np.bool_)):
            raise TypeError("mass must be a real number")
        mass = finite_scalar(mass, "mass")
        if mass < 0:
            raise ValueError("mass must be non-negative")
        com = real_array(com, name="com", copy=True)
        inertia = real_array(inertia, name="inertia", copy=True)
        if com.shape != (3,) or not np.isfinite(com).all():
            raise ValueError("com must be a finite vector of size 3")
        if inertia.shape != (6,) or not np.isfinite(inertia).all():
            raise ValueError("inertia must be a finite vector of size 6")
        xx, yy, zz, xy, xz, yz = inertia
        matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        moments = np.linalg.eigvalsh(matrix)
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(moments))))
        if moments[0] < -tolerance or moments[2] > moments[0] + moments[1] + tolerance:
            raise ValueError(
                "inertia must be positive semidefinite and satisfy the moment triangle inequality"
            )
        properties = np.concatenate(([mass], com, inertia))
        converted = _float32(properties, "inertial properties")
        with self._torch.inference_mode(False), self._torch.enable_grad():
            planner = self._planning_backend()
            if (
                link_name
                not in planner.kinematics.kinematics_config.link_name_to_idx_map
            ):
                raise ValueError(f"unknown cuRobo link: {link_name}")
            com_tensor = self._torch.as_tensor(converted[1:4], device=self._device)
            inertia_tensor = self._torch.as_tensor(converted[4:], device=self._device)
            try:
                planner.update_link_inertial(
                    link_name, float(converted[0]), com_tensor, inertia_tensor
                )
                planner.reset_seed()
                self._torch.cuda.synchronize(self._device)
            except BaseException:
                self._robot_update_failed = "inertial"
                raise

    @_serialized_backend_call
    def update_link_collision_spheres(self, link_name, spheres):
        """Replace a link's collision spheres, or restore its configuration.

        ``spheres`` contains link-frame [x, y, z, radius] rows in meters.
        Radii must be positive and the link must have enough preallocated
        sphere slots. Unused slots are disabled. Passing None restores all
        original spheres from the robot configuration. Inertial properties
        and world obstacles are unchanged; update them separately if needed.
        A backend update failure requires recreating this instance.
        """
        if not isinstance(link_name, str) or not link_name:
            raise ValueError("link_name must be a nonempty string")
        if spheres is not None:
            spheres = real_array(spheres, name="spheres", copy=True)
            if (
                spheres.ndim != 2
                or spheres.shape[1] != 4
                or spheres.shape[0] == 0
                or not np.isfinite(spheres).all()
                or np.any(spheres[:, 3] <= 0.0)
            ):
                raise ValueError(
                    "spheres must be finite [N, 4] rows with positive radii"
                )
            converted = _float32(spheres, "spheres")
            if np.any(converted[:, 3] <= 0.0):
                raise ValueError("sphere radii round to zero in float32")
        with self._torch.inference_mode(False), self._torch.enable_grad():
            planner = self._planning_backend()
            params = planner.attachment_manager.kinematics_params
            if link_name not in params.link_name_to_idx_map:
                raise ValueError(f"unknown cuRobo link: {link_name}")
            capacity = params.get_number_of_spheres(link_name)
            if capacity == 0:
                raise ValueError(
                    f"cuRobo link has no collision sphere slots: {link_name}"
                )
            if spheres is not None:
                if len(spheres) > capacity:
                    raise ValueError(
                        f"{link_name} has {capacity} sphere slots, requested {len(spheres)}"
                    )
                padded = np.zeros((capacity, 4), dtype=np.float32)
                padded[:, 3] = -100.0
                padded[: len(spheres)] = converted
                tensor = self._torch.as_tensor(padded, device=self._device)
            try:
                if spheres is None:
                    params.reset_link_spheres(link_name)
                else:
                    params.update_link_spheres(link_name, tensor)
                planner.reset_seed()
                self._torch.cuda.synchronize(self._device)
            except BaseException:
                self._robot_update_failed = "collision sphere"
                raise

    def _position(self, value, name, limit_tolerance=0.0):
        position = real_array(value, name=name, copy=True)
        if (
            position.shape != (len(self._joint_names),)
            or not np.isfinite(position).all()
        ):
            raise ValueError(f"{name} must be a finite scalar joint vector")
        if np.any(position < self._limits[0] - limit_tolerance) or np.any(
            position > self._limits[1] + limit_tolerance
        ):
            raise ValueError(f"{name} is outside the configured joint limits")
        converted = _float32(position, name)
        if np.any(converted < self._limits[0] - limit_tolerance) or np.any(
            converted > self._limits[1] + limit_tolerance
        ):
            raise ValueError(f"{name} rounds outside the configured joint limits")
        return position, self._joint_state(converted)

    def _joint_state(self, positions):
        tensor = self._torch.as_tensor(
            np.asarray(positions, dtype=np.float32)[self._to_backend][None, :],
            dtype=self._torch.float32,
            device=self._device,
        )
        return self._JointState.from_position(
            tensor, joint_names=list(self._backend_names)
        )

    @_serialized_backend_call
    def plan(
        self,
        start,
        goal,
        *,
        max_attempts=4,
        enable_graph_attempt=1,
        endpoint_tolerance=1e-3,
    ):
        """Plan a rest-to-rest joint motion, preserving supplied scalar order.

        Invalid inputs raise before solving. Upstream planning failure or
        rejected output returns ``success=False`` and no trajectory. This is
        a sampled result; it is not a continuous collision/torque certificate.
        ``enable_graph_attempt=0`` uses RRT on the first attempt; the default
        first tries direct optimization, then enables RRT on the second attempt.
        """
        max_attempts = _count(max_attempts, "max_attempts")
        enable_graph_attempt = _count(
            enable_graph_attempt, "enable_graph_attempt", minimum=0
        )
        endpoint_tolerance = _positive(endpoint_tolerance, "endpoint_tolerance")
        with self._torch.inference_mode(False), self._torch.enable_grad():
            planner = self._planning_backend()
            start_position, current_state = self._position(start, "start")
            goal_position, goal_state = self._position(goal, "goal")
            result = planner.plan_cspace(
                goal_state,
                current_state,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
            )
            return self._planning_result(
                result, start_position, goal_position, endpoint_tolerance
            )

    @_serialized_backend_call
    def forward_kinematics(self, position):
        """Return owned base-frame 4x4 transforms for the configured tools."""
        with self._torch.inference_mode(False), self._torch.enable_grad():
            planner = self._open()
            # Accept the same float32 bound roundoff as a returned trajectory,
            # so callers can evaluate FK on any accepted result sample.
            _, state = self._position(position, "position", _POSITION_LIMIT_TOLERANCE)
            return self._tool_transforms(planner, state)

    def _tool_transforms(self, planner, state):
        poses = planner.compute_kinematics(state).tool_poses
        return {
            name: validated_transform(
                self._numpy(poses.get_link_pose(name).get_matrix()).reshape(4, 4),
                name=f"cuRobo tool {name}",
            )
            for name in self._tool_frames
        }

    @_serialized_backend_call
    def plan_pose(self, start, tool_poses, *, max_attempts=4, enable_graph_attempt=1):
        """Plan from rest to named base-frame tool targets given as 4x4 matrices.

        Supply every configured tool. Position/orientation tolerances come
        from construction and are checked again with FK on the returned end
        configuration. No external quaternion convention is assumed.
        """
        max_attempts = _count(max_attempts, "max_attempts")
        enable_graph_attempt = _count(
            enable_graph_attempt, "enable_graph_attempt", minimum=0
        )
        if (
            not self._tool_frames
            or not isinstance(tool_poses, Mapping)
            or set(tool_poses) != set(self._tool_frames)
        ):
            raise ValueError(
                "tool_poses must map every configured tool to a 4x4 transform"
            )
        targets = {
            name: validated_transform(tool_poses[name], name=name)
            for name in self._tool_frames
        }
        with self._torch.inference_mode(False), self._torch.enable_grad():
            planner = self._planning_backend()
            start_position, current_state = self._position(start, "start")
            poses = {}
            for name, transform in targets.items():
                converted = _float32(transform, "tool pose")
                poses[name] = self._Pose.from_matrix(
                    self._torch.as_tensor(
                        converted, device=self._device, dtype=self._torch.float32
                    )
                )
            goal = self._GoalToolPose.from_poses(poses)
            result = self._planning_result(
                planner.plan_pose(
                    goal,
                    current_state,
                    max_attempts=max_attempts,
                    enable_graph_attempt=enable_graph_attempt,
                ),
                start_position,
            )
            if not result.success:
                return result
            end = self._joint_state(result.trajectory.position[-1])
            try:
                actual = self._tool_transforms(planner, end)
            except (TypeError, ValueError) as error:
                return CuroboPlanningResult(False, str(error))
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
            message = str(getattr(result, "status", None) or "cuRobo planning failed")
            debug = getattr(result, "debug_info", None)
            validation = debug.get("validation") if isinstance(debug, Mapping) else None
            if isinstance(validation, Mapping):
                failed = []
                for key, label in (
                    ("collision_valid", "collision"),
                    ("limits_and_dynamics_valid", "joint limits/dynamics"),
                    ("path_valid", "path/pose"),
                ):
                    mask = validation.get(key)
                    if (
                        self._torch.is_tensor(mask)
                        and mask.dtype == self._torch.bool
                        and mask.numel()
                        and not bool(mask.any().item())
                    ):
                        failed.append(label)
                if failed:
                    message += " (failed checks: " + ", ".join(failed) + ")"
            return CuroboPlanningResult(False, message)
        try:
            trajectory = self._trajectory(result.get_interpolated_plan())
            if np.max(np.abs(trajectory.position[0] - start)) > endpoint_tolerance or (
                goal is not None
                and np.max(np.abs(trajectory.position[-1] - goal)) > endpoint_tolerance
            ):
                raise ValueError("cuRobo trajectory endpoints do not match the request")
            if np.any(
                trajectory.position < self._limits[0] - _POSITION_LIMIT_TOLERANCE
            ) or np.any(
                trajectory.position > self._limits[1] + _POSITION_LIMIT_TOLERANCE
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
        dt = finite_scalar(dt.item(), "cuRobo trajectory dt")
        if dt < 0.0:
            raise ValueError("cuRobo trajectory dt must be non-negative")
        if dt == 0.0:
            position = arrays["position"]
            if (
                position.shape[0] == 0
                or not np.all(position == position[0])
                or any(
                    value is not None
                    and (value.shape != position.shape or np.any(value != 0.0))
                    for name, value in arrays.items()
                    if name != "position"
                )
            ):
                raise ValueError("a zero timestep requires a stationary trajectory")
            arrays = {
                name: None if value is None else value[:1]
                for name, value in arrays.items()
            }
        times = np.arange(arrays["position"].shape[0], dtype=float) * dt
        return CuroboTrajectory(self._joint_names, times, **arrays)

    @_serialized_backend_call
    def close(self):
        """Release backend resources; already returned trajectories stay valid."""
        if self._planner is not None:
            planner = self._planner
            self._planner = None
            try:
                planner.destroy()
                self._torch.cuda.synchronize(self._device)
            finally:
                _LIVE_PLANNERS[:] = [
                    reference
                    for reference in _LIVE_PLANNERS
                    if (other := reference()) is not None and other is not self
                ]

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_args):
        self.close()
