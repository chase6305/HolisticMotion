"""Generic Pinocchio inverse-kinematics retargeting solver.

The solver is adapted from dexe_teleoperate's pose retargeting solver.  This
version deliberately excludes ROS, robot-specific YAML, and controller state;
all frame and joint mappings are supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Optional, Union

import numpy as np

from ..._validation import validated_transform
from .modes import RetargetingMode, RetargetingModeManager


def _readonly_array(value, *, shape=None, name="array") -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _positive_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _load_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as error:
        raise ImportError(
            "PinocchioRetargetingSolver requires Python Pinocchio; install "
            "HolisticMotion with the 'retargeting' extra"
        ) from error
    return pin


@dataclass(frozen=True)
class RetargetingTarget:
    """A named 4x4 world-frame target pose and its relative weight."""

    pose: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        pose = validated_transform(self.pose, name="target pose", readonly=True)
        weight = _positive_float(self.weight, "target weight")
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class RetargetingResult:
    configuration: np.ndarray
    success: bool
    iterations: int
    residual: float
    solve_ms: float
    mode: RetargetingMode
    position_residual: float = float("nan")
    orientation_residual: float = float("nan")
    target_residuals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    termination_reason: str = "legacy"
    accepted_steps: int = 0
    limit_hits: int = 0
    collision_cost: float = 0.0
    collision_evaluations: int = 0
    collision_gradient_evaluations: int = 0
    objective: float = float("nan")
    center_of_mass_residual: float = float("nan")
    support_polygon_violation: float = 0.0
    zmp_residual: float = float("nan")

    def __post_init__(self) -> None:
        configuration = _readonly_array(
            self.configuration, name="result configuration"
        ).reshape(-1)
        if not isinstance(self.mode, RetargetingMode):
            raise TypeError("result mode must be RetargetingMode")
        if not isinstance(self.success, (bool, np.bool_)):
            raise TypeError("result success must be boolean")
        if not isinstance(self.target_residuals, Mapping):
            raise TypeError("target_residuals must be a mapping")
        residuals = {}
        for name, value in self.target_residuals.items():
            if not isinstance(name, str) or not name:
                raise ValueError("target residual names must be non-empty strings")
            try:
                pair = tuple(value)
            except TypeError as error:
                raise ValueError(
                    f"target residual {name!r} must contain two values"
                ) from error
            if len(pair) != 2:
                raise ValueError(f"target residual {name!r} must contain two values")
            try:
                pair = (float(pair[0]), float(pair[1]))
            except (TypeError, ValueError) as error:
                raise TypeError("target residual values must be numeric") from error
            if not all(np.isfinite(item) and item >= 0.0 for item in pair):
                raise ValueError("target residuals must be finite and non-negative")
            residuals[name] = pair

        integer_fields = (
            "iterations",
            "accepted_steps",
            "limit_hits",
            "collision_evaluations",
            "collision_gradient_evaluations",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"result {name} must be an integer")
            if value < 0:
                raise ValueError(f"result {name} must be non-negative")
            object.__setattr__(self, name, int(value))
        required_non_negative = (
            "residual",
            "solve_ms",
            "collision_cost",
            "support_polygon_violation",
        )
        for name in required_non_negative:
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise TypeError(f"result {name} must be numeric") from error
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"result {name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        optional_non_negative = (
            "objective",
            "position_residual",
            "orientation_residual",
            "center_of_mass_residual",
            "zmp_residual",
        )
        for name in optional_non_negative:
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise TypeError(f"result {name} must be numeric") from error
            if not np.isnan(value) and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"result {name} must be non-negative or NaN")
            object.__setattr__(self, name, value)
        if not isinstance(self.termination_reason, str) or not self.termination_reason:
            raise ValueError("termination_reason must be a non-empty string")
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "target_residuals", MappingProxyType(residuals))


@dataclass(frozen=True)
class _RetargetingModePlan:
    targets: tuple[str, ...]
    frame_ids: tuple[int, ...]
    active_velocity_indices: np.ndarray


@dataclass
class _PinocchioSolveWorkspace:
    """Mode-sized mutable arrays reused by the single-threaded solve path."""

    weighted_error: np.ndarray
    weighted_jacobian: Optional[np.ndarray]
    system: Optional[np.ndarray]
    right_hand_side: Optional[np.ndarray]
    full_velocity: np.ndarray


class PinocchioRetargetingSolver:
    """Damped least-squares multi-frame retargeting with warm starts."""

    def __init__(
        self,
        urdf_path: Union[str, Path],
        frames: Mapping[str, str],
        joint_groups: Mapping[str, Sequence[str]],
        *,
        mode_manager: Optional[RetargetingModeManager] = None,
        damping: float = 1e-6,
        step_size: float = 0.5,
        tolerance: float = 1e-4,
        max_iterations: int = 50,
    ) -> None:
        self.pin = _load_pinocchio()
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF does not exist: {self.urdf_path}")
        damping = _positive_float(damping, "damping")
        step_size = _positive_float(step_size, "step_size")
        tolerance = _positive_float(tolerance, "tolerance")
        if not isinstance(max_iterations, Integral) or isinstance(
            max_iterations, bool
        ):
            raise TypeError("max_iterations must be an integer")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not isinstance(frames, Mapping):
            raise TypeError("frames must be a mapping")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(frame, str)
            or not frame
            for name, frame in frames.items()
        ):
            raise ValueError("frame mappings must contain non-empty string names")
        if not isinstance(joint_groups, Mapping):
            raise TypeError("joint_groups must be a mapping")
        normalized_groups = {}
        for group, names in joint_groups.items():
            if not isinstance(group, str) or not group:
                raise ValueError("joint group names must be non-empty strings")
            if isinstance(names, str):
                raise TypeError("joint groups must contain sequences of joint names")
            try:
                names = tuple(names)
            except TypeError as error:
                raise TypeError(
                    "joint groups must contain sequences of joint names"
                ) from error
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError(
                    "joint groups must contain non-empty string joint names"
                )
            if len(set(names)) != len(names):
                raise ValueError(f"joint group {group!r} contains duplicate joints")
            normalized_groups[group] = names

        self.model = self.pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.frames = MappingProxyType(dict(frames))
        self.joint_groups = MappingProxyType(normalized_groups)
        self.mode_manager = mode_manager or RetargetingModeManager()
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self._frame_ids = self._resolve_frames()
        self._group_velocity_indices = self._resolve_joint_groups()
        self._mode_plans: dict[RetargetingMode, _RetargetingModePlan] = {}
        self._solve_workspaces: dict[
            RetargetingMode, _PinocchioSolveWorkspace
        ] = {}
        self._lower_position_limits = np.asarray(
            self.model.lowerPositionLimit, dtype=float
        ).copy()
        self._upper_position_limits = np.asarray(
            self.model.upperPositionLimit, dtype=float
        ).copy()
        self._finite_lower_position_limits = np.isfinite(
            self._lower_position_limits
        )
        self._finite_upper_position_limits = np.isfinite(
            self._upper_position_limits
        )
        self._neutral_q = self._project_limits(
            np.asarray(self.pin.neutral(self.model), dtype=float)
        )
        self._last_q = self._neutral_q.copy()

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def mode(self) -> RetargetingMode:
        return self.mode_manager.mode

    def set_mode(self, mode: Union[RetargetingMode, str]) -> None:
        self.mode_manager.set_mode(mode)

    def prepare(
        self, mode: Optional[Union[RetargetingMode, str]] = None
    ) -> None:
        """Prepare immutable indices for a mode outside the solve hot path."""

        previous = self.mode
        if mode is not None:
            self.mode_manager.set_mode(mode)
        try:
            plan = self._mode_plan()
            self._solve_workspace(plan)
        except Exception:
            if self.mode is not previous:
                self.mode_manager.set_mode(previous)
            raise

    def reset(self, configuration: Optional[Sequence[float]] = None) -> None:
        self._last_q = self._configuration(configuration)

    def solve(
        self,
        targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
        seed: Optional[Sequence[float]] = None,
    ) -> RetargetingResult:
        normalized = self._normalize_targets(targets)
        plan = self._mode_plan()
        desired_poses = {
            name: self.pin.SE3(target.pose[:3, :3], target.pose[:3, 3])
            for name, target in normalized.items()
        }
        q = self._configuration(seed) if seed is not None else self._last_q.copy()
        active = plan.active_velocity_indices
        started = perf_counter()
        residual = float("inf")
        workspace = self._solve_workspace(plan)
        weighted_error = workspace.weighted_error
        weighted_jacobian = workspace.weighted_jacobian
        full_velocity = workspace.full_velocity
        full_velocity.fill(0.0)

        for iteration in range(1, self.max_iterations + 1):
            if active.size:
                self.pin.computeJointJacobians(self.model, self.data, q)
            else:
                self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            for target_index, (name, frame_id) in enumerate(
                zip(plan.targets, plan.frame_ids)
            ):
                target = normalized[name]
                desired = desired_poses[name]
                current = self.data.oMf[frame_id]
                error = np.asarray(self.pin.log6(current.inverse() * desired).vector)
                rows = slice(6 * target_index, 6 * (target_index + 1))
                scale = np.sqrt(target.weight)
                weighted_error[rows] = scale * error
                if active.size:
                    jacobian = np.asarray(
                        self.pin.getFrameJacobian(
                            self.model,
                            self.data,
                            frame_id,
                            self.pin.ReferenceFrame.LOCAL,
                        ),
                        dtype=float,
                    ).reshape(6, self.model.nv)
                    weighted_jacobian[rows] = scale * jacobian[:, active]
            residual = float(np.linalg.norm(weighted_error))
            if residual <= self.tolerance or not active.size:
                break
            jacobian = weighted_jacobian
            if jacobian.shape[0] <= jacobian.shape[1]:
                system = workspace.system
                np.matmul(jacobian, jacobian.T, out=system)
                system.flat[:: system.shape[0] + 1] += self.damping
                velocity = jacobian.T @ self._linear_solve(system, weighted_error)
            else:
                system = workspace.system
                np.matmul(jacobian.T, jacobian, out=system)
                system.flat[:: system.shape[0] + 1] += self.damping
                right_hand_side = workspace.right_hand_side
                np.matmul(jacobian.T, weighted_error, out=right_hand_side)
                velocity = self._linear_solve(system, right_hand_side)
            full_velocity[active] = self.step_size * velocity
            q = np.asarray(self.pin.integrate(self.model, q, full_velocity))
            q = self._project_limits(q)

        # If the budget is exhausted, the final integration above advances q
        # beyond the state whose residual was computed at the loop entrance.
        # Re-evaluate q so every returned metric describes configuration.
        if (
            active.size
            and iteration == self.max_iterations
            and residual > self.tolerance
        ):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            squared_residual = 0.0
            for name, frame_id in zip(plan.targets, plan.frame_ids):
                current = self.data.oMf[frame_id]
                error = np.asarray(
                    self.pin.log6(current.inverse() * desired_poses[name]).vector
                )
                squared_residual += normalized[name].weight * float(error @ error)
            residual = float(np.sqrt(squared_residual))

        self._last_q = q.copy()
        return RetargetingResult(
            configuration=q,
            success=residual <= self.tolerance,
            iterations=iteration,
            residual=residual,
            solve_ms=(perf_counter() - started) * 1000.0,
            mode=self.mode,
            objective=0.5 * residual * residual,
        )

    def solve_ik(
        self,
        head=None,
        left_wrist=None,
        right_wrist=None,
        ik_seed=None,
        **_unused,
    ):
        """Compatibility adapter for dexe_teleoperate's ``solve_ik`` API."""
        targets = {}
        if left_wrist is not None:
            targets["left_hand"] = left_wrist
        if right_wrist is not None:
            targets["right_hand"] = right_wrist
        if head is not None:
            targets["head"] = head
        result = self.solve(targets, seed=ik_seed)
        info = {
            "solve_ms": result.solve_ms,
            "iter_count": result.iterations,
            "residual": result.residual,
            "mode": result.mode.value,
        }
        # The modern result is immutable, while the legacy adapter historically
        # returned an owned mutable vector. Preserve that compatibility boundary.
        return result.configuration.copy(), result.success, False, [], info

    def reset_wrist_history(self) -> None:
        """Compatibility alias; generic targets do not keep wrist history."""
        self.reset()

    @staticmethod
    def _linear_solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, vector, rcond=None)[0]

    def _configuration(self, value: Optional[Sequence[float]]) -> np.ndarray:
        if value is None:
            return self._neutral_q.copy()
        q = np.asarray(value, dtype=float).reshape(-1)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError(
                f"configuration must be a finite vector of size {self.model.nq}"
            )
        return self._project_limits(q)

    def _normalize_targets(
        self,
        targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
    ) -> dict[str, RetargetingTarget]:
        if not isinstance(targets, Mapping):
            raise TypeError("targets must be a mapping")
        self.mode_manager.validate_targets(targets)
        plan = self._mode_plan()
        normalized = {
            name: targets[name]
            if isinstance(targets[name], RetargetingTarget)
            else RetargetingTarget(targets[name])
            for name in plan.targets
        }
        return normalized

    def _mode_plan(self) -> _RetargetingModePlan:
        cached = self._mode_plans.get(self.mode)
        if cached is not None:
            return cached
        spec = self.mode_manager.spec
        missing_frames = [name for name in spec.targets if name not in self._frame_ids]
        if missing_frames:
            raise ValueError(
                f"mode '{self.mode.value}' has no frame mappings for "
                f"{sorted(missing_frames)}"
            )
        missing_groups = [
            group
            for group in spec.active_joint_groups
            if group not in self._group_velocity_indices
        ]
        if missing_groups:
            raise ValueError(
                f"mode '{self.mode.value}' has no joint groups for "
                f"{sorted(missing_groups)}"
            )
        active = np.unique(
            np.concatenate(
                [
                    self._group_velocity_indices[group]
                    for group in spec.active_joint_groups
                ]
            )
        )
        active.setflags(write=False)
        plan = _RetargetingModePlan(
            targets=spec.targets,
            frame_ids=tuple(self._frame_ids[name] for name in spec.targets),
            active_velocity_indices=active,
        )
        self._mode_plans[self.mode] = plan
        return plan

    def _solve_workspace(
        self, plan: _RetargetingModePlan
    ) -> _PinocchioSolveWorkspace:
        cached = self._solve_workspaces.get(self.mode)
        if cached is not None:
            return cached
        task_dimension = 6 * len(plan.targets)
        active_size = plan.active_velocity_indices.size
        system_size = min(task_dimension, active_size) if active_size else 0
        cached = _PinocchioSolveWorkspace(
            weighted_error=np.empty(task_dimension, dtype=float),
            weighted_jacobian=(
                np.empty((task_dimension, active_size), dtype=float)
                if active_size
                else None
            ),
            system=(
                np.empty((system_size, system_size), dtype=float)
                if system_size
                else None
            ),
            right_hand_side=(
                np.empty(active_size, dtype=float)
                if active_size and task_dimension > active_size
                else None
            ),
            full_velocity=np.zeros(self.model.nv, dtype=float),
        )
        self._solve_workspaces[self.mode] = cached
        return cached

    def _resolve_frames(self) -> dict[str, int]:
        resolved = {}
        for logical_name, frame_name in self.frames.items():
            frame_id = int(self.model.getFrameId(frame_name))
            if frame_id >= len(self.model.frames):
                raise ValueError(f"URDF has no frame named {frame_name!r}")
            resolved[logical_name] = frame_id
        return resolved

    def _validate_mode_configuration(self) -> None:
        self._mode_plan()

    def _project_limits(self, q: np.ndarray) -> np.ndarray:
        bounded = q.copy()
        finite_lower = self._finite_lower_position_limits
        finite_upper = self._finite_upper_position_limits
        np.maximum(
            bounded,
            self._lower_position_limits,
            out=bounded,
            where=finite_lower,
        )
        np.minimum(
            bounded,
            self._upper_position_limits,
            out=bounded,
            where=finite_upper,
        )
        return np.asarray(self.pin.normalize(self.model, bounded), dtype=float)

    def _resolve_joint_groups(self) -> dict[str, np.ndarray]:
        resolved = {"whole_body": np.arange(self.model.nv, dtype=int)}
        model_names = set(self.model.names)
        for group, names in self.joint_groups.items():
            unknown = set(names).difference(model_names)
            if unknown:
                raise ValueError(
                    f"joint group {group!r} has unknown joints: {sorted(unknown)}"
                )
            indices = []
            for name in names:
                joint = self.model.joints[self.model.getJointId(name)]
                indices.extend(range(joint.idx_v, joint.idx_v + joint.nv))
            resolved[group] = np.asarray(sorted(set(indices)), dtype=int)
        return resolved

    def _active_velocity_indices(self) -> np.ndarray:
        return self._mode_plan().active_velocity_indices


# Compatibility with the class name used by dexe_teleoperate.
pinocchio_retargeting_solver = PinocchioRetargetingSolver
