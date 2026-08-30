"""Generic Pinocchio inverse-kinematics retargeting solver.

The solver is adapted from dexe_teleoperate's pose retargeting solver.  This
version deliberately excludes ROS, robot-specific YAML, and controller state;
all frame and joint mappings are supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Optional, Union

import numpy as np

from .modes import RetargetingMode, RetargetingModeManager

_IDENTITY_ROTATION = np.eye(3)
_IDENTITY_ROTATION.setflags(write=False)
_HOMOGENEOUS_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0])
_HOMOGENEOUS_LAST_ROW.setflags(write=False)


def _readonly_array(value, *, shape=None, name="array") -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
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
        pose = np.asarray(self.pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("target pose must be a finite 4x4 matrix")
        if not np.allclose(pose[3], _HOMOGENEOUS_LAST_ROW, rtol=0.0, atol=1e-8):
            raise ValueError("target pose must have a valid homogeneous last row")
        rotation = pose[:3, :3]
        determinant = (
            rotation[0, 0]
            * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
            - rotation[0, 1]
            * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
            + rotation[0, 2]
            * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
        )
        if (
            not np.allclose(
                rotation.T @ rotation, _IDENTITY_ROTATION, rtol=1e-5, atol=1e-6
            )
            or abs(determinant - 1.0) > 1.1e-5
        ):
            raise ValueError("target pose rotation must be a proper orthonormal matrix")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("target weight must be finite and positive")
        pose = pose.copy()
        pose.setflags(write=False)
        object.__setattr__(self, "pose", pose)


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
        residuals = {
            str(name): (float(value[0]), float(value[1]))
            for name, value in self.target_residuals.items()
        }
        if not np.isfinite(self.collision_cost) or self.collision_cost < 0.0:
            raise ValueError("result collision_cost must be finite and non-negative")
        if self.collision_evaluations < 0 or self.collision_gradient_evaluations < 0:
            raise ValueError("result collision evaluation counts must be non-negative")
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "target_residuals", MappingProxyType(residuals))


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
        numeric_options = (damping, step_size, tolerance)
        if not all(np.isfinite(value) and value > 0.0 for value in numeric_options):
            raise ValueError(
                "damping, step_size, and tolerance must be finite and positive"
            )
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        self.model = self.pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.frames = dict(frames)
        self.joint_groups = {key: tuple(value) for key, value in joint_groups.items()}
        self.mode_manager = mode_manager or RetargetingModeManager()
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self._frame_ids = self._resolve_frames()
        self._group_velocity_indices = self._resolve_joint_groups()
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

    def reset(self, configuration: Optional[Sequence[float]] = None) -> None:
        self._last_q = self._configuration(configuration)

    def solve(
        self,
        targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
        seed: Optional[Sequence[float]] = None,
    ) -> RetargetingResult:
        self.mode_manager.validate_targets(targets)
        normalized = {
            name: targets[name]
            if isinstance(targets[name], RetargetingTarget)
            else RetargetingTarget(targets[name])
            for name in self.mode_manager.spec.targets
        }
        self._validate_mode_configuration()
        desired_poses = {
            name: self.pin.SE3(target.pose[:3, :3], target.pose[:3, 3])
            for name, target in normalized.items()
        }
        q = self._configuration(seed) if seed is not None else self._last_q.copy()
        active = self._active_velocity_indices()
        started = perf_counter()
        residual = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            if active.size:
                self.pin.computeJointJacobians(self.model, self.data, q)
            else:
                self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            errors, jacobians = [], []
            for name in self.mode_manager.spec.targets:
                target = normalized[name]
                frame_id = self._frame_ids[name]
                desired = desired_poses[name]
                current = self.data.oMf[frame_id]
                error = np.asarray(self.pin.log6(current.inverse() * desired).vector)
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
                else:
                    jacobian = np.zeros((6, 0))
                scale = np.sqrt(target.weight)
                errors.append(scale * error)
                jacobians.append(scale * jacobian)
            error = np.concatenate(errors)
            residual = float(np.linalg.norm(error))
            if residual <= self.tolerance or not active.size:
                break
            jacobian = np.vstack(jacobians)[:, active]
            if jacobian.shape[0] <= jacobian.shape[1]:
                system = jacobian @ jacobian.T
                system.flat[:: system.shape[0] + 1] += self.damping
                velocity = jacobian.T @ self._linear_solve(system, error)
            else:
                system = jacobian.T @ jacobian
                system.flat[:: system.shape[0] + 1] += self.damping
                velocity = self._linear_solve(system, jacobian.T @ error)
            full_velocity = np.zeros(self.model.nv)
            full_velocity[active] = self.step_size * velocity
            q = np.asarray(self.pin.integrate(self.model, q, full_velocity))
            q = self._project_limits(q)

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

    def _resolve_frames(self) -> dict[str, int]:
        resolved = {}
        for logical_name, frame_name in self.frames.items():
            frame_id = int(self.model.getFrameId(frame_name))
            if frame_id >= len(self.model.frames):
                raise ValueError(f"URDF has no frame named {frame_name!r}")
            resolved[logical_name] = frame_id
        return resolved

    def _validate_mode_configuration(self) -> None:
        missing_frames = set(self.mode_manager.spec.targets).difference(self._frame_ids)
        if missing_frames:
            raise ValueError(
                f"mode '{self.mode.value}' has no frame mappings for "
                f"{sorted(missing_frames)}"
            )
        missing_groups = set(self.mode_manager.spec.active_joint_groups).difference(
            self._group_velocity_indices
        )
        if missing_groups:
            raise ValueError(
                f"mode '{self.mode.value}' has no joint groups for "
                f"{sorted(missing_groups)}"
            )

    def _project_limits(self, q: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.model.lowerPositionLimit, dtype=float)
        upper = np.asarray(self.model.upperPositionLimit, dtype=float)
        bounded = q.copy()
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        bounded[finite_lower] = np.maximum(bounded[finite_lower], lower[finite_lower])
        bounded[finite_upper] = np.minimum(bounded[finite_upper], upper[finite_upper])
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
        groups = self.mode_manager.spec.active_joint_groups
        missing = set(groups).difference(self._group_velocity_indices)
        if missing:
            raise ValueError(f"mode references unknown joint groups: {sorted(missing)}")
        return np.unique(
            np.concatenate([self._group_velocity_indices[group] for group in groups])
        )


# Compatibility with the class name used by dexe_teleoperate.
pinocchio_retargeting_solver = PinocchioRetargetingSolver
