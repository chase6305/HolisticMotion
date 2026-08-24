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
from typing import Optional, Union

import numpy as np

from .modes import RetargetingMode, RetargetingModeManager


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
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("target weight must be finite and positive")
        object.__setattr__(self, "pose", pose.copy())


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
        if damping <= 0.0 or step_size <= 0.0 or tolerance <= 0.0:
            raise ValueError("damping, step_size, and tolerance must be positive")
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
        self._last_q = np.asarray(self.pin.neutral(self.model), dtype=float)

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
        normalized = {
            name: value
            if isinstance(value, RetargetingTarget)
            else RetargetingTarget(value)
            for name, value in targets.items()
        }
        self.mode_manager.validate_targets(normalized)
        q = self._configuration(seed) if seed is not None else self._last_q.copy()
        active = self._active_velocity_indices()
        started = perf_counter()
        residual = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            errors, jacobians = [], []
            for name in self.mode_manager.spec.targets:
                target = normalized[name]
                frame_id = self._frame_ids[name]
                desired = self.pin.SE3(target.pose[:3, :3], target.pose[:3, 3])
                current = self.data.oMf[frame_id]
                error = np.asarray(self.pin.log6(current.inverse() * desired).vector)
                jacobian = self.pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    frame_id,
                    self.pin.ReferenceFrame.LOCAL,
                )
                scale = np.sqrt(target.weight)
                errors.append(scale * error)
                jacobians.append(scale * np.asarray(jacobian))
            error = np.concatenate(errors)
            residual = float(np.linalg.norm(error))
            if residual <= self.tolerance:
                break
            jacobian = np.vstack(jacobians)[:, active]
            system = jacobian @ jacobian.T
            velocity = jacobian.T @ np.linalg.solve(
                system + self.damping * np.eye(system.shape[0]), error
            )
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
        return result.configuration, result.success, False, [], info

    def reset_wrist_history(self) -> None:
        """Compatibility alias; generic targets do not keep wrist history."""
        self.reset()

    def _configuration(self, value: Optional[Sequence[float]]) -> np.ndarray:
        if value is None:
            return np.asarray(self.pin.neutral(self.model), dtype=float)
        q = np.asarray(value, dtype=float).reshape(-1)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError(
                f"configuration must be a finite vector of size {self.model.nq}"
            )
        return q.copy()

    def _resolve_frames(self) -> dict[str, int]:
        required = set()
        for spec in self.mode_manager.mode_specs.values():
            required.update(spec.targets)
        missing = required.difference(self.frames)
        if missing:
            raise ValueError(f"missing frame mappings for {sorted(missing)}")
        resolved = {}
        for logical_name, frame_name in self.frames.items():
            frame_id = int(self.model.getFrameId(frame_name))
            if frame_id >= len(self.model.frames):
                raise ValueError(f"URDF has no frame named {frame_name!r}")
            resolved[logical_name] = frame_id
        return resolved

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
