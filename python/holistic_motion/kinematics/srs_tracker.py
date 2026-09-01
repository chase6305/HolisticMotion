"""Continuous branch tracking for the native stateless SRS solver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Union

import numpy as np

from .._validation import validated_transform

Limit = Union[float, Sequence[float], np.ndarray]


@dataclass(frozen=True)
class SRSContinuousOptions:
    """Weights and hard limits for continuous SRS candidate selection."""

    position_weight: float = 1.0
    velocity_weight: float = 0.05
    acceleration_weight: float = 0.002
    branch_switch_penalty: float = 1.0
    joint_limit_weight: float = 1e-4
    branch_hysteresis_frames: int = 3
    candidate_refresh_interval: int = 10
    singular_value_threshold: float = 1e-5
    position_tolerance: float = 1e-5
    angle_tolerance: float = 1e-5
    max_velocity: Limit = np.inf
    max_acceleration: Limit = np.inf

    def __post_init__(self) -> None:
        scalars = (
            self.position_weight,
            self.velocity_weight,
            self.acceleration_weight,
            self.branch_switch_penalty,
            self.joint_limit_weight,
            self.singular_value_threshold,
            self.position_tolerance,
            self.angle_tolerance,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError(
                "tracker weights and thresholds must be finite and non-negative"
            )
        if not isinstance(self.branch_hysteresis_frames, Integral) or isinstance(
            self.branch_hysteresis_frames, bool
        ):
            raise TypeError("branch_hysteresis_frames must be an integer")
        if self.branch_hysteresis_frames < 1:
            raise ValueError("branch_hysteresis_frames must be positive")
        if not isinstance(self.candidate_refresh_interval, Integral) or isinstance(
            self.candidate_refresh_interval, bool
        ):
            raise TypeError("candidate_refresh_interval must be an integer")
        if self.candidate_refresh_interval < 1:
            raise ValueError("candidate_refresh_interval must be positive")
        object.__setattr__(
            self, "branch_hysteresis_frames", int(self.branch_hysteresis_frames)
        )
        object.__setattr__(
            self, "candidate_refresh_interval", int(self.candidate_refresh_interval)
        )
        for name in ("max_velocity", "max_acceleration"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim == 0:
                normalized = float(value)
                valid = not np.isnan(normalized) and normalized > 0.0
            else:
                value = value.reshape(-1)
                valid = (
                    value.shape == (7,)
                    and not np.isnan(value).any()
                    and np.all(value > 0.0)
                )
                normalized = tuple(float(item) for item in value)
            if not valid:
                raise ValueError(f"{name} must be positive with one or seven values")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class SRSContinuousResult:
    """One accepted sample from :class:`SRSContinuousTracker`."""

    joints: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    configuration: tuple[int, int, int]
    redundancy: float
    branch_changed: bool
    near_singularity: bool
    minimum_singular_value: float
    position_error: float
    angle_error: float
    candidate_count: int

    def __post_init__(self) -> None:
        for name in ("joints", "velocity", "acceleration"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain seven finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.configuration, tuple)
            or len(self.configuration) != 3
            or any(
                not isinstance(value, Integral) or isinstance(value, bool)
                for value in self.configuration
            )
        ):
            raise ValueError("configuration must contain three integer branch values")
        object.__setattr__(
            self, "configuration", tuple(int(value) for value in self.configuration)
        )
        metrics = (
            self.redundancy,
            self.minimum_singular_value,
            self.position_error,
            self.angle_error,
        )
        if not all(np.isfinite(value) for value in metrics):
            raise ValueError("result metrics must be finite")
        if self.minimum_singular_value < 0.0:
            raise ValueError("minimum_singular_value must be non-negative")
        if self.position_error < 0.0 or self.angle_error < 0.0:
            raise ValueError("pose errors must be non-negative")
        for name in ("branch_changed", "near_singularity"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(value))
        for name in (
            "redundancy",
            "minimum_singular_value",
            "position_error",
            "angle_error",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not isinstance(self.candidate_count, Integral) or isinstance(
            self.candidate_count, bool
        ):
            raise TypeError("candidate_count must be an integer")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        object.__setattr__(self, "candidate_count", int(self.candidate_count))


class SRSContinuousTracker:
    """Select temporally continuous solutions from native SRS candidates."""

    _dof = 7
    _solver_label = "SRS"
    _options_type = SRSContinuousOptions
    _result_type = SRSContinuousResult

    def __init__(
        self,
        solver,
        initial_joints: Sequence[float],
        options: SRSContinuousOptions | None = None,
    ) -> None:
        if not isinstance(options, (type(None), self._options_type)):
            raise TypeError(f"options must be {self._options_type.__name__} or None")
        required_api = ("configuration", "forward", "jacobian", "solve")
        if not getattr(solver, "compatible", False) or any(
            not callable(getattr(solver, name, None)) for name in required_api
        ):
            raise ValueError(
                f"{type(self).__name__} requires a compatible "
                f"{self._solver_label} solver"
            )
        self._solver = solver
        self._options = options or self._options_type()
        try:
            lower, upper = solver.joint_limits
        except (TypeError, ValueError) as error:
            raise ValueError("solver returned invalid seven-axis joint limits") from error
        self._lower = np.array(lower, dtype=float, copy=True)
        self._upper = np.array(upper, dtype=float, copy=True)
        if (
            self._lower.shape != (self._dof,)
            or self._upper.shape != (self._dof,)
            or np.any(np.isnan(self._lower))
            or np.any(np.isnan(self._upper))
            or np.any(self._lower > self._upper)
        ):
            raise ValueError("solver returned invalid seven-axis joint limits")
        self._max_velocity = self._limit_vector(
            self.options.max_velocity, "max_velocity"
        )
        self._max_acceleration = self._limit_vector(
            self.options.max_acceleration, "max_acceleration"
        )
        self._pending_branch: tuple[int, int, int] | None = None
        self._pending_count = 0
        self._frame_index = 0
        self.reset(initial_joints)

    @property
    def solver(self):
        """Native solver used by this tracker (read-only)."""
        return self._solver

    @property
    def options(self) -> SRSContinuousOptions:
        """Validated immutable tracker options."""
        return self._options

    @property
    def joints(self) -> np.ndarray:
        return self._joints.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    @property
    def configuration(self) -> tuple[int, int, int]:
        return self._configuration

    @property
    def redundancy(self) -> float:
        return self._redundancy

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def reset(self, joints: Sequence[float]) -> None:
        values = np.asarray(joints, dtype=float)
        if values.shape != (self._dof,) or not np.all(np.isfinite(values)):
            raise ValueError("initial_joints must contain seven finite values")
        if np.any(values < self._lower) or np.any(values > self._upper):
            raise ValueError("initial_joints violate joint limits")
        native = self.solver.configuration(values)
        configuration, redundancy = self._validated_configuration(native)

        self._joints = values.copy()
        self._velocity = np.zeros(self._dof)
        self._configuration = configuration
        self._redundancy = redundancy
        self._pending_branch = None
        self._pending_count = 0
        self._frame_index = 0

    def solve(self, target: np.ndarray, dt: float) -> SRSContinuousResult:
        pose = validated_transform(target, name="target")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        predicted = self._joints + self._velocity * dt
        candidates = self._candidates(pose)
        evaluated = []
        for raw in candidates:
            try:
                joints = self._unwrap(raw, predicted)
            except ValueError:
                # A native enumeration may contain an unusable branch near a
                # joint limit. Keep evaluating the other independent branches.
                continue
            velocity = (joints - self._joints) / dt
            acceleration = (velocity - self._velocity) / dt
            if np.any(np.abs(velocity) > self._max_velocity + 1e-12):
                continue
            if np.any(np.abs(acceleration) > self._max_acceleration + 1e-12):
                continue
            branch = self._configuration_of(joints)
            actual = np.asarray(self.solver.forward(joints), dtype=float)
            position_error, angle_error = self._pose_error(actual, pose)
            if position_error > self.options.position_tolerance:
                continue
            if angle_error > self.options.angle_tolerance:
                continue
            score = self._score(joints, velocity, acceleration, predicted, branch, dt)
            evaluated.append(
                (
                    score,
                    joints,
                    velocity,
                    acceleration,
                    branch,
                    position_error,
                    angle_error,
                )
            )
        if not evaluated:
            raise ValueError(
                f"no {self._solver_label} candidate satisfies continuity limits"
            )
        evaluated.sort(key=lambda item: item[0])
        previous_pending = self._pending_branch, self._pending_count
        try:
            selected = self._apply_hysteresis(evaluated)
            (
                _,
                joints,
                velocity,
                acceleration,
                branch,
                position_error,
                angle_error,
            ) = selected
            previous_branch = self._configuration
            native = self.solver.configuration(joints)
            redundancy = self._unwrap_scalar(
                float(native.redundancy), self._redundancy
            )
            singular_values = np.linalg.svd(
                np.asarray(self.solver.jacobian(joints), dtype=float),
                compute_uv=False,
            )
            minimum = float(singular_values[-1]) if singular_values.size else 0.0
            result = self._result_type(
                joints=joints.copy(),
                velocity=velocity.copy(),
                acceleration=acceleration.copy(),
                configuration=branch,
                redundancy=redundancy,
                branch_changed=branch != previous_branch,
                near_singularity=minimum <= self.options.singular_value_threshold,
                minimum_singular_value=minimum,
                position_error=position_error,
                angle_error=angle_error,
                candidate_count=len(evaluated),
            )
        except Exception:
            self._pending_branch, self._pending_count = previous_pending
            raise

        self._joints = joints
        self._velocity = velocity
        self._configuration = branch
        self._redundancy = redundancy
        self._frame_index += 1
        return result

    def _candidates(self, target: np.ndarray) -> list[np.ndarray]:
        candidates = []
        try:
            candidates.extend(
                self.solver.solve(target, self._joints, self._seeded_method())
            )
        except ValueError:
            pass
        singular_values = np.linalg.svd(
            np.asarray(self.solver.jacobian(self._joints), dtype=float),
            compute_uv=False,
        )
        near_singularity = (
            singular_values.size == 0
            or singular_values[-1] <= self.options.singular_value_threshold
        )
        refresh = (
            not candidates
            or any(
                self._configuration_of(np.asarray(item)) != self._configuration
                for item in candidates
            )
            or near_singularity
            or self._frame_index % self.options.candidate_refresh_interval == 0
        )
        if refresh:
            try:
                candidates.extend(
                    self.solver.solve(
                        target,
                        self._joints,
                        self._all_configurations_method(),
                    )
                )
            except ValueError:
                pass
        if not candidates:
            raise ValueError(
                f"{self._solver_label} inverse kinematics did not converge"
            )
        unique = []
        for candidate in candidates:
            values = np.asarray(candidate, dtype=float)
            if values.shape != (self._dof,) or not np.all(np.isfinite(values)):
                continue
            if not any(
                np.linalg.norm(self._periodic_delta(values, item)) < 1e-8
                for item in unique
            ):
                unique.append(values)
        return unique

    def _all_configurations_method(self):
        import holistic_motion as hm

        return hm.SRSSolveMethod.ALL_CONFIGURATIONS

    def _seeded_method(self):
        import holistic_motion as hm

        return hm.SRSSolveMethod.SEEDED_NUMERICAL

    def _apply_hysteresis(self, evaluated):
        best = evaluated[0]
        best_branch = best[4]
        if best_branch == self._configuration:
            self._pending_branch = None
            self._pending_count = 0
            return best
        current = next(
            (item for item in evaluated if item[4] == self._configuration), None
        )
        if current is None:
            self._pending_branch = None
            self._pending_count = 0
            return best
        if self._pending_branch == best_branch:
            self._pending_count += 1
        else:
            self._pending_branch = best_branch
            self._pending_count = 1
        if self._pending_count < self.options.branch_hysteresis_frames:
            return current
        self._pending_branch = None
        self._pending_count = 0
        return best

    def _score(self, joints, velocity, acceleration, predicted, branch, dt) -> float:
        limit_distance = np.minimum(joints - self._lower, self._upper - joints)
        finite = np.isfinite(limit_distance)
        limit_penalty = np.sum(1.0 / np.maximum(limit_distance[finite], 1e-6) ** 2)
        return float(
            self.options.position_weight * np.sum((joints - predicted) ** 2)
            + self.options.velocity_weight * np.sum((velocity * dt) ** 2)
            + self.options.acceleration_weight * np.sum((acceleration * dt * dt) ** 2)
            + self.options.branch_switch_penalty * (branch != self._configuration)
            + self.options.joint_limit_weight * limit_penalty
        )

    def _unwrap(self, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        values = np.asarray(candidate, dtype=float)
        if values.shape != (self._dof,) or not np.all(np.isfinite(values)):
            raise ValueError("candidate must contain seven finite values")
        result = values.copy()
        period = 2.0 * np.pi
        for index in range(self._dof):
            turn = int(np.rint((reference[index] - values[index]) / period))
            if np.isfinite(self._lower[index]):
                minimum = int(np.ceil((self._lower[index] - values[index]) / period))
                turn = max(turn, minimum)
            if np.isfinite(self._upper[index]):
                maximum = int(np.floor((self._upper[index] - values[index]) / period))
                turn = min(turn, maximum)
            result[index] = values[index] + turn * period
            if result[index] < self._lower[index] - 1e-12:
                raise ValueError("candidate has no joint-limit-equivalent angle")
            if result[index] > self._upper[index] + 1e-12:
                raise ValueError("candidate has no joint-limit-equivalent angle")
        return result

    @staticmethod
    def _periodic_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        delta = first - second
        return np.arctan2(np.sin(delta), np.cos(delta))

    @staticmethod
    def _unwrap_scalar(value: float, reference: float) -> float:
        return value + 2.0 * np.pi * np.rint((reference - value) / (2.0 * np.pi))

    def _configuration_of(self, joints: np.ndarray) -> tuple[int, int, int]:
        configuration, _ = self._validated_configuration(
            self.solver.configuration(joints)
        )
        return configuration

    @staticmethod
    def _validated_configuration(native) -> tuple[tuple[int, int, int], float]:
        try:
            raw_branch = (native.shoulder, native.elbow, native.wrist)
            redundancy = float(native.redundancy)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("solver returned an invalid configuration") from error
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in raw_branch
        ) or not np.isfinite(redundancy):
            raise ValueError("solver returned an invalid configuration")
        return tuple(int(value) for value in raw_branch), redundancy

    def _limit_vector(self, value: Limit, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.ndim == 0:
            result = np.full(self._dof, float(result))
        if (
            result.shape != (self._dof,)
            or np.any(np.isnan(result))
            or np.any(result <= 0.0)
        ):
            raise ValueError(f"{name} must be positive with one or seven values")
        return result

    @staticmethod
    def _pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        relative = actual[:3, :3].T @ target[:3, :3]
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        return (
            float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
            float(np.arccos(cosine)),
        )
