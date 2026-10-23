"""Continuous branch tracking with a separate history per worker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2, hypot, isfinite
from numbers import Integral
from typing import NamedTuple, Union

import numpy as np

from .._validation import real_array, require_real, validated_transform

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
        scalar_names = (
            "position_weight",
            "velocity_weight",
            "acceleration_weight",
            "branch_switch_penalty",
            "joint_limit_weight",
            "singular_value_threshold",
            "position_tolerance",
            "angle_tolerance",
        )
        scalars = tuple(
            require_real(getattr(self, name), name=name) for name in scalar_names
        )
        if any(isinstance(value, np.ndarray) and value.ndim != 0 for value in scalars):
            raise ValueError("tracker weights and thresholds must be scalars")
        try:
            scalars = tuple(float(value) for value in scalars)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "tracker weights and thresholds must be finite and non-negative"
            ) from error
        if not all(isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError(
                "tracker weights and thresholds must be finite and non-negative"
            )
        for name, value in zip(scalar_names, scalars):
            object.__setattr__(self, name, value)
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
            value = real_array(getattr(self, name), name=name)
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
            value = real_array(getattr(self, name), name=name, copy=True)
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
        metric_names = (
            "redundancy",
            "minimum_singular_value",
            "position_error",
            "angle_error",
        )
        metrics = tuple(
            require_real(getattr(self, name), name=name) for name in metric_names
        )
        if any(isinstance(value, np.ndarray) and value.ndim != 0 for value in metrics):
            raise ValueError("result metrics must be scalars")
        try:
            metrics = tuple(float(value) for value in metrics)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("result metrics must be finite") from error
        if not all(isfinite(value) for value in metrics):
            raise ValueError("result metrics must be finite")
        for name, value in zip(metric_names, metrics):
            object.__setattr__(self, name, value)
        if self.minimum_singular_value < 0.0:
            raise ValueError("minimum_singular_value must be non-negative")
        if self.position_error < 0.0 or self.angle_error < 0.0:
            raise ValueError("pose errors must be non-negative")
        for name in ("branch_changed", "near_singularity"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(value))
        if not isinstance(self.candidate_count, Integral) or isinstance(
            self.candidate_count, bool
        ):
            raise TypeError("candidate_count must be an integer")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        object.__setattr__(self, "candidate_count", int(self.candidate_count))


class _EvaluatedCandidate(NamedTuple):
    """Validated values used for ranking and publishing the selected sample."""

    score: float
    joints: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    configuration: tuple[int, int, int]
    redundancy: float
    position_error: float
    angle_error: float


class SRSContinuousTracker:
    """Select temporally continuous solutions from native SRS candidates.

    History is mutable, so each worker owns a tracker. A native solver may be
    shared for read-only queries; its configuration must remain unchanged.
    Tracker reset, option changes and step calls must not overlap.
    """

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
            raise ValueError(
                "solver returned invalid seven-axis joint limits"
            ) from error
        self._lower = real_array(lower, name="joint lower limits", copy=True)
        self._upper = real_array(upper, name="joint upper limits", copy=True)
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
        values = real_array(joints, name="initial_joints")
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
        dt = require_real(dt, name="dt")
        if isinstance(dt, np.ndarray) and dt.ndim != 0:
            raise ValueError("dt must be a scalar")
        try:
            dt = float(dt)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("dt must be finite and positive") from error
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        with np.errstate(over="ignore", invalid="ignore"):
            predicted = self._joints + self._velocity * dt
        if not np.isfinite(predicted).all():
            raise ValueError("dt produces a nonfinite joint prediction")
        candidates, refreshed = self._candidates(pose)
        evaluated = self._evaluate_candidates(candidates, pose, predicted, dt)
        if not evaluated and not refreshed:
            candidates, _ = self._candidates(pose, force_refresh=True)
            evaluated = self._evaluate_candidates(candidates, pose, predicted, dt)
        if not evaluated:
            raise ValueError(
                f"no {self._solver_label} candidate satisfies continuity limits"
            )
        evaluated.sort(key=lambda item: item.score)
        previous_pending = self._pending_branch, self._pending_count
        try:
            selected = self._apply_hysteresis(evaluated)
            previous_branch = self._configuration
            redundancy = self._unwrap_scalar(selected.redundancy, self._redundancy)
            singular_values = np.linalg.svd(
                np.asarray(self.solver.jacobian(selected.joints), dtype=float),
                compute_uv=False,
            )
            minimum = float(singular_values[-1]) if singular_values.size else 0.0
            result = self._result_type(
                joints=selected.joints,
                velocity=selected.velocity,
                acceleration=selected.acceleration,
                configuration=selected.configuration,
                redundancy=redundancy,
                branch_changed=selected.configuration != previous_branch,
                near_singularity=minimum <= self.options.singular_value_threshold,
                minimum_singular_value=minimum,
                position_error=selected.position_error,
                angle_error=selected.angle_error,
                candidate_count=len(evaluated),
            )
        except BaseException:
            self._pending_branch, self._pending_count = previous_pending
            raise

        self._joints = selected.joints
        self._velocity = selected.velocity
        self._configuration = selected.configuration
        self._redundancy = redundancy
        self._frame_index += 1
        return result

    def _evaluate_candidates(self, candidates, pose, predicted, dt):
        evaluated = []
        for raw in candidates:
            try:
                joints = self._unwrap(raw, predicted)
            except ValueError:
                # A native enumeration may contain an unusable branch near a
                # joint limit. Keep evaluating the other independent branches.
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                velocity = (joints - self._joints) / dt
                acceleration = (velocity - self._velocity) / dt
            if not np.all(
                np.isfinite(velocity) & (np.abs(velocity) <= self._max_velocity + 1e-12)
            ):
                continue
            if not np.all(
                np.isfinite(acceleration)
                & (np.abs(acceleration) <= self._max_acceleration + 1e-12)
            ):
                continue
            branch, redundancy = self._validated_configuration(
                self.solver.configuration(joints)
            )
            actual = np.asarray(self.solver.forward(joints), dtype=float)
            if actual.shape != (4, 4) or not np.isfinite(actual).all():
                continue
            position_error, angle_error = self._pose_error(actual, pose)
            if not isfinite(position_error) or not isfinite(angle_error):
                continue
            if position_error > self.options.position_tolerance:
                continue
            if angle_error > self.options.angle_tolerance:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                score = self._score(
                    joints, velocity, acceleration, predicted, branch, dt
                )
            if not isfinite(score):
                continue
            evaluated.append(
                _EvaluatedCandidate(
                    score=score,
                    joints=joints,
                    velocity=velocity,
                    acceleration=acceleration,
                    configuration=branch,
                    redundancy=redundancy,
                    position_error=position_error,
                    angle_error=angle_error,
                )
            )
        return evaluated

    def _candidates(
        self, target: np.ndarray, *, force_refresh: bool = False
    ) -> tuple[list[np.ndarray], bool]:
        candidates = []
        refresh = force_refresh
        if not force_refresh:
            try:
                candidates.extend(
                    self._finite_candidates(
                        self.solver.solve(target, self._joints, self._seeded_method())
                    )
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
                    self._configuration_of(item) != self._configuration
                    for item in candidates
                )
                or near_singularity
                or self._frame_index % self.options.candidate_refresh_interval == 0
            )
        if refresh:
            try:
                candidates.extend(
                    self._finite_candidates(
                        self.solver.solve(
                            target,
                            self._joints,
                            self._all_configurations_method(),
                        )
                    )
                )
            except ValueError:
                pass
        if not candidates:
            raise ValueError(
                f"{self._solver_label} inverse kinematics did not converge"
            )
        unique = []
        for values in candidates:
            if not any(
                np.linalg.norm(self._periodic_delta(values, item)) < 1e-8
                for item in unique
            ):
                unique.append(values)
        return unique, refresh

    def _finite_candidates(self, candidates):
        # Validate before querying branch metadata. Native configuration APIs
        # reject malformed vectors, which must not mask other valid branches
        # or prevent an all-configurations refresh after an unusable seed.
        for candidate in candidates:
            try:
                values = np.asarray(candidate)
                # Native real numeric arrays need only the shape/finiteness
                # checks below; object containers need recursive real guards.
                if values.dtype.kind not in "biuf":
                    values = require_real(values, name="solver candidate")
                values = np.asarray(values, dtype=float)
            except (TypeError, ValueError, OverflowError):
                continue
            if values.shape == (self._dof,) and np.isfinite(values).all():
                yield values

    def _all_configurations_method(self):
        import holistic_motion as hm

        return hm.SRSSolveMethod.ALL_CONFIGURATIONS

    def _seeded_method(self):
        import holistic_motion as hm

        return hm.SRSSolveMethod.SEEDED_NUMERICAL

    def _apply_hysteresis(self, evaluated):
        best = evaluated[0]
        best_branch = best.configuration
        if best_branch == self._configuration:
            self._pending_branch = None
            self._pending_count = 0
            return best
        current = next(
            (item for item in evaluated if item.configuration == self._configuration),
            None,
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
        options = self.options
        limit_distance = np.minimum(joints - self._lower, self._upper - joints)
        finite = np.isfinite(limit_distance)
        limit_penalty = (1.0 / np.maximum(limit_distance[finite], 1e-6) ** 2).sum()
        return float(
            options.position_weight * ((joints - predicted) ** 2).sum()
            + options.velocity_weight * ((velocity * dt) ** 2).sum()
            + options.acceleration_weight * ((acceleration * dt * dt) ** 2).sum()
            + options.branch_switch_penalty * (branch != self._configuration)
            + options.joint_limit_weight * limit_penalty
        )

    def _unwrap(self, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        values = np.asarray(candidate, dtype=float)
        if values.shape != (self._dof,) or not np.all(np.isfinite(values)):
            raise ValueError("candidate must contain seven finite values")
        period = 2.0 * np.pi
        turns = np.rint((reference - values) / period)
        minimum = np.ceil((self._lower - values) / period)
        maximum = np.floor((self._upper - values) / period)
        turns = np.minimum(np.maximum(turns, minimum), maximum)
        # Match the scalar integer path's canonical positive zero turn.
        turns[turns == 0.0] = 0.0
        result = values + turns * period
        if not np.isfinite(result).all() or np.any(
            (result < self._lower - 1e-12) | (result > self._upper + 1e-12)
        ):
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
            redundancy = require_real(native.redundancy, name="solver redundancy")
            if isinstance(redundancy, np.ndarray) and redundancy.ndim != 0:
                raise ValueError("redundancy must be a scalar")
            redundancy = float(redundancy)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("solver returned an invalid configuration") from error
        if not isfinite(redundancy):
            raise ValueError("solver returned an invalid configuration")
        if all(type(value) is int for value in raw_branch):
            return raw_branch, redundancy
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in raw_branch
        ):
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
        cosine = min(
            1.0,
            max(
                -1.0,
                float((relative[0, 0] + relative[1, 1] + relative[2, 2] - 1.0) * 0.5),
            ),
        )
        sine = 0.5 * hypot(
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        )
        return (
            hypot(*(actual[:3, 3] - target[:3, 3])),
            atan2(sine, cosine),
        )
