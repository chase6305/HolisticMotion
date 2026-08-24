"""Pink-style task-space retargeting on top of Pinocchio.

This module implements the weighted differential-IK objective locally.  It
does not import Pink or qpsolvers at runtime; Pinocchio remains the only
optional robotics dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Optional, Union

import numpy as np

from .pinocchio_solver import (
    PinocchioRetargetingSolver,
    RetargetingResult,
    RetargetingTarget,
)
from .tasks import FrameTask, PostureTask


class PinkRetargetingSolver(PinocchioRetargetingSolver):
    """Weighted task-space differential IK with posture and velocity limits."""

    def __init__(
        self,
        *args,
        frame_tasks: Optional[Mapping[str, FrameTask]] = None,
        posture_task: Optional[PostureTask] = None,
        integration_dt: float = 0.05,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
        stagnation_tolerance: float = 1e-8,
        stagnation_iterations: int = 6,
        max_backtracks: int = 5,
        acceleration_limits: Optional[
            Union[Sequence[float], Mapping[str, float]]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.frame_tasks = {name: FrameTask() for name in self._frame_ids}
        if frame_tasks:
            unknown = set(frame_tasks).difference(self._frame_ids)
            if unknown:
                raise ValueError(
                    f"frame tasks reference unknown targets: {sorted(unknown)}"
                )
            self.frame_tasks.update(frame_tasks)
        self.posture_task = posture_task or PostureTask()
        if not np.isfinite(integration_dt) or integration_dt <= 0.0:
            raise ValueError("integration_dt must be finite and positive")
        self.integration_dt = float(integration_dt)
        self.position_tolerance = float(position_tolerance or self.tolerance)
        self.orientation_tolerance = float(orientation_tolerance or self.tolerance)
        if self.position_tolerance <= 0.0 or self.orientation_tolerance <= 0.0:
            raise ValueError("task tolerances must be positive")
        if stagnation_tolerance < 0.0 or stagnation_iterations < 1:
            raise ValueError("stagnation settings must be non-negative and positive")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative")
        self.stagnation_tolerance = float(stagnation_tolerance)
        self.stagnation_iterations = int(stagnation_iterations)
        self.max_backtracks = int(max_backtracks)
        self._posture_q = np.asarray(self.pin.neutral(self.model), dtype=float)
        self._acceleration_limits = self._resolve_acceleration_limits(
            acceleration_limits
        )
        self._last_velocity = np.zeros(self.model.nv)
        self._active_cache = {
            mode: self._indices_for_groups(spec.active_joint_groups)
            for mode, spec in self.mode_manager.mode_specs.items()
        }

    def set_posture_target(self, configuration: Sequence[float]) -> None:
        self._posture_q = self._configuration(configuration)

    def reset(self, configuration: Optional[Sequence[float]] = None) -> None:
        super().reset(configuration)
        self._last_velocity.fill(0.0)

    def step(
        self,
        targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
        seed: Optional[Sequence[float]] = None,
    ) -> RetargetingResult:
        """Perform one bounded differential-IK control cycle."""

        return self.solve(
            targets,
            seed=seed,
            max_iterations=1,
            enforce_acceleration=True,
        )

    def solve(
        self,
        targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
        seed: Optional[Sequence[float]] = None,
        *,
        max_iterations: Optional[int] = None,
        enforce_acceleration: bool = False,
    ) -> RetargetingResult:
        normalized = {
            name: value
            if isinstance(value, RetargetingTarget)
            else RetargetingTarget(value)
            for name, value in targets.items()
        }
        self.mode_manager.validate_targets(normalized)
        q = self._configuration(seed) if seed is not None else self._last_q.copy()
        active = self._active_cache[self.mode]
        velocity_limits = np.asarray(self.model.velocityLimit, dtype=float)[active]
        velocity_limits = np.where(
            np.isfinite(velocity_limits) & (velocity_limits > 0.0),
            velocity_limits,
            np.inf,
        )
        started = perf_counter()
        regularization = self.damping
        accepted_steps = 0
        limit_hits = 0
        stagnant = 0
        termination_reason = "maximum_iterations"

        iteration_limit = (
            self.max_iterations if max_iterations is None else int(max_iterations)
        )
        if iteration_limit < 1:
            raise ValueError("max_iterations must be positive")
        for iteration in range(1, iteration_limit + 1):
            state = self._task_state(q, normalized, active, regularization, True)
            if self._converged(state):
                termination_reason = "converged"
                break
            lower, upper = self._displacement_bounds(
                q, active, velocity_limits, enforce_acceleration
            )
            unconstrained = self._linear_solve(state["hessian"], state["gradient"])
            displacement = self._solve_box_qp(
                state["hessian"], state["gradient"], lower, upper
            )
            limit_hits += int(
                np.count_nonzero(np.abs(displacement - unconstrained) > 1e-10)
            )

            accepted = False
            candidate_state = state
            backtrack_count = 0 if enforce_acceleration else self.max_backtracks
            for backtrack in range(backtrack_count + 1):
                scale = (
                    1.0 if enforce_acceleration else self.step_size * (0.5**backtrack)
                )
                full_displacement = np.zeros(self.model.nv)
                full_displacement[active] = scale * displacement
                candidate = np.asarray(
                    self.pin.integrate(self.model, q, full_displacement)
                )
                candidate = self._project_limits(candidate)
                candidate_state = self._task_state(
                    candidate, normalized, active, regularization, False
                )
                if enforce_acceleration or (
                    candidate_state["objective"] < state["objective"]
                ):
                    accepted = True
                    break
            if not accepted:
                regularization = min(regularization * 10.0, 1e6)
                stagnant += 1
            else:
                improvement = state["objective"] - candidate_state["objective"]
                q = candidate
                accepted_steps += 1
                if enforce_acceleration:
                    self._last_velocity.fill(0.0)
                    self._last_velocity[active] = (
                        scale * displacement / self.integration_dt
                    )
                regularization = max(self.damping, regularization * 0.5)
                stagnant = (
                    stagnant + 1 if improvement <= self.stagnation_tolerance else 0
                )
            if stagnant >= self.stagnation_iterations:
                termination_reason = "stagnated"
                break

        state = self._task_state(q, normalized, active, regularization, False)
        success = self._converged(state)
        if success:
            termination_reason = "converged"
        self._last_q = q.copy()
        return RetargetingResult(
            configuration=q,
            success=success,
            iterations=iteration,
            residual=state["residual"],
            solve_ms=(perf_counter() - started) * 1000.0,
            mode=self.mode,
            position_residual=state["position_residual"],
            orientation_residual=state["orientation_residual"],
            target_residuals=state["target_residuals"],
            termination_reason=termination_reason,
            accepted_steps=accepted_steps,
            limit_hits=limit_hits,
        )

    def _indices_for_groups(self, groups) -> np.ndarray:
        missing = set(groups).difference(self._group_velocity_indices)
        if missing:
            raise ValueError(f"mode references unknown joint groups: {sorted(missing)}")
        return np.unique(
            np.concatenate([self._group_velocity_indices[group] for group in groups])
        )

    def _converged(self, state: dict) -> bool:
        return bool(
            state["position_residual"] <= self.position_tolerance
            and state["orientation_residual"] <= self.orientation_tolerance
        )

    def _task_state(
        self,
        q: np.ndarray,
        targets: Mapping[str, RetargetingTarget],
        active: np.ndarray,
        regularization: float,
        build_system: bool,
    ) -> dict:
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        hessian = regularization * np.eye(active.size) if build_system else None
        gradient = np.zeros(active.size) if build_system else None
        weighted_errors = []
        target_residuals = {}
        for name in self.mode_manager.spec.targets:
            target = targets[name]
            task = self.frame_tasks[name]
            desired = self.pin.SE3(target.pose[:3, :3], target.pose[:3, 3])
            current = self.data.oMf[self._frame_ids[name]]
            error = np.asarray(self.pin.log6(current.inverse() * desired).vector)
            target_residuals[name] = (
                float(np.linalg.norm(error[:3])),
                float(np.linalg.norm(error[3:])),
            )
            weights = task.cost * np.sqrt(target.weight)
            weighted_error = weights * task.gain * error
            weighted_errors.append(weighted_error)
            if build_system:
                jacobian = np.asarray(
                    self.pin.computeFrameJacobian(
                        self.model,
                        self.data,
                        q,
                        self._frame_ids[name],
                        self.pin.ReferenceFrame.LOCAL,
                    )
                )[:, active]
                weighted_jacobian = weights[:, None] * jacobian
                hessian += weighted_jacobian.T @ weighted_jacobian
                if task.lm_damping:
                    hessian += (
                        task.lm_damping * float(error @ error) * np.eye(active.size)
                    )
                gradient += weighted_jacobian.T @ weighted_error

        posture_objective = 0.0
        if self.posture_task.cost > 0.0:
            posture_error = np.asarray(
                self.pin.difference(self.model, q, self._posture_q), dtype=float
            )[active]
            weight = self.posture_task.cost
            posture_objective = 0.5 * float(
                (weight * posture_error) @ (weight * posture_error)
            )
            if build_system:
                hessian += weight * weight * np.eye(active.size)
                gradient += weight * weight * self.posture_task.gain * posture_error

        joined = np.concatenate(weighted_errors)
        position_residual = max(value[0] for value in target_residuals.values())
        orientation_residual = max(value[1] for value in target_residuals.values())
        return {
            "hessian": hessian,
            "gradient": gradient,
            "objective": 0.5 * float(joined @ joined) + posture_objective,
            "residual": float(np.linalg.norm(joined)),
            "position_residual": position_residual,
            "orientation_residual": orientation_residual,
            "target_residuals": target_residuals,
        }

    @staticmethod
    def _linear_solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, vector, rcond=None)[0]

    @classmethod
    def _solve_box_qp(
        cls,
        hessian: np.ndarray,
        gradient: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Solve a positive-definite QP with projected-gradient refinement."""

        unconstrained = cls._linear_solve(hessian, gradient)
        if np.all(unconstrained >= lower) and np.all(unconstrained <= upper):
            return unconstrained
        solution = np.clip(unconstrained, lower, upper)
        lipschitz = max(float(np.linalg.eigvalsh(hessian)[-1]), 1e-12)
        for _ in range(40):
            candidate = np.clip(
                solution - (hessian @ solution - gradient) / lipschitz,
                lower,
                upper,
            )
            if np.max(np.abs(candidate - solution)) <= 1e-9:
                solution = candidate
                break
            solution = candidate
        return solution

    def _displacement_bounds(
        self,
        q: np.ndarray,
        active: np.ndarray,
        velocity_limits: np.ndarray,
        enforce_acceleration: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        max_displacement = velocity_limits * self.integration_dt
        lower = -max_displacement
        upper = max_displacement
        if enforce_acceleration:
            delta_velocity = self._acceleration_limits[active] * self.integration_dt
            lower = np.maximum(
                lower,
                (self._last_velocity[active] - delta_velocity) * self.integration_dt,
            )
            upper = np.minimum(
                upper,
                (self._last_velocity[active] + delta_velocity) * self.integration_dt,
            )
        if self.model.nq == self.model.nv:
            position_lower = (
                np.asarray(self.model.lowerPositionLimit)[active] - q[active]
            )
            position_upper = (
                np.asarray(self.model.upperPositionLimit)[active] - q[active]
            )
            dynamic_lower = lower.copy()
            dynamic_upper = upper.copy()
            lower = np.maximum(lower, position_lower)
            upper = np.minimum(upper, position_upper)
            # Position safety has priority when acceleration-limited braking
            # would otherwise force the next state across a hard joint bound.
            infeasible = lower > upper
            force_lower = infeasible & (position_lower > dynamic_upper)
            force_upper = infeasible & (position_upper < dynamic_lower)
            lower[force_lower] = position_lower[force_lower]
            upper[force_lower] = position_lower[force_lower]
            lower[force_upper] = position_upper[force_upper]
            upper[force_upper] = position_upper[force_upper]
        if np.any(lower > upper):
            raise RuntimeError("joint displacement bounds remain infeasible")
        return lower, upper

    def _resolve_acceleration_limits(
        self,
        limits: Optional[Union[Sequence[float], Mapping[str, float]]],
    ) -> np.ndarray:
        resolved = np.full(self.model.nv, np.inf)
        if limits is None:
            return resolved
        if isinstance(limits, Mapping):
            unknown = set(limits).difference(self.model.names)
            if unknown:
                raise ValueError(
                    f"unknown acceleration-limit joints: {sorted(unknown)}"
                )
            for name, value in limits.items():
                joint = self.model.joints[self.model.getJointId(name)]
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError("acceleration limits must be finite and positive")
                resolved[joint.idx_v : joint.idx_v + joint.nv] = float(value)
            return resolved
        values = np.asarray(limits, dtype=float).reshape(-1)
        if values.shape != (self.model.nv,) or not np.isfinite(values).all():
            raise ValueError(f"acceleration_limits must contain {self.model.nv} values")
        if np.any(values <= 0.0):
            raise ValueError("acceleration limits must be positive")
        return values


pink_retargeting_solver = PinkRetargetingSolver
