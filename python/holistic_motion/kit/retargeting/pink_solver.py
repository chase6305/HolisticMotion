"""Pink-style task-space retargeting on top of Pinocchio.

This module implements the weighted differential-IK objective locally.  It
does not import Pink or qpsolvers at runtime; Pinocchio remains the only
optional robotics dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from time import perf_counter
from typing import Callable, Optional, Union

import numpy as np

from .modes import RetargetingMode
from .pinocchio_solver import (
    PinocchioRetargetingSolver,
    RetargetingResult,
    RetargetingTarget,
)
from .tasks import CenterOfMassTask, FrameTask, PostureTask, SupportPolygonTask, ZmpTask


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
        collision_cost: Optional[Callable[[np.ndarray], float]] = None,
        collision_gradient: Optional[Callable[[np.ndarray], Sequence[float]]] = None,
        collision_cost_gradient: Optional[
            Callable[[np.ndarray], tuple[float, Sequence[float]]]
        ] = None,
        collision_cost_weight: float = 1.0,
        collision_tolerance: float = 0.0,
        collision_finite_difference_step: float = 1e-4,
        center_of_mass_task: Optional[CenterOfMassTask] = None,
        center_of_mass_tolerance: Optional[float] = None,
        support_polygon_task: Optional[SupportPolygonTask] = None,
        support_polygon_tolerance: float = 1e-6,
        zmp_task: Optional[ZmpTask] = None,
        zmp_tolerance: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.frame_tasks = {name: FrameTask() for name in self._frame_ids}
        if frame_tasks is not None and not isinstance(frame_tasks, Mapping):
            raise TypeError("frame_tasks must be a mapping or None")
        if frame_tasks:
            unknown = set(frame_tasks).difference(self._frame_ids)
            if unknown:
                raise ValueError(
                    f"frame tasks reference unknown targets: {sorted(unknown)}"
                )
            if any(not isinstance(task, FrameTask) for task in frame_tasks.values()):
                raise TypeError("frame_tasks values must be FrameTask objects")
            self.frame_tasks.update(frame_tasks)
        if posture_task is not None and not isinstance(posture_task, PostureTask):
            raise TypeError("posture_task must be PostureTask or None")
        self.posture_task = posture_task or PostureTask()
        if not np.isfinite(integration_dt) or integration_dt <= 0.0:
            raise ValueError("integration_dt must be finite and positive")
        self.integration_dt = float(integration_dt)
        self.position_tolerance = float(
            self.tolerance if position_tolerance is None else position_tolerance
        )
        self.orientation_tolerance = float(
            self.tolerance if orientation_tolerance is None else orientation_tolerance
        )
        if not (
            np.isfinite(self.position_tolerance)
            and self.position_tolerance > 0.0
            and np.isfinite(self.orientation_tolerance)
            and self.orientation_tolerance > 0.0
        ):
            raise ValueError("task tolerances must be finite and positive")
        if not isinstance(stagnation_iterations, Integral) or isinstance(
            stagnation_iterations, bool
        ):
            raise TypeError("stagnation_iterations must be an integer")
        if not np.isfinite(stagnation_tolerance) or stagnation_tolerance < 0.0:
            raise ValueError("stagnation tolerance must be finite and non-negative")
        if stagnation_iterations < 1:
            raise ValueError("stagnation_iterations must be positive")
        if not isinstance(max_backtracks, Integral) or isinstance(max_backtracks, bool):
            raise TypeError("max_backtracks must be an integer")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative")
        self.stagnation_tolerance = float(stagnation_tolerance)
        self.stagnation_iterations = int(stagnation_iterations)
        self.max_backtracks = int(max_backtracks)
        callbacks = {
            "collision_cost": collision_cost,
            "collision_gradient": collision_gradient,
            "collision_cost_gradient": collision_cost_gradient,
        }
        for name, callback in callbacks.items():
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None")
        if collision_gradient is not None and collision_cost is None:
            raise ValueError("collision_gradient requires collision_cost")
        if collision_cost_gradient is not None and collision_cost is None:
            raise ValueError("collision_cost_gradient requires collision_cost")
        if collision_cost_gradient is not None and collision_gradient is not None:
            raise ValueError(
                "provide collision_gradient or collision_cost_gradient, not both"
            )
        collision_options = (
            collision_cost_weight,
            collision_tolerance,
            collision_finite_difference_step,
        )
        if not all(np.isfinite(value) for value in collision_options):
            raise ValueError("collision options must be finite")
        if collision_cost_weight < 0.0 or collision_tolerance < 0.0:
            raise ValueError("collision weight and tolerance must be non-negative")
        if collision_finite_difference_step <= 0.0:
            raise ValueError("collision finite-difference step must be positive")
        self.collision_cost = collision_cost
        self.collision_gradient = collision_gradient
        self.collision_cost_gradient = collision_cost_gradient
        self.collision_cost_weight = float(collision_cost_weight)
        self.collision_tolerance = float(collision_tolerance)
        self.collision_finite_difference_step = float(collision_finite_difference_step)
        if center_of_mass_task is not None and not isinstance(
            center_of_mass_task, CenterOfMassTask
        ):
            raise TypeError("center_of_mass_task must be CenterOfMassTask or None")
        self.center_of_mass_task = center_of_mass_task
        self.center_of_mass_tolerance = float(
            self.tolerance
            if center_of_mass_tolerance is None
            else center_of_mass_tolerance
        )
        if (
            not np.isfinite(self.center_of_mass_tolerance)
            or self.center_of_mass_tolerance <= 0.0
        ):
            raise ValueError("center-of-mass tolerance must be finite and positive")
        self._center_of_mass_target = None
        if (
            not np.isfinite(support_polygon_tolerance)
            or support_polygon_tolerance < 0.0
        ):
            raise ValueError(
                "support polygon tolerance must be finite and non-negative"
            )
        if support_polygon_task is not None and not isinstance(
            support_polygon_task, SupportPolygonTask
        ):
            raise TypeError("support_polygon_task must be SupportPolygonTask or None")
        self.support_polygon_task = support_polygon_task
        self.support_polygon_tolerance = float(support_polygon_tolerance)
        if zmp_task is not None and not isinstance(zmp_task, ZmpTask):
            raise TypeError("zmp_task must be ZmpTask or None")
        self.zmp_task = zmp_task
        self.zmp_tolerance = float(
            self.tolerance if zmp_tolerance is None else zmp_tolerance
        )
        if not np.isfinite(self.zmp_tolerance) or self.zmp_tolerance <= 0.0:
            raise ValueError("ZMP tolerance must be finite and positive")
        self._zmp_target = None
        self._center_of_mass_acceleration = np.zeros(3)
        if (
            self.support_polygon_task is not None
            and self.support_polygon_task.reference == "zmp"
            and self.zmp_task is None
        ):
            raise ValueError("ZMP support polygon reference requires zmp_task")
        self._posture_q = self._neutral_q.copy()
        self._acceleration_limits = self._resolve_acceleration_limits(
            acceleration_limits
        )
        self._last_velocity = np.zeros(self.model.nv)
        self._active_cache = {}
        self._system_workspace = {}
        model_velocity_limits = np.asarray(self.model.velocityLimit, dtype=float)
        self._model_velocity_limits = model_velocity_limits
        self._velocity_limit_cache = {}

    def set_posture_target(self, configuration: Sequence[float]) -> None:
        self._posture_q = self._configuration(configuration)

    def prepare(
        self, mode: Optional[Union[RetargetingMode, str]] = None
    ) -> None:
        """Prepare mode indices, limits, and numerical workspaces."""

        super().prepare(mode)
        active, _ = self._mode_limits()
        if self.mode not in self._system_workspace:
            self._system_workspace[self.mode] = (
                np.empty((active.size, active.size), dtype=float),
                np.empty(active.size, dtype=float),
            )

    def set_center_of_mass_target(self, target: Sequence[float]) -> None:
        value = np.asarray(target, dtype=float).reshape(-1)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("center-of-mass target must contain three finite values")
        self._center_of_mass_target = value.copy()

    def set_zmp_target(self, target: Sequence[float]) -> None:
        value = np.asarray(target, dtype=float).reshape(-1)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("ZMP target must contain two finite XY values")
        self._zmp_target = value.copy()

    def set_center_of_mass_acceleration(self, acceleration: Sequence[float]) -> None:
        value = np.asarray(acceleration, dtype=float).reshape(-1)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(
                "center-of-mass acceleration must contain three finite values"
            )
        self._center_of_mass_acceleration = value.copy()

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
        started = perf_counter()
        desired_poses, frame_weights = self._prepare_targets(targets)
        initial_q = self._last_q.copy()
        initial_velocity = self._last_velocity.copy()
        try:
            return self._solve_seed(
                desired_poses,
                frame_weights,
                seed,
                max_iterations=max_iterations,
                enforce_acceleration=enforce_acceleration,
                started=started,
            )
        except Exception:
            self._last_q = initial_q
            self._last_velocity = initial_velocity
            raise

    def _prepare_targets(
        self,
        targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
    ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        normalized = self._normalize_targets(targets)
        if (
            self.support_polygon_task is not None
            and self.support_polygon_task.reference == "zmp"
            and self.zmp_task is None
        ):
            raise ValueError("ZMP support polygon reference requires zmp_task")
        if self.center_of_mass_task is not None and self._center_of_mass_target is None:
            raise ValueError("center_of_mass_task requires a center-of-mass target")
        if (
            self.zmp_task is not None
            and np.max(self.zmp_task.cost) > 0.0
            and self._zmp_target is None
        ):
            raise ValueError("zmp_task requires a ZMP target")
        desired_poses = {
            name: self.pin.SE3(target.pose[:3, :3], target.pose[:3, 3])
            for name, target in normalized.items()
        }
        frame_weights = {
            name: self.frame_tasks[name].cost * np.sqrt(target.weight)
            for name, target in normalized.items()
        }
        return desired_poses, frame_weights

    def _solve_seed(
        self,
        desired_poses: Mapping[str, object],
        frame_weights: Mapping[str, np.ndarray],
        seed: Optional[Sequence[float]],
        *,
        max_iterations: Optional[int],
        enforce_acceleration: bool,
        started: Optional[float] = None,
        collision_cost_cache: Optional[dict[bytes, float]] = None,
        collision_gradient_cache: Optional[dict[bytes, np.ndarray]] = None,
    ) -> RetargetingResult:
        q = self._configuration(seed) if seed is not None else self._last_q.copy()
        active, velocity_limits = self._mode_limits()
        started = perf_counter() if started is None else started
        regularization = self.damping
        accepted_steps = 0
        limit_hits = 0
        stagnant = 0
        termination_reason = "maximum_iterations"
        collision_evaluations = [0, 0]
        collision_cost_cache = (
            {} if collision_cost_cache is None else collision_cost_cache
        )
        collision_gradient_cache = (
            {} if collision_gradient_cache is None else collision_gradient_cache
        )

        if max_iterations is not None and (
            not isinstance(max_iterations, Integral) or isinstance(max_iterations, bool)
        ):
            raise TypeError("max_iterations must be an integer or None")
        iteration_limit = (
            self.max_iterations if max_iterations is None else int(max_iterations)
        )
        if iteration_limit < 1:
            raise ValueError("max_iterations must be positive")
        full_displacement = np.zeros(self.model.nv)
        for iteration in range(1, iteration_limit + 1):
            state = self._task_state(
                q,
                frame_weights,
                desired_poses,
                active,
                regularization,
                True,
                True,
                collision_evaluations,
                collision_cost_cache,
                collision_gradient_cache,
            )
            if self._converged(state):
                termination_reason = "converged"
                break
            if not active.size:
                termination_reason = "no_active_dofs"
                break
            lower, upper = self._displacement_bounds(
                q, active, velocity_limits, enforce_acceleration
            )
            unconstrained = self._linear_solve(state["hessian"], state["gradient"])
            displacement = self._solve_box_qp(
                state["hessian"],
                state["gradient"],
                lower,
                upper,
                unconstrained=unconstrained,
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
                full_displacement[active] = scale * displacement
                candidate = np.asarray(
                    self.pin.integrate(self.model, q, full_displacement)
                )
                candidate = self._project_limits(candidate)
                candidate_state = self._task_state(
                    candidate,
                    frame_weights,
                    desired_poses,
                    active,
                    regularization,
                    False,
                    False,
                    collision_evaluations,
                    collision_cost_cache,
                    collision_gradient_cache,
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
                previous_q = q
                q = candidate
                accepted_steps += 1
                if enforce_acceleration:
                    actual_velocity = (
                        np.asarray(
                            self.pin.difference(self.model, previous_q, q), dtype=float
                        )
                        / self.integration_dt
                    )
                    self._last_velocity.fill(0.0)
                    self._last_velocity[active] = actual_velocity[active]
                regularization = max(self.damping, regularization * 0.5)
                stagnant = (
                    stagnant + 1 if improvement <= self.stagnation_tolerance else 0
                )
            if stagnant >= self.stagnation_iterations:
                termination_reason = "stagnated"
                break

        state = self._task_state(
            q,
            frame_weights,
            desired_poses,
            active,
            regularization,
            False,
            True,
            collision_evaluations,
            collision_cost_cache,
            collision_gradient_cache,
        )
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
            objective=state["objective"],
            position_residual=state["position_residual"],
            orientation_residual=state["orientation_residual"],
            target_residuals=state["target_residuals"],
            termination_reason=termination_reason,
            accepted_steps=accepted_steps,
            limit_hits=limit_hits,
            collision_cost=state["collision_cost"],
            collision_evaluations=collision_evaluations[0],
            collision_gradient_evaluations=collision_evaluations[1],
            center_of_mass_residual=state["center_of_mass_residual"],
            support_polygon_violation=state["support_polygon_violation"],
            zmp_residual=state["zmp_residual"],
        )

    def _mode_limits(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mode not in self._active_cache:
            active = self._mode_plan().active_velocity_indices
            self._active_cache[self.mode] = active
            values = self._model_velocity_limits[active]
            self._velocity_limit_cache[self.mode] = np.where(
                np.isfinite(values) & (values > 0.0), values, np.inf
            )
        return self._active_cache[self.mode], self._velocity_limit_cache[self.mode]

    def _converged(self, state: dict) -> bool:
        return bool(
            state["position_convergence_residual"] <= self.position_tolerance
            and state["orientation_convergence_residual"] <= self.orientation_tolerance
            and state["collision_cost"] <= self.collision_tolerance
            and (
                self.center_of_mass_task is None
                or state["center_of_mass_convergence_residual"]
                <= self.center_of_mass_tolerance
            )
            and (
                self.support_polygon_task is None
                or self.support_polygon_task.cost == 0.0
                or state["support_polygon_violation"] <= self.support_polygon_tolerance
            )
            and (
                self.zmp_task is None
                or np.max(self.zmp_task.cost) == 0.0
                or state["zmp_convergence_residual"] <= self.zmp_tolerance
            )
        )

    def _task_state(
        self,
        q: np.ndarray,
        frame_weights: Mapping[str, np.ndarray],
        desired_poses: Mapping[str, object],
        active: np.ndarray,
        regularization: float,
        build_system: bool,
        compute_metrics: bool,
        collision_evaluations: list[int],
        collision_cost_cache: dict[bytes, float],
        collision_gradient_cache: dict[bytes, np.ndarray],
    ) -> dict:
        frame_jacobians_required = (
            build_system
            and active.size
            and any(np.max(weights) > 0.0 for weights in frame_weights.values())
        )
        center_jacobian_required = (
            build_system
            and active.size
            and (
                (
                    self.center_of_mass_task is not None
                    and np.max(self.center_of_mass_task.cost) > 0.0
                )
                or (
                    self.support_polygon_task is not None
                    and self.support_polygon_task.cost > 0.0
                )
                or (self.zmp_task is not None and np.max(self.zmp_task.cost) > 0.0)
            )
        )
        if build_system and (frame_jacobians_required or center_jacobian_required):
            # Build the model-wide joint Jacobians once, then extract every
            # active frame below. This matters for dual-arm and full-body
            # modes, where computeFrameJacobian would repeat the traversal for
            # each target.
            self.pin.computeJointJacobians(self.model, self.data, q)
        else:
            self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        hessian = None
        gradient = None
        if build_system:
            workspace = self._system_workspace.get(self.mode)
            if workspace is None or workspace[0].shape != (active.size, active.size):
                workspace = (
                    np.empty((active.size, active.size), dtype=float),
                    np.empty(active.size, dtype=float),
                )
                self._system_workspace[self.mode] = workspace
            hessian, gradient = workspace
            hessian.fill(0.0)
            gradient.fill(0.0)
            hessian.flat[:: active.size + 1] = regularization
        weighted_error_squared = 0.0
        target_residuals = {} if compute_metrics else None
        position_convergence_residual = 0.0
        orientation_convergence_residual = 0.0
        for name in self._mode_plan().targets:
            task = self.frame_tasks[name]
            desired = desired_poses[name]
            current = self.data.oMf[self._frame_ids[name]]
            error = np.asarray(self.pin.log6(current.inverse() * desired).vector)
            if compute_metrics:
                position_error = float(np.linalg.norm(error[:3]))
                orientation_error = float(np.linalg.norm(error[3:]))
                target_residuals[name] = (position_error, orientation_error)
                position_convergence_residual = max(
                    position_convergence_residual,
                    position_error
                    if np.min(task.position_cost) > 0.0
                    else float(np.linalg.norm(error[:3][task.position_cost > 0.0])),
                )
                orientation_convergence_residual = max(
                    orientation_convergence_residual,
                    orientation_error
                    if np.min(task.orientation_cost) > 0.0
                    else float(np.linalg.norm(error[3:][task.orientation_cost > 0.0])),
                )
            weights = frame_weights[name]
            weighted_error = weights * task.gain * error
            weighted_error_squared += float(weighted_error @ weighted_error)
            if build_system:
                if active.size and np.max(weights) > 0.0:
                    jacobian = np.asarray(
                        self.pin.getFrameJacobian(
                            self.model,
                            self.data,
                            self._frame_ids[name],
                            self.pin.ReferenceFrame.LOCAL,
                        ),
                        dtype=float,
                    ).reshape(6, self.model.nv)[:, active]
                    weighted_jacobian = weights[:, None] * jacobian
                    hessian += weighted_jacobian.T @ weighted_jacobian
                    gradient += weighted_jacobian.T @ weighted_error
                if task.lm_damping:
                    hessian.flat[:: active.size + 1] += task.lm_damping * float(
                        weighted_error @ weighted_error
                    )

        posture_objective = 0.0
        if self.posture_task.cost > 0.0:
            posture_error = np.asarray(
                self.pin.difference(self.model, q, self._posture_q), dtype=float
            )[active]
            weight = self.posture_task.cost
            weighted_posture_error = weight * posture_error
            posture_objective = 0.5 * float(
                weighted_posture_error @ weighted_posture_error
            )
            if build_system:
                hessian.flat[:: active.size + 1] += weight * weight
                gradient += weight * weight * self.posture_task.gain * posture_error

        current_center = None
        center_jacobian = None
        if (
            self.center_of_mass_task is not None
            or self.support_polygon_task is not None
            or self.zmp_task is not None
        ):
            if build_system and center_jacobian_required:
                center_jacobian = np.asarray(
                    self.pin.jacobianCenterOfMass(self.model, self.data, False),
                    dtype=float,
                ).reshape(3, self.model.nv)[:, active]
                # Joint placements and Jacobians are already current. The
                # no-q overload avoids repeating kinematics, while updating
                # data.com for the residual below.
                current_center = np.asarray(self.data.com[0], dtype=float).reshape(3)
            else:
                current_center = np.asarray(
                    self.pin.centerOfMass(self.model, self.data, False), dtype=float
                ).reshape(3)
                if build_system:
                    center_jacobian = np.zeros((3, 0))

        center_of_mass_residual = float("nan")
        center_of_mass_convergence_residual = float("nan")
        if self.center_of_mass_task is not None:
            center_error = self._center_of_mass_target - current_center
            if compute_metrics:
                center_of_mass_residual = float(np.linalg.norm(center_error))
                center_of_mass_convergence_residual = (
                    center_of_mass_residual
                    if np.min(self.center_of_mass_task.cost) > 0.0
                    else float(
                        np.linalg.norm(
                            center_error[self.center_of_mass_task.cost > 0.0]
                        )
                    )
                )
            center_weights = self.center_of_mass_task.cost
            weighted_center_error = (
                center_weights * self.center_of_mass_task.gain * center_error
            )
            weighted_error_squared += float(
                weighted_center_error @ weighted_center_error
            )
            if build_system:
                if np.max(center_weights) > 0.0:
                    weighted_center_jacobian = center_weights[:, None] * center_jacobian
                    hessian += weighted_center_jacobian.T @ weighted_center_jacobian
                    gradient += weighted_center_jacobian.T @ weighted_center_error
                if self.center_of_mass_task.lm_damping:
                    hessian.flat[:: active.size + 1] += (
                        self.center_of_mass_task.lm_damping
                        * float(weighted_center_error @ weighted_center_error)
                    )

        zmp_residual = float("nan")
        zmp_convergence_residual = float("nan")
        current_zmp = None
        zmp_jacobian = None
        if self.zmp_task is not None:
            task = self.zmp_task
            acceleration_scale = self._center_of_mass_acceleration[:2] / task.gravity
            current_zmp = (
                current_center[:2]
                - (current_center[2] - task.plane_height) * acceleration_scale
            )
            if build_system:
                zmp_jacobian = center_jacobian[:2] - np.outer(
                    acceleration_scale, center_jacobian[2]
                )
            if self._zmp_target is not None:
                zmp_error = self._zmp_target - current_zmp
                if compute_metrics:
                    zmp_residual = float(np.linalg.norm(zmp_error))
                    zmp_convergence_residual = (
                        zmp_residual
                        if np.min(task.cost) > 0.0
                        else float(np.linalg.norm(zmp_error[task.cost > 0.0]))
                    )
                weighted_zmp_error = task.cost * task.gain * zmp_error
                weighted_error_squared += float(weighted_zmp_error @ weighted_zmp_error)
                if build_system:
                    if np.max(task.cost) > 0.0:
                        weighted_zmp_jacobian = task.cost[:, None] * zmp_jacobian
                        hessian += weighted_zmp_jacobian.T @ weighted_zmp_jacobian
                        gradient += weighted_zmp_jacobian.T @ weighted_zmp_error
                    if task.lm_damping:
                        hessian.flat[:: active.size + 1] += task.lm_damping * float(
                            weighted_zmp_error @ weighted_zmp_error
                        )

        support_polygon_violation = 0.0
        if self.support_polygon_task is not None:
            task = self.support_polygon_task
            support_point = (
                current_zmp if task.reference == "zmp" else current_center[:2]
            )
            support_point_jacobian = None
            if build_system:
                support_point_jacobian = (
                    zmp_jacobian if task.reference == "zmp" else center_jacobian[:2]
                )
            signed_distances = task.normals @ support_point - task.offsets
            violations = np.maximum(0.0, task.margin - signed_distances)
            if compute_metrics:
                support_polygon_violation = float(np.max(violations))
            weighted_violations = task.cost * task.gain * violations
            weighted_error_squared += float(weighted_violations @ weighted_violations)
            if build_system and task.cost > 0.0 and np.any(violations > 0.0):
                active_edges = violations > 0.0
                support_jacobian = (
                    task.cost * task.normals[active_edges] @ support_point_jacobian
                )
                hessian += support_jacobian.T @ support_jacobian
                gradient += support_jacobian.T @ weighted_violations[active_edges]

        if (
            build_system
            and self.collision_cost_weight > 0.0
            and self.collision_cost_gradient is not None
        ):
            self._collision_cost_gradient_value(
                q,
                active,
                collision_evaluations,
                collision_cost_cache,
                collision_gradient_cache,
            )
        collision_cost = self._collision_cost_value(
            q, collision_evaluations, collision_cost_cache
        )
        collision_objective = self.collision_cost_weight * collision_cost
        if not np.isfinite(collision_objective):
            raise ValueError("weighted collision cost must be finite")
        if build_system and collision_objective > 0.0:
            with np.errstate(over="ignore", invalid="ignore"):
                collision_update = (
                    self.collision_cost_weight
                    * self._collision_gradient_value(
                        q,
                        collision_cost,
                        active,
                        collision_evaluations,
                        collision_cost_cache,
                        collision_gradient_cache,
                    )
                )
            if not np.isfinite(collision_update).all():
                raise ValueError("weighted collision gradient must be finite")
            gradient -= collision_update

        objective = (
            0.5 * weighted_error_squared + posture_objective + collision_objective
        )
        if not np.isfinite(objective):
            raise ValueError("retargeting objective must be finite")
        if not compute_metrics:
            return {"objective": objective}
        position_residual = max(value[0] for value in target_residuals.values())
        orientation_residual = max(value[1] for value in target_residuals.values())
        if build_system and (
            not np.isfinite(hessian).all() or not np.isfinite(gradient).all()
        ):
            raise ValueError("retargeting linear system must be finite")
        return {
            "hessian": hessian,
            "gradient": gradient,
            "objective": objective,
            "residual": float(np.sqrt(weighted_error_squared)),
            "position_residual": position_residual,
            "orientation_residual": orientation_residual,
            "position_convergence_residual": position_convergence_residual,
            "orientation_convergence_residual": orientation_convergence_residual,
            "target_residuals": target_residuals,
            "collision_cost": collision_cost,
            "center_of_mass_residual": center_of_mass_residual,
            "center_of_mass_convergence_residual": center_of_mass_convergence_residual,
            "support_polygon_violation": support_polygon_violation,
            "zmp_residual": zmp_residual,
            "zmp_convergence_residual": zmp_convergence_residual,
        }

    def _collision_cost_value(
        self,
        q: np.ndarray,
        evaluations: list[int],
        cache: Optional[dict[bytes, float]] = None,
    ) -> float:
        if self.collision_cost is None or self.collision_cost_weight == 0.0:
            return 0.0
        key = np.asarray(q, dtype=float).tobytes()
        if cache is not None and key in cache:
            return cache[key]
        evaluations[0] += 1
        value = float(self.collision_cost(q.copy()))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("collision_cost must return a finite non-negative value")
        if cache is not None:
            cache[key] = value
        return value

    def _collision_gradient_value(
        self,
        q: np.ndarray,
        current_cost: float,
        active: np.ndarray,
        evaluations: list[int],
        cache: Optional[dict[bytes, float]] = None,
        gradient_cache: Optional[dict[bytes, np.ndarray]] = None,
    ) -> np.ndarray:
        key = np.asarray(q, dtype=float).tobytes()
        if gradient_cache is not None and key in gradient_cache:
            return gradient_cache[key]
        if self.collision_cost_gradient is not None:
            return self._collision_cost_gradient_value(
                q, active, evaluations, cache, gradient_cache
            )[1]
        evaluations[1] += 1
        if self.collision_gradient is not None:
            value = np.asarray(self.collision_gradient(q.copy()), dtype=float).reshape(
                -1
            )
            if value.shape != (self.model.nv,) or not np.isfinite(value).all():
                raise ValueError(
                    f"collision_gradient must return {self.model.nv} finite values"
                )
            result = value[active].copy()
            if gradient_cache is not None:
                gradient_cache[key] = result
            return result

        gradient = np.zeros(active.size)
        step = self.collision_finite_difference_step
        tangent = np.zeros(self.model.nv)
        for output_index, velocity_index in enumerate(active):
            tangent[velocity_index] = step
            positive = self._project_limits(
                np.asarray(self.pin.integrate(self.model, q, tangent), dtype=float)
            )
            tangent[velocity_index] = -step
            negative = self._project_limits(
                np.asarray(self.pin.integrate(self.model, q, tangent), dtype=float)
            )
            tangent[velocity_index] = 0.0
            positive_delta = np.asarray(
                self.pin.difference(self.model, q, positive), dtype=float
            )[velocity_index]
            negative_delta = np.asarray(
                self.pin.difference(self.model, negative, q), dtype=float
            )[velocity_index]
            span = positive_delta + negative_delta
            if span <= 1e-15:
                continue
            positive_cost = (
                current_cost
                if positive_delta <= 1e-15
                else self._collision_cost_value(positive, evaluations, cache)
            )
            negative_cost = (
                current_cost
                if negative_delta <= 1e-15
                else self._collision_cost_value(negative, evaluations, cache)
            )
            gradient[output_index] = (positive_cost - negative_cost) / span
        if gradient_cache is not None:
            gradient_cache[key] = gradient
        return gradient

    def _collision_cost_gradient_value(
        self,
        q: np.ndarray,
        active: np.ndarray,
        evaluations: list[int],
        cost_cache: Optional[dict[bytes, float]],
        gradient_cache: Optional[dict[bytes, np.ndarray]],
    ) -> tuple[float, np.ndarray]:
        key = np.asarray(q, dtype=float).tobytes()
        if (
            cost_cache is not None
            and gradient_cache is not None
            and key in cost_cache
            and key in gradient_cache
        ):
            return cost_cache[key], gradient_cache[key]
        evaluations[0] += 1
        evaluations[1] += 1
        cost, gradient = self.collision_cost_gradient(q.copy())
        cost = float(cost)
        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError(
                "collision_cost_gradient must return a finite non-negative cost"
            )
        if gradient.shape != (self.model.nv,) or not np.isfinite(gradient).all():
            raise ValueError(
                "collision_cost_gradient must return a gradient with "
                f"{self.model.nv} finite values"
            )
        active_gradient = gradient[active].copy()
        if cost_cache is not None:
            cost_cache[key] = cost
        if gradient_cache is not None:
            gradient_cache[key] = active_gradient
        return cost, active_gradient

    @classmethod
    def _solve_box_qp(
        cls,
        hessian: np.ndarray,
        gradient: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        unconstrained: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve a positive-definite QP with projected-gradient refinement."""

        if unconstrained is None:
            unconstrained = cls._linear_solve(hessian, gradient)
        if np.all(unconstrained >= lower) and np.all(unconstrained <= upper):
            return unconstrained
        solution = np.clip(unconstrained, lower, upper)
        # status: -1 at lower bound, +1 at upper bound, 0 free. Solving the
        # reduced positive-definite system gives the exact minimizer for one
        # active set; violated bounds are activated and KKT-violating bounds
        # are released until the global box optimum is reached.
        status = np.zeros(solution.size, dtype=np.int8)
        status[unconstrained < lower] = -1
        status[unconstrained > upper] = 1
        scale = max(1.0, float(np.max(np.abs(gradient))))
        kkt_tolerance = 1e-10 * scale
        for _ in range(10 * solution.size + 10):
            free = status == 0
            active = ~free
            if np.any(free):
                rhs = gradient[free]
                if np.any(active):
                    rhs = rhs - hessian[np.ix_(free, active)] @ solution[active]
                free_solution = cls._linear_solve(hessian[np.ix_(free, free)], rhs)
                free_indices = np.flatnonzero(free)
                below = free_solution < lower[free_indices]
                above = free_solution > upper[free_indices]
                if np.any(below | above):
                    solution[free_indices] = np.clip(
                        free_solution, lower[free_indices], upper[free_indices]
                    )
                    status[free_indices[below]] = -1
                    status[free_indices[above]] = 1
                    continue
                solution[free_indices] = free_solution

            derivative = hessian @ solution - gradient
            lower_violation = (status == -1) & (derivative < -kkt_tolerance)
            upper_violation = (status == 1) & (derivative > kkt_tolerance)
            if not np.any(lower_violation | upper_violation):
                break
            violations = np.zeros(solution.size)
            violations[lower_violation] = -derivative[lower_violation]
            violations[upper_violation] = derivative[upper_violation]
            status[int(np.argmax(violations))] = 0
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
            position_lower = self._lower_position_limits[active] - q[active]
            position_upper = self._upper_position_limits[active] - q[active]
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
