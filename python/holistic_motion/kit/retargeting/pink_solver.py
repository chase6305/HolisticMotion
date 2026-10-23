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

from ..._validation import real_array
from ._box_qp import solve_box_qp as _solve_bounded_qp
from ._collision_objective import (
    collision_cost_gradient_value as _evaluate_collision_cost_gradient,
)
from ._collision_objective import (
    collision_cost_value as _evaluate_collision_cost,
)
from ._collision_objective import (
    collision_gradient_value as _evaluate_collision_gradient,
)
from ._task_objective import task_state as _assemble_task_state
from .modes import RetargetingMode
from .pinocchio_solver import (
    PinocchioRetargetingSolver,
    RetargetingResult,
    RetargetingTarget,
)
from .tasks import (
    CenterOfMassTask,
    FrameTask,
    PostureTask,
    SupportPolygonTask,
    ZmpTask,
    _joint_costs,
)


class PinkRetargetingSolver(PinocchioRetargetingSolver):
    """Weighted task-space differential IK with posture and velocity limits.

    Each worker owns its solver, Pinocchio workspace and history. Collision
    callbacks must follow the same ownership contract as their captured state.
    """

    def __init__(
        self,
        *args,
        frame_tasks: Optional[Mapping[str, FrameTask]] = None,
        posture_task: Optional[PostureTask] = None,
        joint_motion_costs: Optional[Mapping[str, float]] = None,
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
        self._resolved_posture_task = None
        self._posture_weights()
        self._joint_motion_weights = self._resolve_joint_costs(
            {} if joint_motion_costs is None else joint_motion_costs,
            0.0,
            "joint motion costs",
        )
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
            and self.support_polygon_task.cost > 0.0
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
        # A continuous/floating joint changes nq relative to nv for the whole
        # model. Scalar arm limits must still constrain their own QP variables.
        self._scalar_position_indices = np.full(self.model.nv, -1, dtype=int)
        for joint_id in range(1, self.model.njoints):
            joint = self.model.joints[joint_id]
            if joint.nq == 1 and joint.nv == 1:
                self._scalar_position_indices[joint.idx_v] = joint.idx_q
        self._scalar_position_indices.setflags(write=False)

    def set_posture_target(self, configuration: Sequence[float]) -> None:
        self._posture_q = self._configuration(configuration)

    def _resolve_joint_costs(self, costs, default: float, name: str) -> np.ndarray:
        costs = _joint_costs(costs, name)
        weights = np.full(self.model.nv, default, dtype=float)
        for joint_name, cost in costs.items():
            if joint_name not in self.model.names:
                raise ValueError(f"{name} references unknown joint {joint_name!r}")
            joint_id = self.model.getJointId(joint_name)
            joint = self.model.joints[joint_id]
            if joint_id == 0 or joint.nv == 0:
                raise ValueError(f"{name} requires movable joint {joint_name!r}")
            weights[joint.idx_v : joint.idx_v + joint.nv] = cost
        with np.errstate(over="ignore"):
            squared = weights * weights
        if not np.isfinite(squared).all():
            raise ValueError(f"{name} squared weights must be finite")
        weights.setflags(write=False)
        return weights

    def _posture_weights(self) -> np.ndarray:
        task = self.posture_task
        if task is not self._resolved_posture_task:
            if not isinstance(task, PostureTask):
                raise TypeError("posture_task must be PostureTask")
            weights = self._resolve_joint_costs(
                task.joint_costs, task.cost, "posture joint costs"
            )
            self._posture_joint_weights = weights
            self._resolved_posture_task = task
        return self._posture_joint_weights

    def prepare(self, mode: Optional[Union[RetargetingMode, str]] = None) -> None:
        """Prepare mode indices, limits, and numerical workspaces."""

        self._posture_weights()
        previous_mode = self.mode
        try:
            super().prepare(mode)
            active, _ = self._mode_limits()
            if self.mode not in self._system_workspace:
                self._system_workspace[self.mode] = (
                    np.empty((active.size, active.size), dtype=float),
                    np.empty(active.size, dtype=float),
                )
        except BaseException:
            self.set_mode(previous_mode)
            raise

    def set_center_of_mass_target(self, target: Sequence[float]) -> None:
        value = real_array(target, name="center-of-mass target").reshape(-1)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("center-of-mass target must contain three finite values")
        self._center_of_mass_target = value.copy()

    def set_zmp_target(self, target: Sequence[float]) -> None:
        value = real_array(target, name="ZMP target").reshape(-1)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("ZMP target must contain two finite XY values")
        self._zmp_target = value.copy()

    def set_center_of_mass_acceleration(self, acceleration: Sequence[float]) -> None:
        value = real_array(acceleration, name="center-of-mass acceleration").reshape(-1)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(
                "center-of-mass acceleration must contain three finite values"
            )
        self._center_of_mass_acceleration = value.copy()

    def reset(self, configuration: Optional[Sequence[float]] = None) -> None:
        super().reset(configuration)
        self._last_velocity.fill(0.0)

    def _snapshot_solve_state(self) -> dict:
        """Snapshot persistent solve history, including subclass diagnostics."""

        return {
            "configuration": self._last_q.copy(),
            "velocity": self._last_velocity.copy(),
        }

    def _restore_solve_state(self, state: dict) -> None:
        self._last_q = state["configuration"]
        self._last_velocity = state["velocity"]

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
        """Solve offline, or return one physical-cycle update when constrained.

        ``enforce_acceleration=True`` caps the iteration budget at one so the
        returned displacement and velocity history describe the same interval.
        ``step()`` is the convenience entry point for this single-cycle mode.
        Exceptions and interrupts restore solve history before propagating.
        """

        started = perf_counter()
        desired_poses, frame_weights = self._prepare_targets(targets)
        initial_state = self._snapshot_solve_state()
        try:
            return self._solve_seed(
                desired_poses,
                frame_weights,
                seed,
                max_iterations=max_iterations,
                enforce_acceleration=enforce_acceleration,
                started=started,
            )
        except BaseException:
            # A cancelled cycle must not leave an unreturned velocity update.
            self._restore_solve_state(initial_state)
            raise

    def _prepare_targets(
        self,
        targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
    ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        normalized = self._normalize_targets(targets)
        if (
            self.support_polygon_task is not None
            and self.support_polygon_task.cost > 0.0
            and self.support_polygon_task.reference == "zmp"
            and self.zmp_task is None
        ):
            raise ValueError("ZMP support polygon reference requires zmp_task")
        if (
            self.center_of_mass_task is not None
            and np.max(self.center_of_mass_task.cost) > 0.0
            and self._center_of_mass_target is None
        ):
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
        if enforce_acceleration:
            # Each accepted update consumes one integration_dt. Combining
            # several here would return a multi-cycle displacement as a single
            # command, while retaining only the last substep's velocity.
            iteration_limit = 1
        full_displacement = np.zeros(self.model.nv)
        final_state = None
        for iteration in range(1, iteration_limit + 1):
            bounds = (
                self._displacement_bounds(
                    q, active, velocity_limits, enforce_acceleration
                )
                if active.size
                else None
            )
            can_move = bounds is not None and bool(
                np.any(bounds[0] != 0.0) or np.any(bounds[1] != 0.0)
            )
            state = self._task_state(
                q,
                frame_weights,
                desired_poses,
                active,
                regularization,
                can_move,
                True,
                collision_evaluations,
                collision_cost_cache,
                collision_gradient_cache,
            )
            can_hold = bounds is None or (
                np.all(bounds[0] <= 0.0) and np.all(bounds[1] >= 0.0)
            )
            # Pose convergence alone does not permit an instantaneous stop.
            # If zero displacement is infeasible, continue through the QP to
            # brake within the active joints' acceleration/position bounds.
            if self._converged(state) and can_hold:
                if enforce_acceleration:
                    self._last_velocity.fill(0.0)
                termination_reason = "converged"
                final_state = state
                break
            if not active.size:
                if enforce_acceleration:
                    self._last_velocity.fill(0.0)
                termination_reason = "no_active_dofs"
                final_state = state
                break
            if not can_move:
                # A box containing only zero displacement needs diagnostics,
                # not derivatives or repeated attempts to solve the same QP.
                if enforce_acceleration:
                    self._last_velocity.fill(0.0)
                termination_reason = "no_feasible_motion"
                final_state = state
                break
            lower, upper = bounds
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
            initial_scale = 1.0 if enforce_acceleration else self.step_size
            if initial_scale > 1.0:
                # Over-relaxation must stay inside the QP box. Cap one shared
                # scale to preserve the coupled search direction, then backtrack
                # from that feasible step rather than repeatedly clipping it.
                positive = displacement > 0.0
                negative = displacement < 0.0
                with np.errstate(over="ignore"):
                    if np.any(positive):
                        initial_scale = min(
                            initial_scale,
                            float(np.min(upper[positive] / displacement[positive])),
                        )
                    if np.any(negative):
                        initial_scale = min(
                            initial_scale,
                            float(np.min(lower[negative] / displacement[negative])),
                        )
            for backtrack in range(backtrack_count + 1):
                scale = initial_scale * (0.5**backtrack)
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

        state = final_state
        if state is None:
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
            values = self._model_velocity_limits[active]
            if np.any(np.isnan(values)) or np.any(values < 0.0):
                raise ValueError(
                    "joint velocity limits must be non-negative and not NaN"
                )
            values.setflags(write=False)
            # Zero is a fixed joint, while positive infinity is unbounded.
            # Publish both caches only after validation has succeeded.
            self._velocity_limit_cache[self.mode] = values
            self._active_cache[self.mode] = active
        return self._active_cache[self.mode], self._velocity_limit_cache[self.mode]

    def _converged(self, state: dict) -> bool:
        return bool(
            state["position_convergence_residual"] <= self.position_tolerance
            and state["orientation_convergence_residual"] <= self.orientation_tolerance
            and state["collision_cost"] <= self.collision_tolerance
            and (
                self.center_of_mass_task is None
                or np.max(self.center_of_mass_task.cost) == 0.0
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
        return _assemble_task_state(
            self,
            q,
            frame_weights,
            desired_poses,
            active,
            regularization,
            build_system,
            compute_metrics,
            collision_evaluations,
            collision_cost_cache,
            collision_gradient_cache,
        )

    def _collision_cost_value(
        self,
        q: np.ndarray,
        evaluations: list[int],
        cache: Optional[dict[bytes, float]] = None,
    ) -> float:
        return _evaluate_collision_cost(self, q, evaluations, cache)

    def _collision_gradient_value(
        self,
        q: np.ndarray,
        current_cost: float,
        active: np.ndarray,
        evaluations: list[int],
        cache: Optional[dict[bytes, float]] = None,
        gradient_cache: Optional[dict[bytes, np.ndarray]] = None,
    ) -> np.ndarray:
        return _evaluate_collision_gradient(
            self, q, current_cost, active, evaluations, cache, gradient_cache
        )

    def _collision_cost_gradient_value(
        self,
        q: np.ndarray,
        active: np.ndarray,
        evaluations: list[int],
        cost_cache: Optional[dict[bytes, float]],
        gradient_cache: Optional[dict[bytes, np.ndarray]],
    ) -> tuple[float, np.ndarray]:
        return _evaluate_collision_cost_gradient(
            self, q, active, evaluations, cost_cache, gradient_cache
        )

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
        return _solve_bounded_qp(
            hessian,
            gradient,
            lower,
            upper,
            unconstrained=unconstrained,
            linear_solve=cls._linear_solve,
        )

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
        else:
            position_indices = self._scalar_position_indices[active]
            scalar = position_indices >= 0
            indices = position_indices[scalar]
            position_lower = np.full(active.size, -np.inf)
            position_upper = np.full(active.size, np.inf)
            position_lower[scalar] = self._lower_position_limits[indices] - q[indices]
            position_upper[scalar] = self._upper_position_limits[indices] - q[indices]
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
                joint_id = self.model.getJointId(name)
                joint = self.model.joints[joint_id]
                if joint_id == 0 or joint.nv == 0:
                    raise ValueError(
                        f"acceleration limits require movable joint {name!r}"
                    )
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
