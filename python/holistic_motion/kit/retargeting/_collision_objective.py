"""Collision objectives and manifold-aware numerical derivatives."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..._validation import real_array, require_real


def collision_cost_value(
    solver,
    q: np.ndarray,
    evaluations: list[int],
    cache: Optional[dict[bytes, float]] = None,
) -> float:
    if solver.collision_cost is None or solver.collision_cost_weight == 0.0:
        return 0.0
    key = np.asarray(q, dtype=float).tobytes()
    if cache is not None and key in cache:
        return cache[key]
    evaluations[0] += 1
    value = solver.collision_cost(q.copy())
    require_real(value, name="collision_cost result")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("collision_cost must return a finite non-negative value")
    if cache is not None:
        cache[key] = value
    return value


def collision_gradient_value(
    solver,
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
    if solver.collision_cost_gradient is not None:
        return solver._collision_cost_gradient_value(
            q, active, evaluations, cache, gradient_cache
        )[1]
    evaluations[1] += 1
    if solver.collision_gradient is not None:
        value = real_array(
            solver.collision_gradient(q.copy()), name="collision_gradient result"
        ).reshape(-1)
        if value.shape != (solver.model.nv,) or not np.isfinite(value).all():
            raise ValueError(
                f"collision_gradient must return {solver.model.nv} finite values"
            )
        result = value[active].copy()
        if gradient_cache is not None:
            gradient_cache[key] = result
        return result

    gradient = np.zeros(active.size)
    step = solver.collision_finite_difference_step
    relative_sample_tolerance = np.sqrt(np.finfo(float).eps)
    tangent = np.zeros(solver.model.nv)
    mode_active, velocity_limits = solver._mode_limits()
    fixed_velocity = set(mode_active[velocity_limits == 0.0])
    for output_index, velocity_index in enumerate(active):
        position_index = solver._scalar_position_indices[velocity_index]
        if velocity_index in fixed_velocity or (
            position_index >= 0
            and solver._lower_position_limits[position_index]
            == solver._upper_position_limits[position_index]
        ):
            # This QP coordinate cannot move. Its objective derivative
            # cannot change the constrained optimum, so do not integrate
            # or query costs for samples that the solver will never use.
            continue
        tangent[velocity_index] = step
        positive = solver._project_limits(
            np.asarray(solver.pin.integrate(solver.model, q, tangent), dtype=float)
        )
        tangent[velocity_index] = -step
        negative = solver._project_limits(
            np.asarray(solver.pin.integrate(solver.model, q, tangent), dtype=float)
        )
        tangent[velocity_index] = 0.0
        positive_delta = np.asarray(
            solver.pin.difference(solver.model, q, positive), dtype=float
        )[velocity_index]
        negative_delta = np.asarray(
            solver.pin.difference(solver.model, negative, q), dtype=float
        )[velocity_index]
        if positive_delta > 0.0 and negative_delta > 0.0:
            # A nearly collapsed side makes the nonuniform three-point
            # formula amplify cost roundoff. Compare sample distances,
            # not an absolute joint-unit threshold, and use the longer
            # one-sided difference when their scales are far apart.
            if positive_delta < relative_sample_tolerance * negative_delta:
                positive_delta = 0.0
            elif negative_delta < relative_sample_tolerance * positive_delta:
                negative_delta = 0.0
        span = positive_delta + negative_delta
        # A fixed absolute epsilon discards valid small-scale motion.
        # Integration/difference already reveals whether the sample is
        # representably distinct from q, including at a projected bound.
        if span <= 0.0:
            continue
        positive_cost = (
            current_cost
            if positive_delta <= 0.0
            else solver._collision_cost_value(positive, evaluations, cache)
        )
        negative_cost = (
            current_cost
            if negative_delta <= 0.0
            else solver._collision_cost_value(negative, evaluations, cache)
        )
        if (
            positive_delta > 0.0
            and negative_delta > 0.0
            and positive_delta != negative_delta
        ):
            # Different distances after limit projection require the
            # derivative of the three-point interpolant at q. A plain
            # secant instead estimates the derivative at the samples'
            # midpoint and can point uphill near a bound.
            forward = (positive_cost - current_cost) / positive_delta
            backward = (current_cost - negative_cost) / negative_delta
            gradient[output_index] = (negative_delta / span) * forward + (
                positive_delta / span
            ) * backward
        else:
            # Preserve symmetric cancellation and one-sided differences
            # at a bound, including reuse of the current point's cost.
            gradient[output_index] = (positive_cost - negative_cost) / span
    if gradient_cache is not None:
        gradient_cache[key] = gradient
    return gradient


def collision_cost_gradient_value(
    solver,
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
    cost, gradient = solver.collision_cost_gradient(q.copy())
    require_real(cost, name="collision_cost_gradient cost")
    cost = float(cost)
    gradient = real_array(gradient, name="collision_cost_gradient gradient").reshape(-1)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(
            "collision_cost_gradient must return a finite non-negative cost"
        )
    if gradient.shape != (solver.model.nv,) or not np.isfinite(gradient).all():
        raise ValueError(
            "collision_cost_gradient must return a gradient with "
            f"{solver.model.nv} finite values"
        )
    active_gradient = gradient[active].copy()
    if cost_cache is not None:
        cost_cache[key] = cost
    if gradient_cache is not None:
        gradient_cache[key] = active_gradient
    return cost, active_gradient
