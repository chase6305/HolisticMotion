"""Dependency-free TOPPRA-style path time parametrization.

The implementation uses the standard path dynamics ``x = s_dot**2`` and
``x[i+1] = x[i] + 2 * ds[i] * u[i]``.  A backward controllable-set pass is
followed by a greedy forward pass.  Conservative polynomial bounds enforce
joint velocity and acceleration limits throughout each path interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import numpy as np


class _NaturalCubicPath:
    """Small vector-valued natural cubic spline with analytic derivatives."""

    def __init__(self, grid: np.ndarray, values: np.ndarray) -> None:
        self.grid = grid
        count = grid.size
        h = np.diff(grid)
        slopes = np.diff(values, axis=0) / h[:, None]
        second = np.zeros_like(values)
        if count > 2:
            # Natural boundaries set the endpoint second derivatives to zero.
            # The interior system is strictly diagonally dominant and
            # tridiagonal, so elimination needs O(count * DOF) work and storage.
            diagonal = 2.0 * (h[:-1] + h[1:])
            rhs = 6.0 * np.diff(slopes, axis=0)
            for index in range(1, count - 2):
                factor = h[index] / diagonal[index - 1]
                diagonal[index] -= factor * h[index]
                rhs[index] -= factor * rhs[index - 1]
            second[-2] = rhs[-1] / diagonal[-1]
            for index in range(count - 4, -1, -1):
                second[index + 1] = (
                    rhs[index] - h[index + 1] * second[index + 2]
                ) / diagonal[index]
        self.a = values[:-1].copy()
        self.b = slopes - h[:, None] * (2.0 * second[:-1] + second[1:]) / 6.0
        self.c = second[:-1] / 2.0
        self.d = np.diff(second, axis=0) / (6.0 * h[:, None])

    def evaluate(self, value: np.ndarray, order: int = 0) -> np.ndarray:
        index, delta = self._segments(value)
        if order == 0:
            return self.a[index] + delta * (
                self.b[index] + delta * (self.c[index] + delta * self.d[index])
            )
        if order == 1:
            return self.b[index] + delta * (
                2.0 * self.c[index] + 3.0 * delta * self.d[index]
            )
        if order == 2:
            return 2.0 * self.c[index] + 6.0 * delta * self.d[index]
        raise ValueError("spline derivative order must be 0, 1, or 2")

    def evaluate_all(
        self, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate position and both derivatives with one segment lookup."""

        index, delta = self._segments(value)
        a = self.a[index]
        b = self.b[index]
        c = self.c[index]
        d = self.d[index]
        position = a + delta * (b + delta * (c + delta * d))
        first = b + delta * (2.0 * c + 3.0 * delta * d)
        second = 2.0 * c + 6.0 * delta * d
        return position, first, second

    def _segments(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(value, dtype=float).reshape(-1)
        index = np.clip(
            np.searchsorted(self.grid, query, side="right") - 1,
            0,
            self.grid.size - 2,
        )
        delta = (query - self.grid[index])[:, None]
        return index, delta


def _positive_vector(value: Sequence[float], dof: int, name: str) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True).reshape(-1)
    if result.shape != (dof,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of size {dof}")
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return result


def _intersect_bounds(
    coefficients: np.ndarray, rhs: np.ndarray, lower: float, upper: float
) -> tuple[float, float]:
    """Intersect a scalar interval with ``coefficients * value <= rhs``."""

    positive = coefficients > 0.0
    negative = coefficients < 0.0
    if np.any((coefficients == 0.0) & (rhs < -1e-12)):
        return 1.0, 0.0
    lower = max(
        lower, float(np.max(rhs[negative] / coefficients[negative], initial=-np.inf))
    )
    upper = min(
        upper, float(np.min(rhs[positive] / coefficients[positive], initial=np.inf))
    )
    return lower, upper


class _IntervalConstraints:
    """Half-planes ``a * x + b * y <= 1`` for adjacent squared speeds."""

    def __init__(self, a: np.ndarray, b: np.ndarray) -> None:
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("path speed constraints must be finite")
        self.a, self.b = a, b
        self.positive = b > 0.0
        self.negative = b < 0.0
        # Eliminate y by comparing each lower bound on y with each upper
        # bound. These pairs do not depend on the next controllable interval,
        # so compute their cap on x once, without iterative feasibility probes.
        coefficients = (
            a[self.negative, None] * b[self.positive]
            - b[self.negative, None] * a[self.positive]
        )
        rhs = b[self.positive] - b[self.negative, None]
        _, self.cap = _intersect_bounds(coefficients, rhs, 0.0, np.inf)

    def project(self, lower: float, upper: float) -> tuple[float, float]:
        """All x >= 0 that can reach some y in [lower, upper]."""

        rhs = np.ones_like(self.a)
        rhs[self.positive] -= self.b[self.positive] * lower
        rhs[self.negative] -= self.b[self.negative] * upper
        return _intersect_bounds(self.a, rhs, 0.0, self.cap)

    def reachable(self, x: float, lower: float, upper: float) -> tuple[float, float]:
        product = self.a * x
        # At a projected boundary, 1 - a*x can lose its significant digits.
        # Dividing that residual by a near-zero b otherwise amplifies a few
        # rounding bits into a spurious empty interval. Apply the tolerance to
        # the dimensionless half-plane residual before dividing, not to the
        # resulting speed bounds; keep small nonzero coefficients intact.
        rounding = 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(product))
        return _intersect_bounds(self.b, 1.0 - product + rounding, lower, upper)


@dataclass(frozen=True)
class ToppraResult:
    """Discrete path parametrization returned by :func:`retime_path`."""

    gridpoints: np.ndarray
    path_speeds: np.ndarray
    path_accelerations: np.ndarray
    times: np.ndarray
    duration: float

    def __post_init__(self) -> None:
        arrays = (
            "gridpoints",
            "path_speeds",
            "path_accelerations",
            "times",
        )
        for name in arrays:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite one-dimensional array")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        count = self.gridpoints.size
        if (
            count < 2
            or self.path_speeds.size != count
            or self.times.size != count
            or self.path_accelerations.size != count - 1
        ):
            raise ValueError("TOPPRA result arrays have inconsistent lengths")
        if not np.isfinite(self.duration) or self.duration < 0.0:
            raise ValueError("TOPPRA result duration must be finite and non-negative")
        if np.any(np.diff(self.gridpoints) <= 0.0):
            raise ValueError("TOPPRA result gridpoints must increase strictly")
        if np.any(self.path_speeds < 0.0):
            raise ValueError("TOPPRA result path speeds must be non-negative")
        if abs(self.times[0]) > 1e-12 or np.any(np.diff(self.times) <= 0.0):
            raise ValueError(
                "TOPPRA result times must start at zero and increase strictly"
            )
        if not np.isclose(self.duration, self.times[-1], rtol=1e-12, atol=1e-12):
            raise ValueError("TOPPRA result duration must match its final time")
        with np.errstate(over="ignore", invalid="ignore"):
            speed_squared = self.path_speeds**2
            change = 2.0 * np.diff(self.gridpoints) * self.path_accelerations
        if not np.isfinite(speed_squared).all() or not np.isfinite(change).all():
            raise ValueError("TOPPRA result path dynamics must be finite")
        # Near a stop, x[i] and 2*ds*u cancel. Judge the residual against all
        # participating terms, not just x[i+1], which can be exactly zero.
        scale = np.maximum(
            np.maximum(speed_squared[:-1], speed_squared[1:]), np.abs(change)
        )
        if np.any(np.abs(np.diff(speed_squared) - change) > 1e-10 + 1e-8 * scale):
            raise ValueError("TOPPRA result violates interval path dynamics")
        distances = (
            0.5 * (self.path_speeds[:-1] + self.path_speeds[1:]) * np.diff(self.times)
        )
        if not np.allclose(np.diff(self.gridpoints), distances, rtol=1e-8, atol=1e-12):
            raise ValueError("TOPPRA result timing is inconsistent with path speeds")


class ToppraTrajectory:
    """Reachability-based timing of a joint-space waypoint path.

    Waypoints are interpolated along normalized chord length.  ``gridpoints``
    can be supplied to densify the constraints independently from waypoints.
    Interval polynomial bounds are conservative; a finer grid can reduce this
    conservatism. Requested endpoint path speeds are preserved or rejected as
    infeasible, never changed by a subsequent time scaling.
    """

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        max_velocity: Sequence[float],
        max_acceleration: Sequence[float],
        *,
        start_path_velocity: float = 0.0,
        end_path_velocity: float = 0.0,
        gridpoints: Optional[Sequence[float]] = None,
        grid_size: int = 200,
    ) -> None:
        points = np.asarray(waypoints, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
            raise ValueError("waypoints must have shape (count >= 2, dof >= 1)")
        if not np.isfinite(points).all():
            raise ValueError("waypoints must be finite")
        self.waypoints = points.copy()
        self.waypoints.setflags(write=False)
        self.dof = points.shape[1]
        self.max_velocity = _positive_vector(max_velocity, self.dof, "max_velocity")
        self.max_acceleration = _positive_vector(
            max_acceleration, self.dof, "max_acceleration"
        )
        self.max_velocity.setflags(write=False)
        self.max_acceleration.setflags(write=False)
        if not (
            np.isfinite(start_path_velocity)
            and start_path_velocity >= 0.0
            and np.isfinite(end_path_velocity)
            and end_path_velocity >= 0.0
        ):
            raise ValueError("boundary path velocities must be finite and non-negative")

        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("consecutive waypoints must be distinct")
        waypoint_s = np.concatenate(([0.0], np.cumsum(lengths)))
        waypoint_s /= waypoint_s[-1]
        self._waypoint_s = waypoint_s
        self._path_model = _NaturalCubicPath(waypoint_s, points)

        if gridpoints is None:
            if not isinstance(grid_size, Integral) or isinstance(grid_size, bool):
                raise TypeError("grid_size must be an integer")
            count = max(int(grid_size), points.shape[0])
            if count < 2:
                raise ValueError("grid_size must be at least two")
            grid = np.unique(np.concatenate((np.linspace(0.0, 1.0, count), waypoint_s)))
        else:
            grid = np.asarray(gridpoints, dtype=float).reshape(-1)
            if (
                grid.size < 2
                or not np.isfinite(grid).all()
                or abs(grid[0]) > 1e-12
                or abs(grid[-1] - 1.0) > 1e-12
                or np.any(np.diff(grid) <= 0.0)
                or np.any(grid[1:-1] < 0.0)
                or np.any(grid[1:-1] > 1.0)
            ):
                raise ValueError("gridpoints must increase strictly from 0 to 1")
            # Endpoint tolerance accepts representation error, not intervals
            # outside the path domain. Own the copy before canonicalizing it.
            grid = grid.copy()
            grid[0], grid[-1] = 0.0, 1.0
            grid = np.unique(np.concatenate((grid, waypoint_s)))
        self._grid = grid
        self._path, self._q_s, self._q_ss = self._path_model.evaluate_all(grid)
        self._result = self._compute(
            float(start_path_velocity), float(end_path_velocity)
        )

    @property
    def result(self) -> ToppraResult:
        """Validated immutable timing snapshot (read-only)."""

        return self._result

    @property
    def duration(self) -> float:
        return self.result.duration

    def _velocity_caps(self) -> np.ndarray:
        ratios = np.full_like(self._q_s, np.inf)
        moving = np.abs(self._q_s) > 1e-12
        limits = np.broadcast_to(self.max_velocity, self._q_s.shape)
        ratios[moving] = limits[moving] / np.abs(self._q_s[moving])
        return np.min(ratios, axis=1) ** 2

    def _interval_constraints(self) -> list[_IntervalConstraints]:
        ds = np.diff(self._grid)[:, None]
        first, last = self._q_s[:-1], self._q_s[1:]
        second_first, second_last = self._q_ss[:-1], self._q_ss[1:]
        # All spline knots are in the grid. On each interval q_s is quadratic
        # with these Bernstein coefficients, and q_ss is linear.
        derivative = np.stack((first, first + 0.5 * ds * second_first, last), axis=1)
        zero = np.zeros_like(first)
        acceleration_x = (
            np.stack((second_first, 0.5 * second_last, zero), axis=1)
            - derivative / (2.0 * ds[:, None, :])
        ) / self.max_acceleration
        acceleration_y = (
            np.stack((zero, 0.5 * second_first, second_last), axis=1)
            + derivative / (2.0 * ds[:, None, :])
        ) / self.max_acceleration

        # Bound |q_s| at both endpoints and its quadratic extremum. Applying
        # the resulting velocity cap independently to x and y also bounds
        # their linear interpolation. Independent caps avoid artificial stops
        # caused by maximizing x against a coupled velocity constraint on y.
        p0, p1, p2 = np.moveaxis(derivative / self.max_velocity, 1, 0)
        quadratic = p0 - 2.0 * p1 + p2
        ratio = np.clip(
            np.divide(
                p0 - p1, quadratic, out=np.zeros_like(p0), where=quadratic != 0.0
            ),
            0.0,
            1.0,
        )
        extremum = (
            (1.0 - ratio) ** 2 * p0 + 2.0 * ratio * (1.0 - ratio) * p1 + ratio**2 * p2
        )
        maximum = np.maximum(np.maximum(np.abs(p0), np.abs(p2)), np.abs(extremum))
        inverse_cap = np.max(maximum**2, axis=1)[:, None]
        count = ds.size
        a = np.concatenate(
            (
                acceleration_x.reshape(count, -1),
                -acceleration_x.reshape(count, -1),
                inverse_cap,
                np.zeros_like(inverse_cap),
            ),
            axis=1,
        )
        b = np.concatenate(
            (
                acceleration_y.reshape(count, -1),
                -acceleration_y.reshape(count, -1),
                np.zeros_like(inverse_cap),
                inverse_cap,
            ),
            axis=1,
        )
        return [_IntervalConstraints(left, right) for left, right in zip(a, b)]

    def _compute(self, start_velocity: float, end_velocity: float) -> ToppraResult:
        caps = self._velocity_caps()
        start_x, end_x = start_velocity**2, end_velocity**2
        if start_x > caps[0] + 1e-10 or end_x > caps[-1] + 1e-10:
            raise ValueError("boundary path velocity violates joint velocity limits")
        constraints = self._interval_constraints()
        controllable = np.empty((self._grid.size, 2))
        controllable[-1] = (end_x, end_x)
        for index in range(len(constraints) - 1, -1, -1):
            lo, hi = constraints[index].project(*controllable[index + 1])
            if not np.isfinite([lo, hi]).all() or lo > hi + 1e-12:
                raise ValueError(f"path is infeasible near gridpoint {index}")
            controllable[index] = (lo, max(lo, hi))
        if start_x < controllable[0, 0] - 1e-12 or start_x > controllable[0, 1] + 1e-12:
            raise ValueError("start velocity cannot reach the requested end velocity")

        x = np.empty_like(self._grid)
        x[0] = start_x
        for index, interval in enumerate(constraints):
            lower, upper = controllable[index + 1]
            low, high = interval.reachable(x[index], lower, upper)
            if low > high + 1e-12:
                raise RuntimeError(f"TOPPRA forward pass failed at gridpoint {index}")
            x[index + 1] = np.clip(high, lower, upper)
        ds = np.diff(self._grid)
        u = np.diff(x) / (2.0 * ds)
        speeds = np.sqrt(np.maximum(x, 0.0))
        denominators = speeds[:-1] + speeds[1:]
        if np.any(denominators <= 1e-14):
            raise ValueError("path contains an interval with zero reachable speed")
        dt = 2.0 * ds / denominators
        times = np.concatenate(([0.0], np.cumsum(dt)))

        return ToppraResult(self._grid.copy(), speeds, u, times, float(times[-1]))

    def sample(
        self, times: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = np.asarray(times, dtype=float).reshape(-1)
        if not np.isfinite(query).all():
            raise ValueError("sample times must be finite")
        clipped = np.clip(query, 0.0, self.duration)
        segment = np.minimum(
            np.searchsorted(self.result.times, clipped, side="right") - 1,
            len(self.result.path_accelerations) - 1,
        )
        segment = np.maximum(segment, 0)
        accel = self.result.path_accelerations[segment]
        elapsed = clipped - self.result.times[segment]
        initial_speed = self.result.path_speeds[segment]
        s = self._grid[segment] + initial_speed * elapsed + 0.5 * accel * elapsed**2
        s = np.clip(s, self._grid[segment], self._grid[segment + 1])
        speed = np.maximum(0.0, initial_speed + accel * elapsed)
        q, q_s, q_ss = self._path_model.evaluate_all(s)
        return (
            q,
            q_s * speed[..., None],
            q_ss * speed[..., None] ** 2 + q_s * accel[..., None],
        )

    def sample_uniform(
        self, count: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(count, Integral) or isinstance(count, bool):
            raise TypeError("count must be an integer")
        if count < 2:
            raise ValueError("count must be at least two")
        times = np.linspace(0.0, self.duration, count)
        return (times, *self.sample(times))


def retime_path(
    waypoints: Sequence[Sequence[float]],
    max_velocity: Sequence[float],
    max_acceleration: Sequence[float],
    **kwargs,
) -> ToppraTrajectory:
    """Convenience factory for :class:`ToppraTrajectory`."""

    return ToppraTrajectory(waypoints, max_velocity, max_acceleration, **kwargs)
