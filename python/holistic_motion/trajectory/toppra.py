"""Dependency-free TOPPRA-style path time parametrization.

The implementation uses the standard path dynamics ``x = s_dot**2`` and
``x[i+1] = x[i] + 2 * ds[i] * u[i]``.  A backward controllable-set pass is
followed by a greedy forward pass.  Joint velocity and acceleration bounds are
enforced at every path grid point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np


class _NaturalCubicPath:
    """Small vector-valued natural cubic spline with analytic derivatives."""

    def __init__(self, grid: np.ndarray, values: np.ndarray) -> None:
        self.grid = grid
        count = grid.size
        h = np.diff(grid)
        matrix = np.zeros((count, count))
        rhs = np.zeros_like(values)
        matrix[0, 0] = matrix[-1, -1] = 1.0
        for index in range(1, count - 1):
            matrix[index, index - 1] = h[index - 1]
            matrix[index, index] = 2.0 * (h[index - 1] + h[index])
            matrix[index, index + 1] = h[index]
            rhs[index] = 6.0 * (
                (values[index + 1] - values[index]) / h[index]
                - (values[index] - values[index - 1]) / h[index - 1]
            )
        second = np.linalg.solve(matrix, rhs)
        self.a = values[:-1].copy()
        self.b = (
            np.diff(values, axis=0) / h[:, None]
            - h[:, None] * (2.0 * second[:-1] + second[1:]) / 6.0
        )
        self.c = second[:-1] / 2.0
        self.d = np.diff(second, axis=0) / (6.0 * h[:, None])

    def evaluate(self, value: np.ndarray, order: int = 0) -> np.ndarray:
        query = np.asarray(value, dtype=float).reshape(-1)
        index = np.clip(
            np.searchsorted(self.grid, query, side="right") - 1,
            0,
            self.grid.size - 2,
        )
        delta = (query - self.grid[index])[:, None]
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


def _positive_vector(value: Sequence[float], dof: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (dof,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of size {dof}")
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return result


@dataclass(frozen=True)
class ToppraResult:
    """Discrete path parametrization returned by :func:`retime_path`."""

    gridpoints: np.ndarray
    path_speeds: np.ndarray
    path_accelerations: np.ndarray
    times: np.ndarray
    duration: float


class ToppraTrajectory:
    """Time-optimal timing of a joint-space waypoint path.

    Waypoints are interpolated along normalized chord length.  ``gridpoints``
    can be supplied to densify the constraints independently from waypoints.
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
        self.dof = points.shape[1]
        self.max_velocity = _positive_vector(max_velocity, self.dof, "max_velocity")
        self.max_acceleration = _positive_vector(
            max_acceleration, self.dof, "max_acceleration"
        )
        if start_path_velocity < 0.0 or end_path_velocity < 0.0:
            raise ValueError("boundary path velocities must be non-negative")

        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("consecutive waypoints must be distinct")
        waypoint_s = np.concatenate(([0.0], np.cumsum(lengths)))
        waypoint_s /= waypoint_s[-1]
        self._waypoint_s = waypoint_s
        self._path_model = _NaturalCubicPath(waypoint_s, points)

        if gridpoints is None:
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
            ):
                raise ValueError("gridpoints must increase strictly from 0 to 1")
            grid = np.unique(np.concatenate((grid, waypoint_s)))
        self._grid = grid
        self._path = self._path_model.evaluate(grid)
        self._q_s = self._path_model.evaluate(grid, order=1)
        self._q_ss = self._path_model.evaluate(grid, order=2)
        self.result = self._compute(
            float(start_path_velocity), float(end_path_velocity)
        )

    @property
    def duration(self) -> float:
        return self.result.duration

    def _velocity_caps(self) -> np.ndarray:
        ratios = np.full_like(self._q_s, np.inf)
        moving = np.abs(self._q_s) > 1e-12
        limits = np.broadcast_to(self.max_velocity, self._q_s.shape)
        ratios[moving] = limits[moving] / np.abs(self._q_s[moving])
        return np.min(ratios, axis=1) ** 2

    def _next_interval(self, index: int, x: float, upper: float) -> tuple[float, float]:
        ds = self._grid[index + 1] - self._grid[index]
        low, high = 0.0, upper
        for q_s, q_ss, limit in zip(
            self._q_s[index], self._q_ss[index], self.max_acceleration
        ):
            coefficient = q_s / (2.0 * ds)
            constant = (q_ss - coefficient) * x
            if abs(coefficient) <= 1e-14:
                if abs(constant) > limit + 1e-10:
                    return 1.0, 0.0
                continue
            a = (-limit - constant) / coefficient
            b = (limit - constant) / coefficient
            low, high = max(low, min(a, b)), min(high, max(a, b))
        return max(0.0, low), high

    def _can_reach(self, index: int, x: float, next_cap: float) -> bool:
        low, high = self._next_interval(index, x, next_cap)
        return low <= high + 1e-11 and high >= -1e-11

    def _compute(self, start_velocity: float, end_velocity: float) -> ToppraResult:
        caps = self._velocity_caps()
        start_x, end_x = start_velocity**2, end_velocity**2
        if start_x > caps[0] + 1e-10 or end_x > caps[-1] + 1e-10:
            raise ValueError("boundary path velocity violates joint velocity limits")
        controllable = caps.copy()
        controllable[-1] = end_x
        for index in range(len(self._grid) - 2, -1, -1):
            upper = caps[index]
            if not np.isfinite(upper):
                upper = max(controllable[index + 1], 1.0)
                while self._can_reach(index, upper, controllable[index + 1]):
                    upper *= 2.0
                    if upper > 1e12:
                        break
            lo, hi = 0.0, upper
            if not self._can_reach(index, lo, controllable[index + 1]):
                raise ValueError(f"path is infeasible near gridpoint {index}")
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if self._can_reach(index, mid, controllable[index + 1]):
                    lo = mid
                else:
                    hi = mid
            controllable[index] = lo
        if start_x > controllable[0] + 1e-8:
            raise ValueError("start velocity cannot reach the requested end velocity")

        x = np.empty_like(self._grid)
        x[0] = start_x
        for index in range(len(self._grid) - 1):
            low, high = self._next_interval(index, x[index], controllable[index + 1])
            if low > high + 1e-9:
                raise RuntimeError(f"TOPPRA forward pass failed at gridpoint {index}")
            x[index + 1] = max(0.0, high)
        x[-1] = end_x
        ds = np.diff(self._grid)
        u = np.diff(x) / (2.0 * ds)
        speeds = np.sqrt(np.maximum(x, 0.0))
        denominators = speeds[:-1] + speeds[1:]
        if np.any(denominators <= 1e-14):
            raise ValueError("path contains an interval with zero reachable speed")
        dt = 2.0 * ds / denominators
        times = np.concatenate(([0.0], np.cumsum(dt)))

        # Grid constraints are exact at their nodes. Apply a small global time
        # scaling based on a dense continuous check so spline extrema between
        # nodes cannot exceed the requested limits.
        check_s = np.linspace(0.0, 1.0, max(1000, 5 * self._grid.size))
        check_x = np.interp(check_s, self._grid, x)
        check_segment = np.clip(
            np.searchsorted(self._grid, check_s, side="right") - 1,
            0,
            u.size - 1,
        )
        check_q_s = self._path_model.evaluate(check_s, order=1)
        check_q_ss = self._path_model.evaluate(check_s, order=2)
        joint_velocity = check_q_s * np.sqrt(check_x)[:, None]
        joint_acceleration = (
            check_q_ss * check_x[:, None] + check_q_s * u[check_segment, None]
        )
        velocity_ratio = float(
            np.max(np.abs(joint_velocity) / self.max_velocity[None, :])
        )
        acceleration_ratio = float(
            np.max(np.abs(joint_acceleration) / self.max_acceleration[None, :])
        )
        scale = max(1.0, velocity_ratio, np.sqrt(acceleration_ratio))
        if scale > 1.0:
            scale *= 1.0 + 1e-9
            speeds /= scale
            u /= scale * scale
            times *= scale
        return ToppraResult(self._grid.copy(), speeds, u, times, float(times[-1]))

    def sample(
        self, times: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = np.asarray(times, dtype=float)
        if not np.isfinite(query).all():
            raise ValueError("sample times must be finite")
        clipped = np.clip(query, 0.0, self.duration)
        s = np.interp(clipped, self.result.times, self._grid)
        speed = np.interp(clipped, self.result.times, self.result.path_speeds)
        segment = np.minimum(
            np.searchsorted(self.result.times, clipped, side="right") - 1,
            len(self.result.path_accelerations) - 1,
        )
        segment = np.maximum(segment, 0)
        accel = self.result.path_accelerations[segment]
        q = self._path_model.evaluate(s)
        q_s = self._path_model.evaluate(s, order=1)
        q_ss = self._path_model.evaluate(s, order=2)
        return (
            q,
            q_s * speed[..., None],
            q_ss * speed[..., None] ** 2 + q_s * accel[..., None],
        )

    def sample_uniform(
        self, count: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
