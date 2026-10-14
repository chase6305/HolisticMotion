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

import numpy as np

_MAX_PATH_SPEED = np.sqrt(np.finfo(float).max)
_MIN_INVERSE_CAP = 1.0 / float(np.finfo(float).max)


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


def _real_array(value, name: str, *, copy: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real")
    if copy:
        return np.array(array, dtype=float, copy=True)
    return np.asarray(array, dtype=float)


def _positive_vector(value: Sequence[float], dof: int, name: str) -> np.ndarray:
    result = _real_array(value, name, copy=True).reshape(-1)
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

    def __init__(
        self, a: np.ndarray, b: np.ndarray, *, cap: float | None = None
    ) -> None:
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("path speed constraints must be finite")
        self.a, self.b = a, b
        self.positive = b > 0.0
        self.negative = b < 0.0
        if cap is not None:
            self.cap = float(cap)
            return
        # Eliminate y by comparing each lower bound on y with each upper
        # bound. These pairs do not depend on the next controllable interval,
        # so compute their cap on x once, without iterative feasibility probes.
        coefficients = (
            a[self.negative, None] * b[self.positive]
            - b[self.negative, None] * a[self.positive]
        )
        rhs = b[self.positive] - b[self.negative, None]
        # rhs is positive, so only positive coefficients can constrain x >= 0.
        mask = coefficients > 0.0
        self.cap = float(np.min(rhs[mask] / coefficients[mask], initial=np.inf))

    def prepare_to_rest(self) -> None:
        """Cache only the coefficient subsets needed by the zero-bound solve."""

        a, b = self.a, self.b
        fixed = (a > 0.0) & (b >= 0.0)
        self.zero_cap = min(self.cap, float(np.min(1.0 / a[fixed], initial=np.inf)))
        coupled = (a > 0.0) & self.negative
        self.coupled_a, self.coupled_b = a[coupled], b[coupled]
        self.forward_a, self.forward_b = a[self.positive], b[self.positive]

    def project_zero(self, upper: float) -> float:
        """Upper controllable bound when the lower bound is known to be zero."""

        return min(
            self.zero_cap,
            float(
                np.min(
                    (1.0 - self.coupled_b * upper) / self.coupled_a,
                    initial=np.inf,
                )
            ),
        )

    def reachable_upper(self, x: float, upper: float) -> float:
        """Largest next speed; the caller checks all half-planes afterward."""

        product = self.forward_a * x
        rounding = 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(product))
        residual = 1.0 - product
        # Preserve exact stops at projected vertices: a tiny positive speed
        # introduced only by rounding would be amplified by its square root.
        rhs = np.where(np.abs(residual) <= rounding, 0.0, residual + rounding)
        return min(upper, float(np.min(rhs / self.forward_b, initial=np.inf)))

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
        # Residuals indistinguishable from zero stay zero, including at stops.
        rounding = 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(product))
        residual = 1.0 - product
        rhs = residual + rounding
        if lower == 0.0:
            rhs[np.abs(residual) <= rounding] = 0.0
        return _intersect_bounds(self.b, rhs, lower, upper)


def _projection_caps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Eliminate y from symmetric strips, in bounded interval chunks.

    The first K columns and next K columns are opposite half-planes. Each
    distinct pair gives |a_i*b_j - b_i*a_j|*x <= |b_i| + |b_j|. Include the
    positive-y velocity half-plane separately; zero-b bounds remain in the
    scalar backward solve. This halves the pair work and batches NumPy calls.
    """
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("path speed constraints must be finite")
    count = (a.shape[1] - 2) // 2
    left, right = np.triu_indices(count, k=1)
    # Target 128 KiB per candidate array to keep CPU scratch cache-friendly.
    chunk = max(1, min(4096, (1 << 14) // (left.size + count)))
    result = np.empty(a.shape[0])
    for offset in range(0, a.shape[0], chunk):
        aa = a[offset : offset + chunk, :count]
        bb = b[offset : offset + chunk, :count]
        velocity_cap = b[offset : offset + chunk, -1:]
        ai, aj = np.take(aa, left, axis=1), np.take(aa, right, axis=1)
        bi, bj = np.take(bb, left, axis=1), np.take(bb, right, axis=1)
        determinant = ai * bj - bi * aj
        np.abs(determinant, out=determinant)
        valid = (bi != 0.0) & (bj != 0.0) & (determinant > 0.0)
        bound = np.abs(bi) + np.abs(bj)
        np.divide(bound, determinant, out=bound, where=valid)
        bound[~valid] = np.inf
        pair_cap = np.min(bound, axis=1, initial=np.inf)
        del ai, aj, bi, bj, determinant, valid, bound

        divisor = np.abs(aa) * velocity_cap
        valid = (((aa < 0.0) & (bb > 0.0)) | ((aa > 0.0) & (bb < 0.0))) & (
            divisor > 0.0
        )
        bound = np.abs(bb) + velocity_cap
        np.divide(bound, divisor, out=bound, where=valid)
        bound[~valid] = np.inf
        result[offset : offset + chunk] = np.minimum(
            pair_cap, np.min(bound, axis=1, initial=np.inf)
        )
        del divisor, valid, bound, pair_cap
    return result


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
            value = _real_array(getattr(self, name), name, copy=True)
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
        if np.iscomplexobj(self.duration):
            raise TypeError("duration must be real")
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
        gridpoints: Sequence[float] | None = None,
        grid_size: int = 200,
    ) -> None:
        points = _real_array(waypoints, "waypoints")
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
        if np.iscomplexobj(start_path_velocity) or np.iscomplexobj(end_path_velocity):
            raise TypeError("boundary path velocities must be real")
        if not (
            np.isfinite(start_path_velocity)
            and start_path_velocity >= 0.0
            and np.isfinite(end_path_velocity)
            and end_path_velocity >= 0.0
        ):
            raise ValueError("boundary path velocities must be finite and non-negative")

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            delta = np.diff(points, axis=0)
            lengths = np.linalg.norm(delta, axis=1)
        if not np.isfinite(delta).all():
            raise ValueError("waypoint differences must be finite")
        if not np.isfinite(lengths).all():
            # Squaring large coordinates can overflow a representable norm.
            # Preserve the ordinary path and use scaled norms only when needed.
            scale = np.max(np.abs(delta), axis=1)
            normalized = np.divide(
                delta,
                scale[:, None],
                out=np.zeros_like(delta),
                where=scale[:, None] > 0,
            )
            with np.errstate(over="ignore"):
                lengths = np.linalg.norm(normalized, axis=1) * scale
        if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
            raise ValueError(
                "consecutive waypoints must be distinct with finite chord lengths"
            )
        with np.errstate(over="ignore"):
            waypoint_s = np.concatenate(([0.0], np.cumsum(lengths)))
        if not np.isfinite(waypoint_s[-1]):
            # Only relative chord lengths enter the spline parameterization.
            waypoint_s = np.concatenate(([0.0], np.cumsum(lengths / np.max(lengths))))
        waypoint_s /= waypoint_s[-1]
        self._waypoint_s = waypoint_s
        self._path_model = _NaturalCubicPath(waypoint_s, points)

        if gridpoints is None:
            if not isinstance(grid_size, Integral) or isinstance(grid_size, bool):
                raise TypeError("grid_size must be an integer")
            if grid_size < 2:
                raise ValueError("grid_size must be at least two")
            # Subdivide each spline segment instead of merging two grids:
            # nominally equal knots can otherwise differ by a few ulps and
            # create intervals whose arrival times cannot be represented.
            requested = np.diff(waypoint_s) * (int(grid_size) - 1)
            nearest = np.rint(requested)
            requested = np.where(
                np.abs(requested - nearest)
                <= 64.0 * np.finfo(float).eps * np.maximum(1.0, requested),
                nearest,
                requested,
            )
            subdivisions = np.maximum(1, np.ceil(requested).astype(int))
            grid = np.concatenate(
                [
                    np.linspace(left, right, int(count), endpoint=False)
                    for left, right, count in zip(
                        waypoint_s[:-1], waypoint_s[1:], subdivisions
                    )
                ]
                + [waypoint_s[-1:]]
            )
        else:
            grid = _real_array(gridpoints, "gridpoints").reshape(-1)
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
            # Prefer the exact spline knot to a custom point within rounding
            # distance. Never remove spline knots or merge genuine segments.
            insertion = np.searchsorted(waypoint_s, grid)
            left = waypoint_s[np.maximum(insertion - 1, 0)]
            right = waypoint_s[np.minimum(insertion, waypoint_s.size - 1)]
            distance = np.minimum(np.abs(grid - left), np.abs(grid - right))
            grid = np.unique(
                np.concatenate((grid[distance > 8.0 * np.finfo(float).eps], waypoint_s))
            )
        self._grid = grid
        self._path, self._q_s, self._q_ss = self._path_model.evaluate_all(grid)
        self._speed_scale = 1.0
        self._solver_velocity = self.max_velocity
        self._solver_acceleration = self.max_acceleration
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
        limits = np.broadcast_to(self._solver_velocity, self._q_s.shape)
        ratios[moving] = limits[moving] / np.abs(self._q_s[moving])
        # A loose velocity bound may exceed representable squared speeds. Its
        # infinite cap is valid; the acceleration constraints still limit x.
        with np.errstate(over="ignore"):
            return np.min(ratios, axis=1) ** 2

    def _interval_constraints(self) -> list[_IntervalConstraints]:
        ds = np.diff(self._grid)[:, None]
        first, last = self._q_s[:-1], self._q_s[1:]
        second_first, second_last = self._q_ss[:-1], self._q_ss[1:]
        # All spline knots are in the grid. On each interval q_s is quadratic
        # with these Bernstein coefficients, and q_ss is linear.
        derivative = np.stack((first, first + 0.5 * ds * second_first, last), axis=1)
        zero = np.zeros_like(first)
        acceleration_x = np.stack(
            (second_first, 0.5 * second_last, zero), axis=1
        ) - derivative / (2.0 * ds[:, None, :])
        acceleration_y = np.stack(
            (zero, 0.5 * second_first, second_last), axis=1
        ) + derivative / (2.0 * ds[:, None, :])
        unit_scale = max(
            float(np.max(self.max_velocity)),
            float(np.sqrt(np.max(self.max_acceleration))),
        )
        with np.errstate(over="ignore", under="ignore"):
            velocity_scale = np.max(np.abs(derivative) / self.max_velocity)
            acceleration_scale = np.max(
                np.sqrt(np.maximum(np.abs(acceleration_x), np.abs(acceleration_y)))
                / np.sqrt(self.max_acceleration)
            )
        inverse_speed_scale = max(velocity_scale, acceleration_scale)
        if (
            unit_scale < 1e-60
            or unit_scale > 1e60
            or inverse_speed_scale < 1e-60
            or inverse_speed_scale > 1e60
        ):
            # A change of speed units keeps elimination products near one.
            # Normalize limits before division rather than rescaling already
            # overflowing/underflowing half-plane coefficients.
            self._speed_scale = 1.0 / inverse_speed_scale
            if not np.isfinite(self._speed_scale) or self._speed_scale <= 0.0:
                raise ValueError("path speed scale is not representable")
            with np.errstate(over="ignore", under="ignore"):
                self._solver_velocity = self.max_velocity / self._speed_scale
                self._solver_acceleration = (
                    self.max_acceleration / self._speed_scale
                ) / self._speed_scale
            # The selected scale bounds every nonzero geometry coefficient.
            # A normalized limit can underflow only for an inactive axis;
            # give its zero numerators a safe denominator.
            self._solver_velocity = np.where(
                self._solver_velocity > 0.0, self._solver_velocity, 1.0
            )
            self._solver_acceleration = np.where(
                self._solver_acceleration > 0.0, self._solver_acceleration, 1.0
            )
        acceleration_x = acceleration_x / self._solver_acceleration
        acceleration_y = acceleration_y / self._solver_acceleration

        # Bound |q_s| at both endpoints and its quadratic extremum. Applying
        # the resulting velocity cap independently to x and y also bounds
        # their linear interpolation. Independent caps avoid artificial stops
        # caused by maximizing x against a coupled velocity constraint on y.
        p0, p1, p2 = np.moveaxis(derivative / self._solver_velocity, 1, 0)
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
        # Such a bound cannot restrict any finite squared speed. Dropping it
        # also avoids overflowing redundant projection and reciprocal caps.
        inverse_cap[inverse_cap < _MIN_INVERSE_CAP] = 0.0
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
        caps = _projection_caps(a, b)
        return [
            _IntervalConstraints(left, right, cap=cap)
            for left, right, cap in zip(a, b, caps)
        ]

    def _compute(self, start_velocity: float, end_velocity: float) -> ToppraResult:
        constraints = self._interval_constraints()
        caps = self._velocity_caps()
        with np.errstate(over="ignore"):
            normalized_start = start_velocity / self._speed_scale
            normalized_end = end_velocity / self._speed_scale
        if normalized_start > _MAX_PATH_SPEED or normalized_end > _MAX_PATH_SPEED:
            raise ValueError("boundary path velocity is not representable")
        start_x, end_x = normalized_start**2, normalized_end**2
        if start_x > caps[0] + 1e-10 * max(start_x, caps[0]) or end_x > caps[
            -1
        ] + 1e-10 * max(end_x, caps[-1]):
            raise ValueError("boundary path velocity violates joint velocity limits")
        if end_velocity == 0.0:
            x = self._solve_to_rest(constraints, start_x)
        else:
            x = self._solve_general(constraints, start_x, end_x)
        ds = np.diff(self._grid)
        u = np.diff(x) / (2.0 * ds)
        speeds = np.sqrt(np.maximum(x, 0.0))
        denominators = speeds[:-1] + speeds[1:]
        if np.any(denominators <= 0.0):
            raise ValueError(
                "path contains an interval with zero reachable speed; "
                "refine gridpoints or increase grid_size"
            )
        dt = (2.0 * ds / denominators) / self._speed_scale
        times = np.concatenate(([0.0], np.cumsum(dt)))

        if self._speed_scale != 1.0:
            self._normalized_speeds = speeds
            self._normalized_u = u
            speeds = speeds * self._speed_scale
            u = (u * self._speed_scale) * self._speed_scale
        return ToppraResult(self._grid.copy(), speeds, u, times, float(times[-1]))

    def _solve_to_rest(
        self, constraints: list[_IntervalConstraints], start_x: float
    ) -> np.ndarray:
        # Rest satisfies every half-plane. With a zero terminal speed, each
        # backward controllable interval therefore has lower bound zero.
        # Keep only its upper bound, including for a nonzero initial speed.
        controllable = np.zeros(self._grid.size)
        for index in range(len(constraints) - 1, -1, -1):
            constraints[index].prepare_to_rest()
            cap = constraints[index].project_zero(controllable[index + 1])
            if not np.isfinite(cap) or cap < 0.0:
                raise ValueError(f"path is infeasible near gridpoint {index}")
            controllable[index] = cap
        if start_x > controllable[0] + 1e-12 * max(start_x, controllable[0]):
            raise ValueError("start velocity cannot reach the requested end velocity")
        x = np.empty_like(self._grid)
        x[0] = start_x
        for index, interval in enumerate(constraints):
            x[index + 1] = max(
                0.0, interval.reachable_upper(x[index], controllable[index + 1])
            )
        # Check both signs and zero coefficients together after the sequential
        # solve, using dimensionless residuals even under a change of units.
        a = np.stack([interval.a for interval in constraints])
        b = np.stack([interval.b for interval in constraints])
        left, right = a * x[:-1, None], b * x[1:, None]
        tolerance = 1e-10 * np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
        if np.any(left + right - 1.0 > tolerance):
            raise RuntimeError("TOPPRA forward pass violates interval constraints")
        return x

    def _solve_general(
        self, constraints: list[_IntervalConstraints], start_x: float, end_x: float
    ) -> np.ndarray:
        controllable = np.empty((self._grid.size, 2))
        controllable[-1] = (end_x, end_x)
        for index in range(len(constraints) - 1, -1, -1):
            lo, hi = constraints[index].project(*controllable[index + 1])
            if not np.isfinite([lo, hi]).all() or lo > hi + 1e-12 * max(
                abs(lo), abs(hi)
            ):
                raise ValueError(f"path is infeasible near gridpoint {index}")
            controllable[index] = (lo, max(lo, hi))
        tolerance = 1e-12 * max(start_x, *np.abs(controllable[0]))
        if (
            start_x < controllable[0, 0] - tolerance
            or start_x > controllable[0, 1] + tolerance
        ):
            raise ValueError("start velocity cannot reach the requested end velocity")

        x = np.empty_like(self._grid)
        x[0] = start_x
        for index, interval in enumerate(constraints):
            lower, upper = controllable[index + 1]
            low, high = interval.reachable(x[index], lower, upper)
            if low > high + 1e-12 * max(abs(low), abs(high)):
                raise RuntimeError(f"TOPPRA forward pass failed at gridpoint {index}")
            x[index + 1] = np.clip(high, lower, upper)
        return x

    def sample(
        self, times: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = _real_array(times, "sample times").reshape(-1)
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
        if self._speed_scale != 1.0:
            elapsed *= self._speed_scale
            accel = self._normalized_u[segment]
            initial_speed = self._normalized_speeds[segment]
        s = self._grid[segment] + initial_speed * elapsed + 0.5 * accel * elapsed**2
        s = np.clip(s, self._grid[segment], self._grid[segment + 1])
        speed = np.maximum(0.0, initial_speed + accel * elapsed)
        q, q_s, q_ss = self._path_model.evaluate_all(s)
        result = (
            q,
            q_s * speed[..., None],
            q_ss * speed[..., None] ** 2 + q_s * accel[..., None],
        )
        if self._speed_scale != 1.0:
            q, velocity, acceleration = result
            result = (
                q,
                velocity * self._speed_scale,
                (acceleration * self._speed_scale) * self._speed_scale,
            )
        return result

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
