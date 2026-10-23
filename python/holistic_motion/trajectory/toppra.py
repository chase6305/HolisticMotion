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

from .._validation import real_array
from ._toppra_constraints import (
    _intersect_bounds as _intersect_bounds,
)
from ._toppra_constraints import (
    _IntervalConstraints as _IntervalConstraints,
)
from ._toppra_constraints import (
    _pair_projection_caps as _pair_projection_caps,
)
from ._toppra_constraints import _projection_caps
from ._toppra_constraints import (
    _screened_projection_caps as _screened_projection_caps,
)

_MAX_PATH_SPEED = np.sqrt(np.finfo(float).max)
_MIN_INVERSE_CAP = 1.0 / float(np.finfo(float).max)
_LARGE_GEOMETRY = np.sqrt(np.finfo(float).max) / 16.0


def _scale_units(
    value: np.ndarray,
    coordinate_scale: np.ndarray,
    speed_scale: float,
    order: int,
    *,
    divisors: tuple = (),
) -> np.ndarray:
    """Combine coordinate/time units without overflowing intermediate scales."""

    mantissa, exponent = np.frexp(value)
    coordinate_mantissa, coordinate_exponent = np.frexp(coordinate_scale)
    speed_mantissa, speed_exponent = np.frexp(speed_scale)
    mantissa = mantissa * coordinate_mantissa
    exponent = exponent + coordinate_exponent
    for divisor in divisors:
        divisor_mantissa, divisor_exponent = np.frexp(divisor)
        mantissa = mantissa / divisor_mantissa
        exponent = exponent - divisor_exponent
    mantissa = mantissa * speed_mantissa**order
    exponent = exponent + order * speed_exponent
    with np.errstate(under="ignore"):
        return np.ldexp(mantissa, exponent)


class _NaturalCubicPath:
    """Small vector-valued natural cubic spline with analytic derivatives."""

    def __init__(self, grid: np.ndarray, values: np.ndarray) -> None:
        self.grid = grid
        h = np.diff(grid)
        if not np.isfinite(h).all() or np.any(h <= 0.0):
            raise ValueError("waypoint parameters must increase strictly")
        delta = np.diff(values, axis=0)
        self.a = values[:-1].copy()
        self._coefficient_scale = None
        self._normalized_coefficients = None
        normalized = np.any(np.abs(delta) > _LARGE_GEOMETRY)
        if not normalized:
            with np.errstate(over="ignore", invalid="ignore"):
                slopes = delta / h[:, None]
                self.b, self.c, self.d = self._derivative_coefficients(h, slopes)
            normalized = not all(
                np.isfinite(block).all() for block in (self.b, self.c, self.d)
            )
        if normalized:
            # The spline solve is linear in each coordinate. Normalize its
            # differences, preserving large offsets in a and small axes
            # independently, before restoring the final coefficients.
            scale = np.max(np.abs(delta), axis=0)
            scale = np.where(scale > 0.0, scale, 1.0)
            with np.errstate(over="ignore", invalid="ignore"):
                slopes = (delta / scale) / h[:, None]
                blocks = self._derivative_coefficients(h, slopes)
                self.b, self.c, self.d = (block * scale for block in blocks)
            if not all(np.isfinite(block).all() for block in (self.b, self.c, self.d)):
                raise ValueError("spline derivative coefficients must be finite")
            self._coefficient_scale = scale
            self._normalized_coefficients = blocks

    @staticmethod
    def _derivative_coefficients(h, slopes):
        count = h.size + 1
        second = np.zeros((count, slopes.shape[1]), dtype=slopes.dtype)
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
        return (
            slopes - h[:, None] * (2.0 * second[:-1] + second[1:]) / 6.0,
            second[:-1] / 2.0,
            np.diff(second, axis=0) / (6.0 * h[:, None]),
        )

    def evaluate(self, value: np.ndarray, order: int = 0) -> np.ndarray:
        if order not in (0, 1, 2):
            raise ValueError("spline derivative order must be 0, 1, or 2")
        if self._coefficient_scale is not None:
            return self.evaluate_all(value)[order]
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

    def evaluate_all(
        self, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate position and both derivatives with one segment lookup."""

        position, first, second = self._evaluate(value)
        if self._coefficient_scale is not None:
            first = first * self._coefficient_scale
            second = second * self._coefficient_scale
        return position, first, second

    def _evaluate(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep large-coordinate derivatives normalized until needed."""

        index, delta = self._segments(value)
        a = self.a[index]
        coefficients = self._normalized_coefficients
        if coefficients is None:
            coefficients = (self.b, self.c, self.d)
        b, c, d = (block[index] for block in coefficients)
        if a.size == 1:
            # Keep the short expression path for a single scalar sample.
            correction = delta * (b + delta * (c + delta * d))
            if self._coefficient_scale is not None:
                correction = correction * self._coefficient_scale
            position = a + correction
            del a, correction
            first = b + delta * (2.0 * c + 3.0 * delta * d)
            del b
            second = 2.0 * c + 6.0 * delta * d
            return position, first, second
        # Advanced indexing above owns all four coefficient buffers.
        # Horner steps retain their grouping while reusing that scratch.
        correction = delta * d
        correction += c
        correction *= delta
        correction += b
        correction *= delta
        if self._coefficient_scale is not None:
            correction *= self._coefficient_scale
        a += correction
        del correction
        c *= 2.0
        first = (3.0 * delta) * d
        first += c
        first *= delta
        first += b
        del b
        d *= 6.0 * delta
        d += c
        return a, first, d

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
    return real_array(value, name=name, copy=copy)


def _positive_vector(value: Sequence[float], dof: int, name: str) -> np.ndarray:
    result = _real_array(value, name, copy=True).reshape(-1)
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
        self._moving_axes = np.any(delta != 0.0, axis=0)
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
        with np.errstate(over="ignore", invalid="ignore"):
            self._path, self._q_s, self._q_ss = self._path_model.evaluate_all(grid)
        if not all(
            np.isfinite(value).all() for value in (self._path, self._q_s, self._q_ss)
        ):
            raise ValueError("physical spline positions and derivatives must be finite")
        self._speed_scale = 1.0
        self._range_safe_velocity_caps = None
        self._solver_velocity = self.max_velocity
        self._solver_acceleration = self.max_acceleration
        self._result = self._compute(
            float(start_path_velocity), float(end_path_velocity)
        )
        coordinate_scale = self._path_model._coefficient_scale
        if coordinate_scale is not None:
            factors = []
            factor = coordinate_scale
            for _ in range(2):
                with np.errstate(over="ignore", under="ignore"):
                    factor = factor * self._speed_scale
                safe = (np.isfinite(factor) & (factor >= np.finfo(float).tiny)).all()
                factors.append(factor if safe else None)
            self._sample_unit_factors = tuple(factors)

    @property
    def result(self) -> ToppraResult:
        """Validated immutable timing snapshot (read-only)."""

        return self._result

    @property
    def duration(self) -> float:
        return self.result.duration

    def _velocity_caps(self) -> np.ndarray:
        if self._range_safe_velocity_caps is not None:
            return self._range_safe_velocity_caps
        ratios = np.full_like(self._q_s, np.inf)
        moving = np.abs(self._q_s) > 1e-12
        if self._normalized_acceleration_constraints:
            limits = np.broadcast_to(self.max_velocity, self._q_s.shape)
            ratios[moving] = (
                limits[moving] / np.abs(self._q_s[moving])
            ) / self._speed_scale
        else:
            limits = np.broadcast_to(self._solver_velocity, self._q_s.shape)
            ratios[moving] = limits[moving] / np.abs(self._q_s[moving])
            overflowed = moving & np.isinf(limits)
            if np.any(overflowed):
                physical_limits = np.broadcast_to(self.max_velocity, self._q_s.shape)
                ratios[overflowed] = _scale_units(
                    physical_limits[overflowed],
                    1.0,
                    self._speed_scale,
                    -1,
                    divisors=(np.abs(self._q_s[overflowed]),),
                )
        # A loose velocity bound may exceed representable squared speeds. Its
        # infinite cap is valid; the acceleration constraints still limit x.
        with np.errstate(over="ignore"):
            return np.min(ratios, axis=1) ** 2

    def _interval_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        ds = np.diff(self._grid)[:, None]
        first, last = self._q_s[:-1], self._q_s[1:]
        second_first, second_last = self._q_ss[:-1], self._q_ss[1:]
        # All spline knots are in the grid. On each interval q_s is quadratic
        # with these Bernstein coefficients, and q_ss is linear.
        zero = np.zeros_like(first)
        with np.errstate(over="ignore", invalid="ignore"):
            # A conservative Bernstein control value may overflow even when
            # the actual geometric derivative stays finite. The range-safe
            # fallback below constructs it in normalized coordinate units.
            derivative = np.stack(
                (first, first + 0.5 * ds * second_first, last), axis=1
            )
            acceleration_x = np.stack(
                (second_first, 0.5 * second_last, zero), axis=1
            ) - derivative / (2.0 * ds[:, None, :])
            acceleration_y = np.stack(
                (zero, 0.5 * second_first, second_last), axis=1
            ) + derivative / (2.0 * ds[:, None, :])
        self._normalized_acceleration_constraints = not (
            np.isfinite(acceleration_x).all() and np.isfinite(acceleration_y).all()
        )
        if self._normalized_acceleration_constraints:
            # Dividing by a short interval can overflow before a large joint
            # acceleration limit restores range. Form those ratios first.
            with np.errstate(over="ignore", invalid="ignore"):
                normalized_derivative = derivative / self.max_acceleration
                acceleration_x = np.stack(
                    (
                        second_first / self.max_acceleration,
                        0.5 * second_last / self.max_acceleration,
                        zero,
                    ),
                    axis=1,
                ) - normalized_derivative / (2.0 * ds[:, None, :])
                acceleration_y = np.stack(
                    (
                        zero,
                        0.5 * second_first / self.max_acceleration,
                        second_last / self.max_acceleration,
                    ),
                    axis=1,
                ) + normalized_derivative / (2.0 * ds[:, None, :])
            if not (
                np.isfinite(acceleration_x).all() and np.isfinite(acceleration_y).all()
            ):
                return self._range_safe_interval_coefficients()
        unit_scale = max(
            float(np.max(self.max_velocity)),
            float(np.sqrt(np.max(self.max_acceleration))),
        )
        with np.errstate(over="ignore", under="ignore"):
            velocity_scale = np.max(np.abs(derivative) / self.max_velocity)
            acceleration_scale = np.sqrt(
                np.maximum(np.abs(acceleration_x), np.abs(acceleration_y))
            )
            if not self._normalized_acceleration_constraints:
                acceleration_scale = acceleration_scale / np.sqrt(self.max_acceleration)
            acceleration_scale = np.max(acceleration_scale)
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
        if self._normalized_acceleration_constraints:
            acceleration_x = (acceleration_x * self._speed_scale) * self._speed_scale
            acceleration_y = (acceleration_y * self._speed_scale) * self._speed_scale
        else:
            overflowed = np.isinf(self._solver_acceleration)
            denominator = np.where(overflowed, 1.0, self._solver_acceleration)
            acceleration_x = acceleration_x / denominator
            acceleration_y = acceleration_y / denominator
            if np.any(overflowed):
                # A large normalized limit is still finite as a ratio to a
                # large geometric coefficient. Combine their exponents;
                # dividing by an overflowing limit would drop the constraint.
                for block in (acceleration_x, acceleration_y):
                    block[..., overflowed] = _scale_units(
                        block[..., overflowed],
                        1.0,
                        self._speed_scale,
                        2,
                        divisors=(self.max_acceleration[overflowed],),
                    )

        # Bound |q_s| at both endpoints and its quadratic extremum. Applying
        # the resulting velocity cap independently to x and y also bounds
        # their linear interpolation. Independent caps avoid artificial stops
        # caused by maximizing x against a coupled velocity constraint on y.
        if self._normalized_acceleration_constraints:
            normalized_velocity = (derivative / self.max_velocity) * self._speed_scale
        else:
            overflowed = np.isinf(self._solver_velocity)
            normalized_velocity = derivative / np.where(
                overflowed, 1.0, self._solver_velocity
            )
            if np.any(overflowed):
                normalized_velocity[..., overflowed] = _scale_units(
                    normalized_velocity[..., overflowed],
                    1.0,
                    self._speed_scale,
                    1,
                    divisors=(self.max_velocity[overflowed],),
                )
        return self._assemble_constraints(
            acceleration_x, acceleration_y, normalized_velocity
        )

    def _range_safe_interval_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        # Choose a binary speed unit from conservative logarithmic bounds.
        # Form each normalized term with its complete scale/limit/interval
        # ratio, even when the corresponding physical coefficient overflows.
        ds = np.diff(self._grid)[:, None]
        _, first, second = self._path_model._evaluate(self._grid)
        coordinate = self._path_model._coefficient_scale
        if coordinate is None:
            coordinate = np.ones(self.dof)
        zero = np.zeros_like(first[:-1])
        derivative = np.stack(
            (first[:-1], first[:-1] + 0.5 * ds * second[:-1], first[1:]), axis=1
        )
        second_x = np.stack((second[:-1], 0.5 * second[1:], zero), axis=1)
        second_y = np.stack((zero, 0.5 * second[:-1], second[1:]), axis=1)
        with np.errstate(divide="ignore"):
            coordinate_log = np.log2(coordinate)
            derivative_log = np.log2(np.abs(derivative))
            second_log = np.log2(np.maximum(np.abs(second_x), np.abs(second_y)))
            acceleration_log = (
                np.logaddexp2(second_log, derivative_log - np.log2(2.0 * ds[:, None]))
                + coordinate_log
                - np.log2(self.max_acceleration)
            )
            velocity_log = derivative_log + coordinate_log - np.log2(self.max_velocity)
        inverse_log = max(
            float(np.max(velocity_log)), 0.5 * float(np.max(acceleration_log))
        )
        exponent = -int(np.ceil(inverse_log)) if np.isfinite(inverse_log) else 1024
        if exponent < -1074 or exponent > 1023:
            raise ValueError("path speed scale is not representable")
        self._speed_scale = float(np.ldexp(1.0, exponent))
        self._normalized_acceleration_constraints = True
        common = _scale_units(
            derivative,
            coordinate,
            self._speed_scale,
            2,
            divisors=(self.max_acceleration, 2.0 * ds[:, None]),
        )
        acceleration_x = (
            _scale_units(
                second_x,
                coordinate,
                self._speed_scale,
                2,
                divisors=(self.max_acceleration,),
            )
            - common
        )
        acceleration_y = (
            _scale_units(
                second_y,
                coordinate,
                self._speed_scale,
                2,
                divisors=(self.max_acceleration,),
            )
            + common
        )
        normalized_velocity = _scale_units(
            derivative, coordinate, self._speed_scale, 1, divisors=(self.max_velocity,)
        )
        nodes = np.concatenate((normalized_velocity[:, 0], normalized_velocity[-1:, 2]))
        inverse = np.max(nodes**2, axis=1)
        self._range_safe_velocity_caps = np.divide(
            1.0, inverse, out=np.full_like(inverse, np.inf), where=inverse > 0.0
        )
        return self._assemble_constraints(
            acceleration_x, acceleration_y, normalized_velocity
        )

    def _assemble_constraints(
        self,
        acceleration_x: np.ndarray,
        acceleration_y: np.ndarray,
        normalized_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p0, p1, p2 = np.moveaxis(normalized_velocity, 1, 0)
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
        count = normalized_velocity.shape[0]
        # A joint whose waypoints are all identical contributes only 0 <= 1.
        # Keep its samples and limits, but omit its redundant symmetric strips.
        if not self._moving_axes.all():
            acceleration_x = acceleration_x[..., self._moving_axes]
            acceleration_y = acceleration_y[..., self._moving_axes]
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
        return a, b

    def _compute(self, start_velocity: float, end_velocity: float) -> ToppraResult:
        a, b = self._interval_coefficients()
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
        if self.waypoints.shape[0] == 2:
            x = self._solve_linear(a, b, start_x, end_x)
        else:
            projected = _projection_caps(a, b)
            if end_velocity == 0.0:
                x = self._solve_to_rest(a, b, projected, start_x)
            else:
                constraints = [
                    _IntervalConstraints(left, right, cap=cap)
                    for left, right, cap in zip(a, b, projected)
                ]
                x = self._solve_general(constraints, start_x, end_x)
                del constraints
            del projected
        ds = np.diff(self._grid)
        conditioned = self._validate_speed_profile(a, b, x)
        # A nearby stationary derivative can hide cancellation even when
        # the half-plane products themselves stay small.
        narrow = np.flatnonzero(ds < 1e-7)
        if narrow.size:
            conditioned = np.union1d(conditioned, narrow)
        if conditioned.size:
            x = self._repair_roundoff_profile(a, b, x, conditioned)
        # The immutable result owns its arrays; release large solver matrices
        # before constructing that snapshot and its validation scratch.
        del a, b, caps
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
        return ToppraResult(self._grid, speeds, u, times, float(times[-1]))

    def _solve_linear(
        self, a: np.ndarray, b: np.ndarray, start_x: float, end_x: float
    ) -> np.ndarray:
        # A two-waypoint natural spline has constant first derivatives and
        # zero second derivatives. Its acceleration strips bound only the
        # difference of adjacent squared speeds; prefix/suffix budgets give
        # the same reachable envelope without quadratic LP elimination.
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("path speed constraints must be finite")
        strip_count = (a.shape[1] - 2) // 2
        coefficient = np.max(np.abs(a[:, :strip_count]), axis=1)
        change = np.divide(
            1.0,
            coefficient,
            out=np.full_like(coefficient, np.inf),
            where=coefficient > 0,
        )
        inverse_cap = a[0, -2]
        cap = 1.0 / inverse_cap if inverse_cap > 0.0 else np.inf
        if end_x > cap + 1e-10 * max(end_x, cap):
            raise ValueError("boundary path velocity violates joint velocity limits")
        # An increment above the velocity cap cannot limit a feasible profile.
        change = np.minimum(change, cap)
        prefix = np.concatenate(([0.0], np.cumsum(change)))
        suffix = np.concatenate((np.cumsum(change[::-1])[::-1], [0.0]))
        upper = np.minimum(cap, end_x + suffix)
        lower_start = max(0.0, end_x - suffix[0])
        tolerance = 1e-12 * max(start_x, lower_start, upper[0])
        if not np.isfinite(upper[0]) or (
            start_x < lower_start - tolerance or start_x > upper[0] + tolerance
        ):
            raise ValueError("start velocity cannot reach the requested end velocity")
        x = np.minimum(start_x + prefix, upper)
        x[0], x[-1] = start_x, end_x
        return x

    @staticmethod
    def _validate_speed_profile(
        a: np.ndarray, b: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Check half-planes and return rows needing a physical-unit check."""
        conditioned = []
        violated = None
        chunk = max(1, (1 << 14) // a.shape[1])
        for begin in range(0, a.shape[0], chunk):
            end = min(a.shape[0], begin + chunk)
            left = a[begin:end] * x[begin:end, None]
            right = b[begin:end] * x[begin + 1 : end + 1, None]
            tolerance = 1e-10 * np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
            if np.any(left + right - 1.0 > tolerance):
                failures = left + right - 1.0 > tolerance
                if np.any(failures[:, -2:]):
                    raise RuntimeError(
                        "TOPPRA forward pass violates interval constraints"
                    )
                if violated is None:
                    violated = []
                violated.append(begin + np.flatnonzero(np.any(failures, axis=1)))
            # Below this product scale, rounding has ample headroom against
            # the physical acceleration check's 1e-8 tolerance.
            if np.max(tolerance, initial=0.0) > 1e-5:
                conditioned.append(
                    begin + np.flatnonzero(np.any(tolerance > 1e-5, axis=1))
                )
        result = np.concatenate(conditioned) if conditioned else np.empty(0, dtype=int)
        if violated is not None:
            if result.size == 0:
                raise RuntimeError("TOPPRA forward pass violates interval constraints")
            # Cancellation can affect the reachable envelope outside the
            # ill-conditioned interval itself. Every failed acceleration row
            # must enter physical verification; velocity caps were checked
            # above and ordinary paths retain the original invariant.
            result = np.union1d(result, np.concatenate(violated))
        return result

    def _acceleration_bounds(self, x: np.ndarray, intervals: np.ndarray) -> np.ndarray:
        """Bound physical acceleration without subtracting large LP products."""
        result = np.empty(intervals.size)
        chunk = max(1, (1 << 13) // self.dof)
        widths = np.diff(self._grid)
        for begin in range(0, intervals.size, chunk):
            index = intervals[begin : begin + chunk]
            ds = widths[index, None]
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                first, last, second_first, second_last = (
                    _scale_units(
                        block,
                        1.0,
                        self._speed_scale,
                        2,
                        divisors=(self.max_acceleration,),
                    )
                    for block in (
                        self._q_s[index],
                        self._q_s[index + 1],
                        self._q_ss[index],
                        self._q_ss[index + 1],
                    )
                )
                initial = np.sqrt(x[index])[:, None] ** 2
                u = ((x[index + 1] - x[index]) / (2.0 * ds[:, 0]))[:, None]
                final = initial + 2.0 * ds * u
                middle_correction = 0.5 * ds * second_first
                middle = first + middle_correction
                left_curvature = second_first * initial
                right_curvature = second_last * final
                cross_left = second_last * initial
                cross_right = second_first * final
                first_u, last_u = first * u, last * u
                g0 = left_curvature + first_u
                g1 = 0.5 * (cross_left + cross_right) + middle * u
                g2 = right_curvature + last_u
                # Include cancellation, unit restoration and integration errors
                # in the certificate; a rounded zero alone is not a safe bound.
                rounding = 64.0 * np.finfo(float).eps
                r0 = np.abs(left_curvature) + np.abs(first_u)
                r1 = (
                    0.5 * (np.abs(cross_left) + np.abs(cross_right))
                    + np.abs(first_u)
                    + np.abs(middle_correction * u)
                )
                r2 = np.abs(right_curvature) + np.abs(last_u)
                # Near a stationary point, derivative evaluation can cancel
                # large spline terms. Bound those terms before cancellation,
                # including the rounding of the integrated terminal speed.
                segment, local = self._path_model._segments(self._grid[index])
                coefficients = self._path_model._normalized_coefficients
                coordinate_scale = self._path_model._coefficient_scale
                if coefficients is None:
                    coefficients = (
                        self._path_model.b,
                        self._path_model.c,
                        self._path_model.d,
                    )
                    coordinate_scale = 1.0
                cb, cc, cd = (
                    np.abs(
                        _scale_units(
                            block[segment],
                            coordinate_scale,
                            self._speed_scale,
                            2,
                            divisors=(self.max_acceleration,),
                        )
                    )
                    for block in coefficients
                )
                far = np.abs(local) + ds
                first_magnitude = cb + far * (2.0 * cc + 3.0 * far * cd)
                second_magnitude = 2.0 * cc + 6.0 * far * cd
                state_magnitude = np.abs(initial) + np.abs(2.0 * ds * u)
                geometry_magnitude = (
                    second_magnitude * state_magnitude + first_magnitude * np.abs(u)
                )
                r0 += geometry_magnitude
                r1 += geometry_magnitude
                r2 += geometry_magnitude
                result[begin : begin + index.size] = np.maximum.reduce(
                    (
                        np.max(np.abs(g0) + rounding * r0, axis=1),
                        np.max(np.abs(g1) + rounding * r1, axis=1),
                        np.max(np.abs(g2) + rounding * r2, axis=1),
                    )
                )
        return result

    def _repair_roundoff_profile(
        self, a: np.ndarray, b: np.ndarray, x: np.ndarray, conditioned: np.ndarray
    ) -> np.ndarray:
        """Keep endpoint speeds and certify a conservative timing if needed."""
        tolerance = 1e-8
        bounds = self._acceleration_bounds(x, conditioned)
        bad = conditioned[~np.isfinite(bounds) | (bounds > 1.0 + tolerance)]
        if bad.size == 0:
            return x
        intervals = np.arange(x.size - 1)

        def acceptable(trial):
            if (
                not np.isfinite(trial).all()
                or np.any(trial < 0.0)
                or np.any((trial[:-1] == 0.0) & (trial[1:] == 0.0))
            ):
                return False
            if trial[0] != x[0] or trial[-1] != x[-1]:
                return False
            # Acceleration is certified directly below. Its cancellation-
            # sensitive half-plane residual must not reject a physical repair.
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                velocity = np.maximum(a[:, -2] * trial[:-1], b[:, -1] * trial[1:])
            if not np.isfinite(velocity).all() or np.any(velocity > 1.0 + 1e-10):
                return False
            values = self._acceleration_bounds(trial, intervals)
            return np.isfinite(values).all() and np.all(values <= 1.0 + tolerance)

        def flatten(indices):
            trial = x.copy()
            if indices.size == 0:
                return trial
            groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            for group in groups:
                begin, end = int(group[0]), int(group[-1]) + 1
                if begin == 0 and end == x.size - 1 and x[0] != x[-1]:
                    continue
                value = (
                    x[0]
                    if begin == 0
                    else x[-1]
                    if end == x.size - 1
                    else np.min(x[begin : end + 1])
                )
                trial[begin : end + 1] = value
            return trial

        # Preserve the inexpensive local repair when it already certifies
        # the whole path. Additional uncertain intervals can require their
        # own plateau before a small global correction is useful.
        trial = flatten(bad)
        if acceptable(trial):
            return trial
        full_bounds = self._acceleration_bounds(x, intervals)
        all_bad = np.flatnonzero(
            ~np.isfinite(full_bounds) | (full_bounds > 1.0 + tolerance)
        )
        expanded = flatten(all_bad)

        # Bridge an uncertain neighborhood between unchanged outer nodes.
        # Gridpoints and spline knots remain in place; only squared speeds
        # are interpolated, and every resulting interval is still certified.
        bridged = x.copy()
        neighborhood = np.unique(
            np.clip(np.concatenate((all_bad - 1, all_bad, all_bad + 1)), 0, x.size - 2)
        )
        groups = (
            np.split(neighborhood, np.flatnonzero(np.diff(neighborhood) > 1) + 1)
            if neighborhood.size
            else []
        )
        for group in groups:
            begin, end = int(group[0]), int(group[-1]) + 1
            fraction = (self._grid[begin : end + 1] - self._grid[begin]) / (
                self._grid[end] - self._grid[begin]
            )
            bridged[begin : end + 1] = x[begin] + fraction * (x[end] - x[begin])
            bridged[begin], bridged[end] = x[begin], x[end]

        # Every proposal retains the requested endpoints. Keep the original
        # blend as a candidate and choose the shortest verified timing, so
        # a conservative plateau cannot worsen an existing feasible repair.
        base = x[0] + self._grid * (x[-1] - x[0])
        base[0], base[-1] = x[0], x[-1]
        base_bound = float(np.max(self._acceleration_bounds(base, intervals)))

        def blend(profile, full_bound):
            if not np.isfinite(base_bound) or base_bound >= 1.0:
                return None
            if np.isfinite(full_bound):
                weight = (
                    1.0
                    if full_bound <= 1.0
                    else min(1.0, (1.0 - base_bound) / (full_bound - base_bound))
                )
            else:
                weight = 0.5
            for iteration in range(12):
                candidate = base + weight * (profile - base)
                candidate[0], candidate[-1] = x[0], x[-1]
                if acceptable(candidate):
                    return candidate
                if iteration < 3:
                    bound = float(
                        np.max(self._acceleration_bounds(candidate, intervals))
                    )
                    if np.isfinite(bound):
                        weight *= max(
                            0.0,
                            1.0 - 2.0 * max(bound - 1.0, 64.0 * np.finfo(float).eps),
                        )
                    else:
                        weight *= 0.5
                else:
                    weight *= 0.5
            return None

        best, best_time = None, np.inf
        proposals = (x, trial, expanded, bridged)
        for index, profile in enumerate(proposals):
            if any(np.array_equal(profile, previous) for previous in proposals[:index]):
                continue
            if acceptable(profile):
                candidate = profile
            else:
                bound = float(np.max(self._acceleration_bounds(profile, intervals)))
                candidate = blend(profile, bound)
            if candidate is None:
                continue
            speeds = np.sqrt(candidate)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                dt = (
                    2.0 * np.diff(self._grid) / (speeds[:-1] + speeds[1:])
                ) / self._speed_scale
                times = np.cumsum(dt)
            if (
                not np.isfinite(times).all()
                or np.any(dt <= 0.0)
                or np.any(np.diff(times) <= 0.0)
            ):
                continue
            time = float(times[-1])
            if time < best_time:
                best, best_time = candidate, time
        if best is not None:
            return best
        raise ValueError(
            "path timing is not numerically resolvable; adjust gridpoints or waypoint spacing"
        )

    def _solve_to_rest(
        self, a: np.ndarray, b: np.ndarray, projected: np.ndarray, start_x: float
    ) -> np.ndarray:
        # Rest satisfies every half-plane. With a zero terminal speed, each
        # backward controllable interval therefore has lower bound zero.
        # Keep only its upper bound, including for a nonzero initial speed.
        positive_a = a > 0.0
        fixed = positive_a & (b >= 0.0)
        reciprocal = np.divide(1.0, a, out=np.full_like(a, np.inf), where=fixed)
        zero_cap = np.minimum(projected, np.min(reciprocal, axis=1, initial=np.inf))
        del reciprocal, fixed

        # Pack variable-length rows once. Boolean selection preserves each
        # row's constraint order while avoiding per-interval objects and
        # repeated small allocations during preparation.
        coupled = positive_a & (b < 0.0)
        back_a, back_b = a[coupled], b[coupled]
        back_offsets = np.concatenate(
            ([0], np.cumsum(np.count_nonzero(coupled, axis=1)))
        )
        positive_b = b > 0.0
        forward_a, forward_b = a[positive_b], b[positive_b]
        forward_offsets = np.concatenate(
            ([0], np.cumsum(np.count_nonzero(positive_b, axis=1)))
        )
        del positive_a, coupled, positive_b

        controllable = np.zeros(self._grid.size)
        for index in range(a.shape[0] - 1, -1, -1):
            begin, end = back_offsets[index], back_offsets[index + 1]
            cap = min(
                zero_cap[index],
                float(
                    np.min(
                        (1.0 - back_b[begin:end] * controllable[index + 1])
                        / back_a[begin:end],
                        initial=np.inf,
                    )
                ),
            )
            if not np.isfinite(cap) or cap < 0.0:
                raise ValueError(f"path is infeasible near gridpoint {index}")
            controllable[index] = cap
        if start_x > controllable[0] + 1e-12 * max(start_x, controllable[0]):
            raise ValueError("start velocity cannot reach the requested end velocity")
        x = np.empty_like(self._grid)
        x[0] = start_x
        rounding_eps = 8.0 * np.finfo(float).eps
        for index in range(a.shape[0]):
            begin, end = forward_offsets[index], forward_offsets[index + 1]
            aa, bb = forward_a[begin:end], forward_b[begin:end]
            upper = controllable[index + 1]
            product = aa * x[index]
            rounding = rounding_eps * np.maximum(1.0, np.abs(product))
            residual = 1.0 - product
            # A roundoff-only residual must preserve an exact stop, except
            # when b*y itself is negligible over the whole reachable range.
            stop = (np.abs(residual) <= rounding) & (bb * upper > rounding)
            rhs = np.where(stop, 0.0, residual + rounding)
            x[index + 1] = max(0.0, min(upper, float(np.min(rhs / bb, initial=np.inf))))
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
        q, q_s, q_ss = self._path_model._evaluate(s)
        coordinate_scale = self._path_model._coefficient_scale
        if coordinate_scale is not None:
            # Each derivative buffer is owned by this query. Restore the
            # physical units only after combining normalized motion terms.
            velocity = q_s * speed[..., None]
            if q_s.size == 1:
                q_ss = (q_ss * speed[..., None]) * speed[..., None] + q_s * accel[
                    ..., None
                ]
            else:
                q_ss *= speed[..., None]
                q_ss *= speed[..., None]
                q_s *= accel[..., None]
                q_ss += q_s
            del q_s
            velocity_factor, acceleration_factor = self._sample_unit_factors
            if velocity_factor is None:
                velocity = _scale_units(
                    velocity, coordinate_scale, self._speed_scale, 1
                )
            else:
                velocity *= velocity_factor
            if acceleration_factor is None:
                q_ss = _scale_units(q_ss, coordinate_scale, self._speed_scale, 2)
            else:
                q_ss *= acceleration_factor
            return q, velocity, q_ss
        if self._normalized_acceleration_constraints:
            # Restore speed units first when large physical derivatives
            # cannot be multiplied by the normalized speeds safely.
            speed = speed * self._speed_scale
            accel = (accel * self._speed_scale) * self._speed_scale
        velocity = q_s * speed[..., None]
        if q_s.size == 1:
            q_ss = q_ss * speed[..., None] ** 2 + q_s * accel[..., None]
        else:
            q_ss *= speed[..., None] ** 2
            q_s *= accel[..., None]
            q_ss += q_s
        del q_s
        if not self._normalized_acceleration_constraints and self._speed_scale != 1.0:
            velocity *= self._speed_scale
            q_ss *= self._speed_scale
            q_ss *= self._speed_scale
        return q, velocity, q_ss

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
