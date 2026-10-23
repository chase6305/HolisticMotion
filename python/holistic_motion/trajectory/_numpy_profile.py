"""Speed-profile feasibility certificates and conservative roundoff repair."""

from __future__ import annotations

import numpy as np

from ._numpy_numeric import _scale_units


def _validate_speed_profile(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
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
                raise RuntimeError("TOPPRA forward pass violates interval constraints")
            if violated is None:
                violated = []
            violated.append(begin + np.flatnonzero(np.any(failures, axis=1)))
        # Below this product scale, rounding has ample headroom against
        # the physical acceleration check's 1e-8 tolerance.
        if np.max(tolerance, initial=0.0) > 1e-5:
            conditioned.append(begin + np.flatnonzero(np.any(tolerance > 1e-5, axis=1)))
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


def _acceleration_bounds(
    trajectory, x: np.ndarray, intervals: np.ndarray
) -> np.ndarray:
    """Bound physical acceleration without subtracting large LP products."""
    result = np.empty(intervals.size)
    chunk = max(1, (1 << 13) // trajectory.dof)
    widths = np.diff(trajectory._grid)
    for begin in range(0, intervals.size, chunk):
        index = intervals[begin : begin + chunk]
        ds = widths[index, None]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            first, last, second_first, second_last = (
                _scale_units(
                    block,
                    1.0,
                    trajectory._speed_scale,
                    2,
                    divisors=(trajectory.max_acceleration,),
                )
                for block in (
                    trajectory._q_s[index],
                    trajectory._q_s[index + 1],
                    trajectory._q_ss[index],
                    trajectory._q_ss[index + 1],
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
            segment, local = trajectory._path_model._segments(trajectory._grid[index])
            coefficients = trajectory._path_model._normalized_coefficients
            coordinate_scale = trajectory._path_model._coefficient_scale
            if coefficients is None:
                coefficients = (
                    trajectory._path_model.b,
                    trajectory._path_model.c,
                    trajectory._path_model.d,
                )
                coordinate_scale = 1.0
            cb, cc, cd = (
                np.abs(
                    _scale_units(
                        block[segment],
                        coordinate_scale,
                        trajectory._speed_scale,
                        2,
                        divisors=(trajectory.max_acceleration,),
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
    trajectory, a: np.ndarray, b: np.ndarray, x: np.ndarray, conditioned: np.ndarray
) -> np.ndarray:
    """Keep endpoint speeds and certify a conservative timing if needed."""
    tolerance = 1e-8
    bounds = trajectory._acceleration_bounds(x, conditioned)
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
        values = trajectory._acceleration_bounds(trial, intervals)
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
    full_bounds = trajectory._acceleration_bounds(x, intervals)
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
        fraction = (trajectory._grid[begin : end + 1] - trajectory._grid[begin]) / (
            trajectory._grid[end] - trajectory._grid[begin]
        )
        bridged[begin : end + 1] = x[begin] + fraction * (x[end] - x[begin])
        bridged[begin], bridged[end] = x[begin], x[end]

    # Every proposal retains the requested endpoints. Keep the original
    # blend as a candidate and choose the shortest verified timing, so
    # a conservative plateau cannot worsen an existing feasible repair.
    base = x[0] + trajectory._grid * (x[-1] - x[0])
    base[0], base[-1] = x[0], x[-1]
    base_bound = float(np.max(trajectory._acceleration_bounds(base, intervals)))

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
                    np.max(trajectory._acceleration_bounds(candidate, intervals))
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
            bound = float(np.max(trajectory._acceleration_bounds(profile, intervals)))
            candidate = blend(profile, bound)
        if candidate is None:
            continue
        speeds = np.sqrt(candidate)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            dt = (
                2.0 * np.diff(trajectory._grid) / (speeds[:-1] + speeds[1:])
            ) / trajectory._speed_scale
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
