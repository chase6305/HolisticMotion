"""Interval reachability and projection kernels for NumPy retiming."""

from __future__ import annotations

import numpy as np


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
            stop = (np.abs(residual) <= rounding) & (np.abs(self.b) * upper > rounding)
            rhs[stop] = 0.0
        return _intersect_bounds(self.b, rhs, lower, upper)


def _pair_projection_caps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
        # All four selections own their buffers. Reuse them for the products,
        # determinant and bound without changing the arithmetic grouping.
        np.multiply(ai, bj, out=ai)
        np.multiply(bi, aj, out=aj)
        np.subtract(ai, aj, out=ai)
        np.abs(ai, out=ai)
        del aj
        valid = (bi != 0.0) & (bj != 0.0) & (ai > 0.0)
        np.abs(bi, out=bi)
        np.abs(bj, out=bj)
        np.add(bi, bj, out=bi)
        del bj
        np.divide(bi, ai, out=bi, where=valid)
        bi[~valid] = np.inf
        pair_cap = np.min(bi, axis=1, initial=np.inf)
        del ai, bi, valid

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


def _projection_caps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Screen large float64 problems before the complete pair elimination."""

    count = (a.shape[1] - 2) // 2
    if count < 48 or a.dtype != np.float64 or b.dtype != np.float64:
        return _pair_projection_caps(a, b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("path speed constraints must be finite")
    result = np.empty(a.shape[0])
    # Keep each normal matrix near 64 KiB, independently of the row count.
    chunk = max(1, min(4096, (1 << 13) // count))
    for offset in range(0, a.shape[0], chunk):
        result[offset : offset + chunk] = _screened_projection_caps(
            a[offset : offset + chunk], b[offset : offset + chunk]
        )
    return result


def _screened_projection_caps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Retain every pair that can improve a pilot cap, or use all pairs.

    With c = a*sign(b), d = abs(b), and any center z, a pair's inverse cap
    is at most max(abs(c_i+d_i*z), abs(c_j+d_j*z)). Exclude a pair only when
    both normals fall below the inverse pilot by a floating-point margin.
    Coefficient guards keep products normal and finite; uncertain rows use
    the complete elimination. The pilot search need not converge.
    """

    count = (a.shape[1] - 2) // 2
    aa, bb = a[:, :count], b[:, :count]
    # Exact opposite normals constrain only y-x. Their pair determinants
    # cancel bit for bit, including when their magnitudes differ by joint.
    unrestricted = np.all(aa == -bb, axis=1)
    nonzero = bb != 0.0
    c, d = aa * np.sign(bb), np.abs(bb)
    maximum = np.maximum(np.max(np.abs(c), axis=1), np.max(d, axis=1))
    minimum = np.minimum(
        np.min(np.where(c != 0.0, np.abs(c), np.inf), axis=1),
        np.min(np.where(nonzero, d, np.inf), axis=1),
    )
    eligible = (
        ~unrestricted
        & (maximum <= 1e100)
        & (minimum >= 1e-100)
        & (np.sum(nonzero, axis=1) >= 2)
    )
    result = np.full(a.shape[0], np.inf)
    indices = np.flatnonzero(eligible)
    if indices.size:
        oriented_a, absolute_b, nonzero_b = c[indices], d[indices], nonzero[indices]
        del c, d, nonzero, maximum, minimum
        ratios = np.divide(
            oriented_a, absolute_b, out=np.zeros_like(oriented_a), where=nonzero_b
        )
        inverse_weights = np.divide(
            1.0, absolute_b, out=np.zeros_like(absolute_b), where=nonzero_b
        )
        row = np.arange(indices.size)
        high = np.argmax(np.where(nonzero_b, ratios, -np.inf), axis=1)
        low = np.argmin(np.where(nonzero_b, ratios, np.inf), axis=1)

        def pair_cap(first, second):
            left, right = np.minimum(first, second), np.maximum(first, second)
            ai, aj = aa[indices, left], aa[indices, right]
            bi, bj = bb[indices, left], bb[indices, right]
            determinant = np.abs(ai * bj - bi * aj)
            return np.divide(
                np.abs(bi) + np.abs(bj),
                determinant,
                out=np.full(indices.size, np.inf),
                where=determinant > 0.0,
            )

        # Improve the pilot by intersecting upper/lower residual envelopes.
        # This is a bounded search for candidates, never a feasibility test.
        best = pair_cap(high, low)
        for _ in range(8):
            inverse = np.divide(
                1.0,
                best,
                out=np.zeros_like(best),
                where=np.isfinite(best) & (best > 0.0),
            )
            # A negligible pilot term can underflow even when every original
            # pair operation is normal. Keep the caller's policy for the LP.
            with np.errstate(under="ignore"):
                weighted = inverse[:, None] * inverse_weights
            upper_index = np.argmax(
                np.where(nonzero_b, ratios - weighted, -np.inf), axis=1
            )
            lower_index = np.argmin(
                np.where(nonzero_b, ratios + weighted, np.inf), axis=1
            )
            del weighted
            candidate = pair_cap(upper_index, lower_index)
            improved = candidate < best
            if not np.any(improved):
                break
            best[improved] = candidate[improved]
            high[improved], low[improved] = upper_index[improved], lower_index[improved]
        inverse = np.divide(
            1.0, best, out=np.zeros_like(best), where=np.isfinite(best) & (best > 0.0)
        )
        center = -(oriented_a[row, high] + oriented_a[row, low]) / (
            absolute_b[row, high] + absolute_b[row, low]
        )
        magnitude = np.max(np.abs(oriented_a), axis=1) + np.max(
            absolute_b, axis=1
        ) * np.abs(center)
        # Residual, determinant, sum, division and reciprocal errors together
        # are bounded by 32 eps times this magnitude in the guarded range.
        # A 128 eps gap leaves room for their rounding in either direction.
        tolerance = 128.0 * np.finfo(float).eps * np.maximum(magnitude, inverse)
        with np.errstate(under="ignore"):
            residual = np.abs(oriented_a + absolute_b * center[:, None])
        active = nonzero_b & (residual >= inverse[:, None] - tolerance[:, None])
        active[row, high] = True
        active[row, low] = True
        active_count = np.count_nonzero(active, axis=1)
        crowded = np.flatnonzero(active_count > 8)
        if crowded.size:
            # Equal oriented normals yield bit-identical absolute determinants
            # against every other strip. Keep one exact representative while
            # all original strips remain in the backward/forward solves.
            crowded_c, crowded_d = oriented_a[crowded], absolute_b[crowded]
            crowded_active = active[crowded]
            order = np.lexsort((crowded_d, crowded_c, ~crowded_active), axis=1)
            sorted_c = np.take_along_axis(crowded_c, order, axis=1)
            sorted_d = np.take_along_axis(crowded_d, order, axis=1)
            keep = np.empty_like(crowded_active)
            keep[:, 0] = True
            keep[:, 1:] = (sorted_c[:, 1:] != sorted_c[:, :-1]) | (
                sorted_d[:, 1:] != sorted_d[:, :-1]
            )
            keep &= np.take_along_axis(crowded_active, order, axis=1)
            unique = np.empty_like(crowded_active)
            np.put_along_axis(unique, order, keep, axis=1)
            active[crowded] = unique
            active_count[crowded] = np.count_nonzero(unique, axis=1)
            del (
                crowded_c,
                crowded_d,
                crowded_active,
                order,
                sorted_c,
                sorted_d,
                keep,
                unique,
            )
        accepted = (
            ((np.isfinite(best) & (best > 0.0)) | (active_count == 1))
            & np.isfinite(magnitude)
            & (active_count <= 8)
        )
        eligible[indices[~accepted]] = False
        accepted_rows, columns = np.nonzero(active & accepted[:, None])
        selected_rows = indices[accepted_rows]
        del (
            oriented_a,
            absolute_b,
            nonzero_b,
            ratios,
            inverse_weights,
            residual,
            active,
            center,
            magnitude,
            tolerance,
        )
        chunk = max(1, (1 << 13) // count)
        for offset in range(0, columns.size, chunk):
            rows = selected_rows[offset : offset + chunk]
            selected = columns[offset : offset + chunk]
            selected_a = aa[rows, selected, None]
            selected_b = bb[rows, selected, None]
            # Multiplication is commutative, and reversing the subtraction
            # changes only its sign. The absolute determinant and positive
            # sum therefore retain the complete pair formula's bits.
            determinant = aa[rows]
            product = bb[rows]
            determinant *= selected_b
            product *= selected_a
            determinant -= product
            del product
            np.abs(determinant, out=determinant)
            bound = bb[rows]
            valid = (bound != 0.0) & (selected_b != 0.0) & (determinant > 0.0)
            np.abs(bound, out=bound)
            bound += np.abs(selected_b)
            np.divide(bound, determinant, out=bound, where=valid)
            bound[~valid] = np.inf
            np.minimum.at(result, rows, np.min(bound, axis=1))
            del determinant, bound, valid

        eligible[indices[accepted]] = True
    eligible |= unrestricted
    if np.all(~eligible):
        return _pair_projection_caps(a, b)
    if np.any(~eligible):
        result[~eligible] = _pair_projection_caps(a[~eligible], b[~eligible])
    if np.any(eligible):
        av, bv = aa[eligible], bb[eligible]
        velocity_cap = b[eligible, -1:]
        divisor = np.abs(av) * velocity_cap
        valid = (((av < 0.0) & (bv > 0.0)) | ((av > 0.0) & (bv < 0.0))) & (
            divisor > 0.0
        )
        bound = np.abs(bv) + velocity_cap
        np.divide(bound, divisor, out=bound, where=valid)
        bound[~valid] = np.inf
        result[eligible] = np.minimum(
            result[eligible], np.min(bound, axis=1, initial=np.inf)
        )
    return result
