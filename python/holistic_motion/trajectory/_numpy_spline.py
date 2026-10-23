"""Natural cubic path geometry for NumPy retiming."""

from __future__ import annotations

import numpy as np

_LARGE_GEOMETRY = np.sqrt(np.finfo(float).max) / 16.0


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
