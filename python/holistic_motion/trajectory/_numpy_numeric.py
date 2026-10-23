"""Range-safe unit conversion for NumPy retiming kernels."""

from __future__ import annotations

import numpy as np


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
