"""Task definitions for Pink-style differential inverse kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np

Cost = Union[float, Sequence[float]]


def _cost_vector(value: Cost, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(size, float(result))
    result = result.reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite scalar or vector of size {size}")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrameTask:
    """An anisotropically weighted world-frame pose task."""

    position_cost: Cost = 1.0
    orientation_cost: Cost = 1.0
    gain: float = 1.0
    lm_damping: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_cost", _cost_vector(self.position_cost, 3, "position_cost")
        )
        object.__setattr__(
            self,
            "orientation_cost",
            _cost_vector(self.orientation_cost, 3, "orientation_cost"),
        )
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
        if not np.isfinite(self.lm_damping) or self.lm_damping < 0.0:
            raise ValueError("lm_damping must be finite and non-negative")

    @property
    def cost(self) -> np.ndarray:
        return np.concatenate((self.position_cost, self.orientation_cost))


@dataclass(frozen=True)
class PostureTask:
    """Regularize the solution toward a preferred configuration."""

    cost: float = 1e-3
    gain: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("posture cost must be finite and non-negative")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
