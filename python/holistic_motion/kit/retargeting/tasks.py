"""Task definitions for Pink-style differential inverse kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union

import numpy as np

Cost = Union[float, Sequence[float]]


def _finite_scalar(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


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
    _cost: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_cost", _cost_vector(self.position_cost, 3, "position_cost")
        )
        object.__setattr__(
            self,
            "orientation_cost",
            _cost_vector(self.orientation_cost, 3, "orientation_cost"),
        )
        gain = _finite_scalar(self.gain, "gain")
        damping = _finite_scalar(self.lm_damping, "lm_damping")
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
        if damping < 0.0:
            raise ValueError("lm_damping must be finite and non-negative")
        cost = np.concatenate((self.position_cost, self.orientation_cost))
        cost.setflags(write=False)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "lm_damping", damping)
        object.__setattr__(self, "_cost", cost)

    @property
    def cost(self) -> np.ndarray:
        return self._cost


@dataclass(frozen=True)
class PostureTask:
    """Regularize the solution toward a preferred configuration."""

    cost: float = 1e-3
    gain: float = 1.0

    def __post_init__(self) -> None:
        cost = _finite_scalar(self.cost, "posture cost")
        gain = _finite_scalar(self.gain, "gain")
        if cost < 0.0:
            raise ValueError("posture cost must be finite and non-negative")
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "gain", gain)


@dataclass(frozen=True)
class CenterOfMassTask:
    """Regulate the robot center of mass in the world frame."""

    cost: Cost = 1.0
    gain: float = 1.0
    lm_damping: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost", _cost_vector(self.cost, 3, "cost"))
        gain = _finite_scalar(self.gain, "gain")
        damping = _finite_scalar(self.lm_damping, "lm_damping")
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
        if damping < 0.0:
            raise ValueError("lm_damping must be finite and non-negative")
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "lm_damping", damping)


@dataclass(frozen=True)
class SupportPolygonTask:
    """Penalize CoM or ZMP positions outside a convex XY polygon."""

    vertices: Sequence[Sequence[float]]
    cost: float = 1.0
    margin: float = 0.0
    gain: float = 1.0
    reference: str = "center_of_mass"

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        if (
            vertices.ndim != 2
            or vertices.shape[0] < 3
            or vertices.shape[1] != 2
            or not np.isfinite(vertices).all()
        ):
            raise ValueError(
                "support polygon must contain at least three finite XY vertices"
            )
        edges = np.roll(vertices, -1, axis=0) - vertices
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("support polygon edges must have positive length")
        next_edges = np.roll(edges, -1, axis=0)
        turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        if np.any(np.abs(turns) <= 1e-12) or np.any(turns * turns[0] < 0.0):
            raise ValueError("support polygon must be strictly convex")
        if turns[0] < 0.0:
            vertices = vertices[::-1].copy()
            edges = np.roll(vertices, -1, axis=0) - vertices
            lengths = np.linalg.norm(edges, axis=1)
        cost = _finite_scalar(self.cost, "support polygon cost")
        margin = _finite_scalar(self.margin, "support polygon margin")
        gain = _finite_scalar(self.gain, "gain")
        if cost < 0.0:
            raise ValueError("support polygon cost must be finite and non-negative")
        if margin < 0.0:
            raise ValueError("support polygon margin must be finite and non-negative")
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
        if self.reference not in ("center_of_mass", "zmp"):
            raise ValueError(
                "support polygon reference must be 'center_of_mass' or 'zmp'"
            )
        normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
        all_edge_distances = np.einsum(
            "evi,ei->ev", vertices[None, :, :] - vertices[:, None, :], normals
        )
        if np.any(all_edge_distances < -1e-12):
            raise ValueError("support polygon must be strictly convex and ordered")
        vertices = vertices.copy()
        normals = normals.copy()
        offsets = np.sum(normals * vertices, axis=1)
        vertices.setflags(write=False)
        normals.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "offsets", offsets)


@dataclass(frozen=True)
class ZmpTask:
    """Track a world-frame zero-moment-point approximation in XY."""

    cost: Cost = 1.0
    gain: float = 1.0
    gravity: float = 9.81
    lm_damping: float = 0.0
    plane_height: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost", _cost_vector(self.cost, 2, "cost"))
        gain = _finite_scalar(self.gain, "gain")
        gravity = _finite_scalar(self.gravity, "gravity")
        damping = _finite_scalar(self.lm_damping, "lm_damping")
        plane_height = _finite_scalar(self.plane_height, "plane_height")
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
        if gravity <= 0.0:
            raise ValueError("gravity must be finite and positive")
        if damping < 0.0:
            raise ValueError("lm_damping must be finite and non-negative")
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "lm_damping", damping)
        object.__setattr__(self, "plane_height", plane_height)
