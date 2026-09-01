"""Shared validation helpers for public Python value objects."""

from __future__ import annotations

import numpy as np

_IDENTITY_ROTATION = np.eye(3)
_IDENTITY_ROTATION.setflags(write=False)
_HOMOGENEOUS_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0])
_HOMOGENEOUS_LAST_ROW.setflags(write=False)


def validated_transform(value, *, name: str, readonly: bool = False) -> np.ndarray:
    """Return an owned, validated homogeneous SE(3) transform."""

    transform = np.array(value, dtype=float, copy=True)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], _HOMOGENEOUS_LAST_ROW, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must have a valid homogeneous last row")
    rotation = transform[:3, :3]
    determinant = (
        rotation[0, 0]
        * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
        - rotation[0, 1]
        * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
        + rotation[0, 2]
        * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
    )
    if (
        not np.allclose(rotation.T @ rotation, _IDENTITY_ROTATION, rtol=1e-5, atol=1e-6)
        or abs(determinant - 1.0) > 1.1e-5
    ):
        raise ValueError(f"{name} rotation must be a proper orthonormal matrix")
    if readonly:
        transform.setflags(write=False)
    return transform
