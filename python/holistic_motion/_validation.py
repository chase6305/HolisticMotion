"""Shared validation helpers for public Python value objects."""

from __future__ import annotations

import numpy as np

_IDENTITY_ROTATION = np.eye(3)
_IDENTITY_ROTATION.setflags(write=False)
# Per-entry absolute and relative bounds for R.T @ R = I.
_ROTATION_TOLERANCE = 1e-6 + 1e-5 * _IDENTITY_ROTATION
_ROTATION_TOLERANCE.setflags(write=False)


def require_real(value, *, name: str):
    """Reject complex values before conversion.

    Python and NumPy float/integer scalars stay scalar; other inputs become
    NumPy arrays.
    """
    if isinstance(value, (float, int, np.floating, np.integer)):
        return value
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real")
    if array.dtype.kind == "O":
        _require_real_objects(array, name)
    return array


def _require_real_objects(array: np.ndarray, name: str) -> None:
    # Object arrays may hold further arrays, including zero-dimensional
    # wrappers that float() would silently convert from NumPy complex
    # scalars. Traverse iteratively, retaining only the active iterators.
    active = {id(array)}
    visited = {}
    pending = [(id(array), array, iter(array.flat))]
    while pending:
        identity, container, items = pending[-1]
        try:
            item = next(items)
        except StopIteration:
            pending.pop()
            active.remove(identity)
            visited[identity] = container
            continue
        if isinstance(item, (float, int, np.floating, np.integer)):
            continue
        identity = id(item)
        if identity in active:
            raise TypeError(f"{name} must not contain cyclic objects")
        if identity in visited:
            continue
        nested = np.asarray(item)
        if np.iscomplexobj(nested):
            raise TypeError(f"{name} must be real")
        if nested.dtype.kind == "O":
            # Ordinary scalar objects (for example Decimal) wrap themselves
            # in an object array; their numeric conversion is left intact.
            if (
                nested.ndim == 0
                and nested.item() is item
                and not isinstance(item, np.ndarray)
            ):
                continue
            active.add(identity)
            pending.append((identity, item, iter(nested.flat)))


def real_array(value, *, name: str, copy: bool = False) -> np.ndarray:
    array = require_real(value, name=name)
    return (
        np.array(array, dtype=float, copy=True)
        if copy
        else np.asarray(array, dtype=float)
    )


def validated_transform(value, *, name: str, readonly: bool = False) -> np.ndarray:
    """Return an owned, validated homogeneous SE(3) transform."""

    transform = real_array(value, name=name, copy=True)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    last_row = transform[3]
    last_row_error = max(
        abs(last_row[0]), abs(last_row[1]), abs(last_row[2]), abs(last_row[3] - 1.0)
    )
    if last_row_error > 1e-8:
        raise ValueError(f"{name} must have a valid homogeneous last row")
    rotation = transform[:3, :3]
    # Small discarded products do not invalidate a rotation. Large invalid
    # entries must produce a validation error under any NumPy error policy.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        orthogonal = np.all(
            np.abs(rotation.T @ rotation - _IDENTITY_ROTATION) <= _ROTATION_TOLERANCE
        )
        determinant = (
            rotation[0, 0]
            * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
            - rotation[0, 1]
            * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
            + rotation[0, 2]
            * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
        )
    if not orthogonal or abs(determinant - 1.0) > 1.1e-5:
        raise ValueError(f"{name} rotation must be a proper orthonormal matrix")
    if readonly:
        transform.setflags(write=False)
    return transform
