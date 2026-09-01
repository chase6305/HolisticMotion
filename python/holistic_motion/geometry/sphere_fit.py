"""Dependency-light collision-sphere fitting and serialization.

The fitter works on caller-provided interior and surface samples.  Mesh I/O is
kept in :func:`fit_trimesh`, which imports the optional ``trimesh`` dependency
only when called.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]


def _finite_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_sphere_links(links) -> dict[str, tuple[SphereSpec, ...]]:
    if not isinstance(links, Mapping) or not links:
        raise ValueError("links must be a non-empty mapping")
    normalized = {}
    for link, spheres in links.items():
        if not isinstance(link, str) or not link:
            raise ValueError("each link requires a non-empty string name")
        try:
            owned = tuple(spheres)
        except TypeError as error:
            raise TypeError(f"link {link!r} spheres must be iterable") from error
        if not owned:
            raise ValueError("each link requires at least one sphere")
        if any(not isinstance(sphere, SphereSpec) for sphere in owned):
            raise TypeError(f"link {link!r} entries must be SphereSpec objects")
        normalized[link] = owned
    return normalized


def _validated_spheres(spheres) -> tuple[SphereSpec, ...]:
    try:
        normalized = tuple(spheres)
    except TypeError as error:
        raise TypeError("spheres must be an iterable of SphereSpec objects") from error
    if not normalized:
        raise ValueError("spheres must not be empty")
    if any(not isinstance(sphere, SphereSpec) for sphere in normalized):
        raise TypeError("spheres must contain only SphereSpec objects")
    return normalized


def _validated_options(options) -> SphereFitOptions:
    if options is None:
        return SphereFitOptions()
    if not isinstance(options, SphereFitOptions):
        raise TypeError("options must be SphereFitOptions or None")
    return options


def _points(value, name: str, *, allow_empty: bool = False) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if not allow_empty and len(points) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain finite values")
    return points


def _minimum_distances(
    query: np.ndarray, reference: np.ndarray, chunk_size: int
) -> np.ndarray:
    output = np.empty(len(query), dtype=float)
    reference_norm = np.sum(reference * reference, axis=1)[None, :]
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        squared = (
            np.sum(chunk * chunk, axis=1)[:, None]
            + reference_norm
            - 2.0 * chunk @ reference.T
        )
        np.maximum(squared, 0.0, out=squared)
        output[start : start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
    return output


@dataclass(frozen=True)
class SphereSpec:
    """One sphere in a link-local coordinate frame."""

    center: tuple[float, float, float]
    radius: float

    def __post_init__(self):
        center = np.asarray(self.center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("sphere center must contain three finite values")
        radius = _finite_float(self.radius, "sphere radius")
        if radius <= 0.0:
            raise ValueError("sphere radius must be finite and positive")
        object.__setattr__(self, "center", tuple(float(x) for x in center))
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class SphereFitOptions:
    """Controls deterministic greedy sphere selection."""

    max_spheres: int = 32
    min_radius: float = 0.002
    padding: float = 0.0
    sampled_coverage: bool = False
    chunk_size: int = 512

    def __post_init__(self):
        if (
            not isinstance(self.max_spheres, Integral)
            or isinstance(self.max_spheres, bool)
            or self.max_spheres < 1
        ):
            raise ValueError("max_spheres must be a positive integer")
        if (
            not isinstance(self.chunk_size, Integral)
            or isinstance(self.chunk_size, bool)
            or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer")
        min_radius = _finite_float(self.min_radius, "min_radius")
        padding = _finite_float(self.padding, "padding")
        if min_radius <= 0.0:
            raise ValueError("min_radius must be finite and positive")
        if padding < 0.0:
            raise ValueError("padding must be finite and non-negative")
        if not isinstance(self.sampled_coverage, (bool, np.bool_)):
            raise TypeError("sampled_coverage must be boolean")
        object.__setattr__(self, "max_spheres", int(self.max_spheres))
        object.__setattr__(self, "min_radius", min_radius)
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "sampled_coverage", bool(self.sampled_coverage))
        object.__setattr__(self, "chunk_size", int(self.chunk_size))


@dataclass(frozen=True)
class SphereFitMetrics:
    sampled_coverage: float
    mean_uncovered_distance: float
    maximum_uncovered_distance: float
    sphere_count: int

    def __post_init__(self) -> None:
        coverage = _finite_float(self.sampled_coverage, "sampled_coverage")
        mean_distance = _finite_float(
            self.mean_uncovered_distance, "mean_uncovered_distance"
        )
        maximum_distance = _finite_float(
            self.maximum_uncovered_distance, "maximum_uncovered_distance"
        )
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("sampled_coverage must be finite and lie in [0, 1]")
        if mean_distance < 0.0 or maximum_distance < 0.0:
            raise ValueError("uncovered distances must be finite and non-negative")
        if mean_distance > maximum_distance + 1e-12:
            raise ValueError("mean uncovered distance cannot exceed the maximum")
        if not isinstance(self.sphere_count, Integral) or isinstance(
            self.sphere_count, bool
        ):
            raise TypeError("sphere_count must be an integer")
        if self.sphere_count < 1:
            raise ValueError("sphere_count must be positive")
        object.__setattr__(self, "sampled_coverage", coverage)
        object.__setattr__(self, "mean_uncovered_distance", mean_distance)
        object.__setattr__(self, "maximum_uncovered_distance", maximum_distance)
        object.__setattr__(self, "sphere_count", int(self.sphere_count))


@dataclass(frozen=True)
class SphereFitResult:
    spheres: tuple[SphereSpec, ...]
    metrics: SphereFitMetrics
    mode: str

    def __post_init__(self) -> None:
        spheres = tuple(self.spheres)
        if not spheres or any(not isinstance(sphere, SphereSpec) for sphere in spheres):
            raise ValueError("spheres must contain at least one SphereSpec")
        if not isinstance(self.metrics, SphereFitMetrics):
            raise TypeError("metrics must be SphereFitMetrics")
        if self.metrics.sphere_count != len(spheres):
            raise ValueError("metrics sphere_count must match spheres")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode must be a non-empty string")
        object.__setattr__(self, "spheres", spheres)


def evaluate_sphere_fit(
    sample_points, spheres: Sequence[SphereSpec], *, chunk_size: int = 2048
) -> SphereFitMetrics:
    """Evaluate coverage against a finite sample set.

    These metrics do not prove continuous triangle-mesh coverage.
    """
    points = _points(sample_points, "sample_points")
    spheres = _validated_spheres(spheres)
    if (
        not isinstance(chunk_size, Integral)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    centers = np.asarray([sphere.center for sphere in spheres], dtype=float)
    radii = np.asarray([sphere.radius for sphere in spheres], dtype=float)
    gaps = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        distances = (
            np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2)
            - radii[None, :]
        )
        gaps[start : start + len(chunk)] = np.maximum(np.min(distances, axis=1), 0.0)
    uncovered = gaps > 1e-12
    return SphereFitMetrics(
        sampled_coverage=float(np.mean(~uncovered)),
        mean_uncovered_distance=float(np.mean(gaps)),
        maximum_uncovered_distance=float(np.max(gaps)),
        sphere_count=len(spheres),
    )


def fit_spheres(
    interior_points,
    surface_points,
    options: Optional[SphereFitOptions] = None,
) -> SphereFitResult:
    """Fit spheres from interior candidates and surface samples.

    In the default ``inscribed`` mode, each candidate radius is its sampled
    distance to the surface.  ``sampled_coverage=True`` expands selected
    spheres just enough to cover the supplied samples; this is useful for a
    broad phase but is not a proof that the continuous mesh is covered.
    """
    options = _validated_options(options)
    interior = _points(interior_points, "interior_points")
    surface = _points(surface_points, "surface_points")
    radii = _minimum_distances(interior, surface, options.chunk_size)
    eligible = np.flatnonzero(radii >= options.min_radius)
    if len(eligible) == 0:
        raise ValueError("no interior candidate satisfies min_radius")

    # Prefer large medial candidates while suppressing centers already covered
    # by an earlier sphere. Stable sorting makes equal inputs deterministic.
    order = eligible[np.argsort(-radii[eligible], kind="stable")]
    selected = []
    for index in order:
        center = interior[index]
        radius = radii[index]
        if selected:
            selected_centers = interior[np.asarray(selected)]
            selected_radii = radii[np.asarray(selected)]
            if np.any(
                np.linalg.norm(selected_centers - center, axis=1)
                <= np.maximum(selected_radii - radius, 0.0)
            ):
                continue
        selected.append(int(index))
        if len(selected) == options.max_spheres:
            break
    if not selected:
        raise RuntimeError("sphere selection produced no candidates")

    centers = interior[np.asarray(selected)]
    selected_radii = radii[np.asarray(selected)].copy()
    mode = "inscribed"
    if options.sampled_coverage:
        all_samples = np.concatenate((interior, surface), axis=0)
        maximum_assigned = np.zeros(len(centers), dtype=float)
        for start in range(0, len(all_samples), options.chunk_size):
            samples = all_samples[start : start + options.chunk_size]
            distances = np.linalg.norm(
                samples[:, None, :] - centers[None, :, :], axis=2
            )
            assignments = np.argmin(distances, axis=1)
            assigned_distances = distances[np.arange(len(samples)), assignments]
            np.maximum.at(maximum_assigned, assignments, assigned_distances)
        selected_radii = np.maximum(selected_radii, maximum_assigned)
        mode = "sampled_coverage"
    selected_radii += options.padding

    spheres = tuple(
        SphereSpec(tuple(center), radius)
        for center, radius in zip(centers, selected_radii)
    )
    metrics = evaluate_sphere_fit(
        np.concatenate((interior, surface), axis=0),
        spheres,
        chunk_size=options.chunk_size,
    )
    return SphereFitResult(spheres=spheres, metrics=metrics, mode=mode)


def fit_trimesh(
    mesh,
    options: Optional[SphereFitOptions] = None,
    *,
    pitch: Optional[float] = None,
    surface_samples: int = 5000,
    random_seed: int = 0,
) -> SphereFitResult:
    """Voxelize a ``trimesh.Trimesh`` and fit spheres in mesh coordinates."""
    import trimesh

    options = _validated_options(options)
    if (
        not isinstance(surface_samples, int)
        or isinstance(surface_samples, bool)
        or surface_samples < 4
    ):
        raise ValueError("surface_samples must be at least four")
    extents = np.asarray(mesh.extents, dtype=float)
    if (
        extents.shape != (3,)
        or not np.all(np.isfinite(extents))
        or np.max(extents) <= 0
    ):
        raise ValueError("mesh must have finite, non-zero extents")
    if pitch is None:
        pitch = float(np.max(extents) / 32.0)
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pitch must be finite and positive")

    voxels = mesh.voxelized(pitch).fill()
    interior = np.asarray(voxels.points, dtype=float)
    if len(interior) == 0:
        raise ValueError("mesh voxelization produced no interior samples")
    # Use the module-level sampler because it exposes an explicit seed across
    # supported trimesh releases, unlike Trimesh.sample in some versions.
    surface, _ = trimesh.sample.sample_surface(mesh, surface_samples, seed=random_seed)
    return fit_spheres(interior, surface, options)


def save_sphere_model(
    path: PathLike,
    links: dict[str, Sequence[SphereSpec]],
    *,
    metadata: Optional[dict] = None,
) -> None:
    """Write a versioned, link-local sphere model as JSON."""
    normalized_links = _validated_sphere_links(links)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    payload = {
        "format": "holistic_motion.sphere_model",
        "version": 1,
        "links": {
            link: [
                {"center": list(sphere.center), "radius": sphere.radius}
                for sphere in spheres
            ]
            for link, spheres in sorted(normalized_links.items())
        },
        "metadata": dict(metadata or {}),
    }
    target = Path(path)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_sphere_model(path: PathLike) -> tuple[dict[str, tuple[SphereSpec, ...]], dict]:
    """Load and validate a version-1 sphere-model JSON file."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError("sphere model root must be an object")
    if (
        payload.get("format") != "holistic_motion.sphere_model"
        or payload.get("version") != 1
    ):
        raise ValueError("unsupported sphere model format or version")
    raw_links = payload.get("links")
    if not isinstance(raw_links, dict) or not raw_links:
        raise ValueError("sphere model links must be a non-empty object")
    links = {}
    for link, raw_spheres in raw_links.items():
        if (
            not isinstance(link, str)
            or not link
            or not isinstance(raw_spheres, list)
            or not raw_spheres
        ):
            raise ValueError("each link requires a name and at least one sphere")
        try:
            links[link] = tuple(
                SphereSpec(tuple(item["center"]), item["radius"])
                for item in raw_spheres
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"link {link!r} contains an invalid sphere entry"
            ) from error
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("sphere model metadata must be an object")
    return links, metadata


def make_collision_spheres(links: dict[str, Sequence[SphereSpec]]):
    """Convert link-local specs to native :class:`CollisionSphere` objects."""
    normalized_links = _validated_sphere_links(links)
    try:
        from holistic_motion import CollisionSphere
    except ImportError as error:
        raise RuntimeError(
            "native collision bindings are required to create CollisionSphere objects"
        ) from error
    output = []
    for link, spheres in sorted(normalized_links.items()):
        output.extend(
            CollisionSphere(
                f"{link}_sphere_{index}", link, sphere.center, sphere.radius
            )
            for index, sphere in enumerate(spheres)
        )
    return output
