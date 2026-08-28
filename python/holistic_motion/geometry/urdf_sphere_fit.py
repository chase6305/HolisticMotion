"""URDF collision-geometry loading for offline sphere fitting."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .sphere_fit import SphereFitOptions, SphereFitResult, fit_trimesh

PathLike = Union[str, Path]


def _vector(text: Optional[str], size: int, default) -> np.ndarray:
    if text is None:
        if default is None:
            raise ValueError(f"required {size}-vector attribute is missing")
        return np.asarray(default, dtype=float)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"expected {size} finite values, got {text!r}")
    return values


def _origin_transform(element) -> np.ndarray:
    xyz = _vector(element.get("xyz") if element is not None else None, 3, [0, 0, 0])
    rpy = _vector(element.get("rpy") if element is not None else None, 3, [0, 0, 0])
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def _resolve_mesh_uri(uri: str, urdf_dir: Path, package_dirs: tuple[Path, ...]) -> Path:
    if not uri:
        raise ValueError("URDF mesh filename must not be empty")
    if uri.startswith("file://"):
        candidate = Path(uri[7:])
    elif uri.startswith("package://"):
        relative = Path(uri[len("package://") :])
        parts = relative.parts
        if len(parts) < 2:
            raise ValueError(f"invalid package mesh URI: {uri}")
        package, inside = parts[0], Path(*parts[1:])
        candidates = []
        for root in package_dirs:
            candidates.append(root / package / inside)
            if root.name == package:
                candidates.append(root / inside)
        for candidate_path in candidates:
            if candidate_path.is_file():
                return candidate_path.resolve()
        raise FileNotFoundError(
            f"cannot resolve {uri}; searched package roots: "
            + ", ".join(str(path) for path in package_dirs)
        )
    else:
        candidate = Path(uri)
        if not candidate.is_absolute():
            candidate = urdf_dir / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"collision mesh does not exist: {candidate}")
    return candidate.resolve()


def _as_mesh(loaded, trimesh):
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("collision mesh scene is empty")
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported collision geometry: {type(loaded).__name__}")
    return loaded


def load_urdf_collision_meshes(
    urdf_path: PathLike,
    *,
    package_dirs: Iterable[PathLike] = (),
) -> dict[str, object]:
    """Load and merge every URDF collision geometry in its link frame.

    Mesh, box, cylinder, and sphere elements are supported. Returned values are
    ``trimesh.Trimesh`` objects, but trimesh remains an optional dependency.
    """
    import trimesh

    urdf = Path(urdf_path).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf}")
    roots = tuple(Path(path).resolve() for path in package_dirs)
    root = ET.parse(urdf).getroot()
    if root.tag != "robot":
        raise ValueError("URDF root element must be <robot>")
    result = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            raise ValueError("URDF link is missing a name")
        geometries = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"link {link_name} has invalid collision geometry")
            shape = geometry[0]
            if shape.tag == "mesh":
                mesh_path = _resolve_mesh_uri(
                    shape.get("filename", ""), urdf.parent, roots
                )
                mesh = _as_mesh(trimesh.load(mesh_path, force="scene"), trimesh)
                scale = _vector(shape.get("scale"), 3, [1, 1, 1])
                if np.any(scale <= 0.0):
                    raise ValueError(f"link {link_name} mesh scale must be positive")
                mesh.apply_scale(scale)
            elif shape.tag == "box":
                size = _vector(shape.get("size"), 3, None)
                if np.any(size <= 0.0):
                    raise ValueError(f"link {link_name} box size must be positive")
                mesh = trimesh.creation.box(extents=size)
            elif shape.tag == "cylinder":
                radius = float(shape.get("radius", "nan"))
                length = float(shape.get("length", "nan"))
                if not np.isfinite(radius + length) or radius <= 0 or length <= 0:
                    raise ValueError(f"link {link_name} cylinder dimensions must be positive")
                mesh = trimesh.creation.cylinder(radius=radius, height=length)
            elif shape.tag == "sphere":
                radius = float(shape.get("radius", "nan"))
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError(f"link {link_name} sphere radius must be positive")
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            else:
                raise ValueError(
                    f"link {link_name} uses unsupported collision geometry {shape.tag!r}"
                )
            mesh.apply_transform(_origin_transform(collision.find("origin")))
            geometries.append(mesh)
        if geometries:
            result[link_name] = trimesh.util.concatenate(geometries)
    if not result:
        raise ValueError("URDF contains no supported collision geometry")
    return result


def fit_urdf_collision_spheres(
    urdf_path: PathLike,
    options: Optional[SphereFitOptions] = None,
    *,
    links: Optional[Iterable[str]] = None,
    package_dirs: Iterable[PathLike] = (),
    pitch: Optional[float] = None,
    surface_samples: int = 5000,
    random_seed: int = 0,
) -> dict[str, SphereFitResult]:
    """Fit selected URDF links and return one result per link."""
    meshes = load_urdf_collision_meshes(urdf_path, package_dirs=package_dirs)
    selected = tuple(links) if links is not None else tuple(meshes)
    if not selected:
        raise ValueError("links must not be empty")
    unknown = sorted(set(selected) - set(meshes))
    if unknown:
        raise ValueError("links have no collision geometry: " + ", ".join(unknown))
    return {
        link: fit_trimesh(
            meshes[link],
            options,
            pitch=pitch,
            surface_samples=surface_samples,
            random_seed=random_seed,
        )
        for link in selected
    }
