"""Shared, renderer-only helpers for repository Viser examples."""

from __future__ import annotations

import threading
import time
from collections import deque
from math import floor, log10
from pathlib import Path

import numpy as np
import trimesh


def _positive_scene_extent(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nice_grid_step(extent: float) -> float:
    """Choose roughly 20--40 cells using a stable 1-2-5 decimal step."""

    # Division underflows for the smallest positive floats. Clamping to the
    # smallest normal value keeps log10() defined and the returned step usable.
    target = max(extent / 40.0, np.finfo(float).tiny)
    exponent = floor(log10(target))
    scale = 10.0**exponent
    fraction = target / scale
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        if fraction <= multiplier:
            return multiplier * scale
    raise AssertionError("decimal grid fraction must be below 10")


def _enable_default_lights(scene) -> None:
    configure = getattr(scene, "configure_default_lights", None)
    if callable(configure):
        try:
            configure(enabled=True, cast_shadow=True)
        except TypeError as error:
            if "unexpected keyword argument" not in str(error):
                raise
            try:
                configure(enabled=True)
            except TypeError as fallback_error:
                if "unexpected keyword argument" not in str(fallback_error):
                    raise
                configure(True)
        return
    enable = getattr(scene, "enable_default_lights", None)
    if callable(enable):
        enable(True)


def configure_scene(
    scene,
    *,
    ground_width: float = 4.0,
    ground_height: float | None = None,
    ground_z: float = 0.0,
    ground_center_xy=(0.0, 0.0),
    grid_cell_size: float | None = None,
):
    """Apply the shared studio environment used by Viser examples.

    Newer Viser releases receive an HDRI background plus explicit fill and key
    lights. Older releases retain their default lights and receive a compatible
    grid. The returned handle is the ground grid.
    """

    width = _positive_scene_extent(ground_width, "ground_width")
    height = (
        width
        if ground_height is None
        else _positive_scene_extent(ground_height, "ground_height")
    )
    try:
        z = float(ground_z)
    except (TypeError, ValueError) as error:
        raise TypeError("ground_z must be numeric") from error
    if not np.isfinite(z):
        raise ValueError("ground_z must be finite")
    center = np.asarray(ground_center_xy, dtype=float).reshape(-1)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("ground_center_xy must contain two finite values")
    cell_size = (
        _nice_grid_step(max(width, height))
        if grid_cell_size is None
        else _positive_scene_extent(grid_cell_size, "grid_cell_size")
    )
    section_size = 5.0 * cell_size
    if not np.isfinite(section_size):
        raise ValueError("grid section size must be finite")

    set_up_direction = getattr(scene, "set_up_direction", None)
    if callable(set_up_direction):
        set_up_direction("+z")

    _enable_default_lights(scene)
    environment_map = getattr(scene, "configure_environment_map", None)
    if callable(environment_map):
        try:
            environment_map(
                hdri="studio",
                background=True,
                background_blurriness=0.72,
                background_intensity=0.82,
                environment_intensity=1.15,
            )
        except TypeError as error:
            if "unexpected keyword argument" not in str(error):
                raise
            try:
                environment_map(hdri="studio", background=True)
            except TypeError as fallback_error:
                if "unexpected keyword argument" not in str(fallback_error):
                    raise
                environment_map(hdri="studio")
        add_fill = getattr(scene, "add_light_hemisphere", None)
        if callable(add_fill):
            try:
                add_fill(
                    "/environment/lights/fill",
                    sky_color=(190, 210, 235),
                    ground_color=(75, 70, 65),
                    intensity=0.75,
                )
            except TypeError as error:
                if "unexpected keyword argument" not in str(error):
                    raise
        add_key = getattr(scene, "add_light_directional", None)
        if callable(add_key):
            try:
                add_key(
                    "/environment/lights/key",
                    color=(255, 238, 214),
                    intensity=2.5,
                    cast_shadow=True,
                    wxyz=(0.9238795, 0.2705981, -0.2705981, 0.0),
                )
            except TypeError as error:
                if "unexpected keyword argument" not in str(error):
                    raise

    grid_options = {
        "width": width,
        "height": height,
        "cell_size": cell_size,
        "cell_color": (82, 92, 108),
        "cell_thickness": 0.7,
        "section_color": (44, 54, 70),
        "section_thickness": 1.15,
        "section_size": section_size,
        "shadow_opacity": 0.32,
        "plane_color": (154, 164, 180),
        "plane_opacity": 0.68,
        "position": (float(center[0]), float(center[1]), z),
    }
    try:
        return scene.add_grid("/environment/ground", **grid_options)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        try:
            return scene.add_grid(
                "/environment/ground",
                width=width,
                height=height,
                position=grid_options["position"],
            )
        except TypeError as fallback_error:
            if "unexpected keyword argument" not in str(fallback_error):
                raise
            return scene.add_grid("/environment/ground", width=width, height=height)


def configure_scene_from_bounds(
    scene,
    bounds,
    *,
    minimum_ground_size: float = 1.0,
    margin: float = 0.15,
):
    """Configure a scene with a floor centered just below finite 3-D bounds."""

    values = np.asarray(bounds, dtype=float)
    if values.shape != (2, 3) or not np.isfinite(values).all():
        raise ValueError("bounds must be a finite array with shape (2, 3)")
    with np.errstate(over="ignore", invalid="ignore"):
        spans = values[1] - values[0]
    if not np.isfinite(spans).all():
        raise ValueError("bounds span must be finite")
    if np.any(spans < 0.0):
        raise ValueError("bounds maximum must not be below its minimum")
    minimum = _positive_scene_extent(minimum_ground_size, "minimum_ground_size")
    try:
        padding_ratio = float(margin)
    except (TypeError, ValueError) as error:
        raise TypeError("margin must be numeric") from error
    if not np.isfinite(padding_ratio) or padding_ratio < 0.0:
        raise ValueError("margin must be finite and non-negative")
    scale = max(minimum, float(np.max(spans)))
    with np.errstate(over="ignore", invalid="ignore"):
        padding = padding_ratio * scale
        center = values[0, :2] + 0.5 * spans[:2]
        clearance = max(0.005, 0.02 * scale)
        ground_width = max(minimum, float(spans[0]) + 2.0 * padding)
        ground_height = max(minimum, float(spans[1]) + 2.0 * padding)
        ground_z = float(values[0, 2] - clearance)
    if not np.isfinite(
        [padding, *center, clearance, ground_width, ground_height, ground_z]
    ).all():
        raise ValueError("bounds and margin must produce a finite ground plane")
    return configure_scene(
        scene,
        ground_width=ground_width,
        ground_height=ground_height,
        ground_z=ground_z,
        ground_center_xy=center,
    )


class ViserPerformanceMonitor:
    """Publish rolling scene-update performance in a Viser GUI panel."""

    def __init__(self, gui, target_fps: float = 60.0, window: int = 120):
        if target_fps <= 0.0 or window < 2:
            raise ValueError("target_fps must be positive and window >= 2")
        self._target_fps = float(target_fps)
        self._frame_times = deque(maxlen=window)
        self._compute_times = deque(maxlen=window)
        self._last_publish = 0.0
        self._lock = threading.Lock()
        self._panel = gui.add_markdown(self._format(0.0, 0.0, 0.0, 0.0))

    def record(self, frame_started: float) -> None:
        """Record one completed update; frame_started uses perf_counter()."""
        completed = time.perf_counter()
        compute_time = max(0.0, completed - frame_started)
        with self._lock:
            if self._frame_times:
                interval = completed - self._frame_times[-1]
                if interval > 1.0:
                    self._frame_times.clear()
                    self._compute_times.clear()
                elif interval > 0.0:
                    self._compute_times.append(compute_time)
            self._frame_times.append(completed)
            if completed - self._last_publish < 0.25:
                return
            self._last_publish = completed
            intervals = np.diff(self._frame_times)
            fps = 1.0 / np.mean(intervals) if len(intervals) else 0.0
            average = np.mean(self._compute_times) if self._compute_times else 0.0
            maximum = np.max(self._compute_times) if self._compute_times else 0.0
            self._panel.content = self._format(fps, compute_time, average, maximum)

    def _format(self, fps: float, latest: float, average: float, maximum: float) -> str:
        budget = average * self._target_fps * 100.0
        return (
            "### Performance\n"
            f"- Update FPS: **{fps:.1f}** / {self._target_fps:.0f}\n"
            f"- Compute: **{latest * 1000.0:.2f} ms** latest, "
            f"{average * 1000.0:.2f} ms avg, {maximum * 1000.0:.2f} ms max\n"
            f"- Frame budget used: **{budget:.1f}%**"
        )


def visual_mesh(hm, visual, asset_directory: Path) -> trimesh.Trimesh:
    if visual.type == hm.GeometryType.MESH:
        path = (asset_directory / visual.mesh_path).resolve()
        loaded = trimesh.load(path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            if not meshes:
                raise ValueError(f"mesh scene is empty: {path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = loaded
    elif visual.type == hm.GeometryType.BOX:
        mesh = trimesh.creation.box(extents=np.asarray(visual.size))
    elif visual.type == hm.GeometryType.CYLINDER:
        mesh = trimesh.creation.cylinder(
            radius=float(visual.radius), height=float(visual.length)
        )
    elif visual.type == hm.GeometryType.SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(visual.radius))
    else:
        raise ValueError(f"unsupported visual geometry: {visual.type}")
    if visual.has_color:
        rgba = np.clip(np.asarray(visual.color) * 255.0, 0, 255).astype(np.uint8)
        mesh.visual.face_colors = rgba
    return mesh


def pose_components(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternion = trimesh.transformations.quaternion_from_matrix(transform)
    return quaternion, transform[:3, 3]


def add_line_segments(
    scene, name: str, points: np.ndarray, colors, line_width: float = 2.0
):
    """Create screen-space lines across old and current Viser APIs."""
    try:
        return scene.add_line_segments(
            name,
            points,
            colors,
            thickness=line_width,
            thickness_units="screen",
        )
    except TypeError as error:
        if "thickness" not in str(error):
            raise
        return scene.add_line_segments(name, points, colors, line_width=line_width)
