"""Shared, renderer-only helpers for repository Viser examples."""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import trimesh


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
            self._panel.content = self._format(
                fps, compute_time, average, maximum
            )

    def _format(
        self, fps: float, latest: float, average: float, maximum: float
    ) -> str:
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
            name, points, colors,
            thickness=line_width, thickness_units="screen",
        )
    except TypeError as error:
        if "thickness" not in str(error):
            raise
        return scene.add_line_segments(
            name, points, colors, line_width=line_width
        )
