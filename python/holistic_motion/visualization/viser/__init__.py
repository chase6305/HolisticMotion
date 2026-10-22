"""Reusable Viser scene primitives."""

from ...runtime import runtime_info

from .scene import (
    ViserPerformanceMonitor,
    add_line_segments,
    configure_scene,
    configure_scene_from_bounds,
    pose_components,
    visual_mesh,
)

__all__ = [
    "ViserPerformanceMonitor",
    "add_line_segments",
    "configure_scene",
    "configure_scene_from_bounds",
    "pose_components",
    "visual_mesh",
]

# Robot helpers consume enums from the compiled extension. Renderer-only
# consumers such as TOPPRA plots remain available in pure-Python mode.
if not runtime_info().pure_python:
    from .robot import chain_joint_names, tree_topology, tree_transforms

    __all__ += ["chain_joint_names", "tree_topology", "tree_transforms"]
