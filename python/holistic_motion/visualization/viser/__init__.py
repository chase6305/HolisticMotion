"""Reusable Viser scene primitives."""

import os

from .scene import (
    ViserPerformanceMonitor,
    add_line_segments,
    pose_components,
    visual_mesh,
)

__all__ = [
    "ViserPerformanceMonitor",
    "add_line_segments",
    "pose_components",
    "visual_mesh",
]

# Robot helpers consume enums from the compiled extension. Renderer-only
# consumers such as TOPPRA plots remain available in pure-Python mode.
if os.environ.get("HOLISTICMOTION_PURE_PYTHON") != "1":
    from .robot import chain_joint_names, tree_topology, tree_transforms

    __all__ += ["chain_joint_names", "tree_topology", "tree_transforms"]
