"""Reusable robot-tree helpers for Viser applications."""

from __future__ import annotations

import numpy as np
import trimesh

from holistic_motion import JointType


def _joint_motion(joint, value: float) -> np.ndarray:
    result = np.eye(4)
    axis = np.asarray(joint.axis, dtype=float)
    if joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
        length = np.linalg.norm(axis)
        if length > 0.0:
            result = trimesh.transformations.rotation_matrix(
                value, axis / length
            )
    elif joint.joint_type == JointType.PRISMATIC:
        result[:3, 3] = axis * value
    return result


def tree_topology(robot) -> list:
    """Return joints in parent-before-child order."""
    ordered = []
    pending = [robot.root_link_name]
    while pending:
        parent_name = pending.pop()
        for joint_name in robot.get_link(parent_name).child_joints:
            joint = robot.get_joint(joint_name)
            ordered.append(joint)
            pending.append(joint.child_link)
    if len(ordered) != len(robot.joints):
        raise RuntimeError("URDF tree contains disconnected links")
    return ordered


def tree_transforms(robot, positions: dict[str, float], topology=None) -> dict:
    """Compute world transforms for the complete URDF tree."""
    transforms = {robot.root_link_name: np.eye(4)}
    for joint in topology or tree_topology(robot):
        value = positions.get(joint.name, 0.0)
        if joint.mimic_joint:
            value = (
                positions.get(joint.mimic_joint, 0.0)
                * joint.mimic_multiplier
                + joint.mimic_offset
            )
        transforms[joint.child_link] = (
            transforms[joint.parent_link]
            @ np.asarray(joint.origin)
            @ _joint_motion(joint, value)
        )
    return transforms


def chain_joint_names(robot, base: str, tip: str) -> list[str]:
    """Return non-fixed joint names on the path from base to tip."""
    names = []
    current = tip
    while current != base:
        link = robot.get_link(current)
        if link is None or not link.parent_joint:
            raise ValueError(f"{tip!r} is not below {base!r}")
        joint = robot.get_joint(link.parent_joint)
        if joint.joint_type != JointType.FIXED:
            names.append(joint.name)
        current = joint.parent_link
    names.reverse()
    return names
