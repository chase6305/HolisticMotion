#!/usr/bin/env python3
"""Validate OPW/UR closed-form IK interactively on an external URDF."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR))
from _bootstrap import import_holistic_motion  # noqa: E402
from _viser_utils import (  # noqa: E402
    ViserPerformanceMonitor, add_line_segments, pose_components, visual_mesh,
)
from viser_robot import _link_transforms, _tree_topology  # noqa: E402

hm = import_holistic_motion()

DEFAULT_URDF = Path(
    "/home/ubuntu/workspace/chase/AssetsModel/RobotModel/"
    "ABB/IRB1200_5_90/IRB1200_5_90.urdf"
)

UR_GEOMETRY = {
    "UR3": (0.1519, -0.24365, -0.21325, 0.11235, 0.08535, 0.0819),
    "UR3e": (0.152, -0.244, -0.213, 0.131, 0.085, 0.092),
    "UR5": (0.089159, -0.425, -0.39225, 0.10915, 0.09465, 0.0823),
    "UR5e": (0.163, -0.425, -0.392, 0.134, 0.100, 0.100),
    "UR10": (0.1273, -0.612, -0.5723, 0.163941, 0.1157, 0.0922),
    "UR10e": (0.181, -0.613, -0.571, 0.174, 0.120, 0.117),
}


def make_solver(robot, urdf: Path):
    lower, upper = robot.kinematics.joint_limits
    model = urdf.stem
    if model in UR_GEOMETRY:
        params = hm.URParameters()
        (params.d1, params.a2, params.a3, params.d4, params.d5,
         params.d6) = UR_GEOMETRY[model]
        params.rotation_directions = [1] * 6
        params.offsets = [0.0] * 6
        params.base_transform = np.diag([-1.0, -1.0, 1.0, 1.0])
        params.tool_transform = np.array([
            [0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ])
        return hm.URKinematics(params, lower, upper), "UR"
    if model == "IRB1200_5_90":
        params = hm.OPWParameters()
        (params.a1, params.a2, params.b, params.c1, params.c2,
         params.c3, params.c4) = (0.0, -0.042, 0.0, 0.0,
                                      0.448, 0.451, 0.082)
        params.offsets = [0.0, 0.0, -np.pi / 2.0, 0.0, 0.0, 0.0]
        params.rotation_directions = [1] * 6
        base = np.eye(4)
        base[2, 3] = 0.3991
        params.base_transform = base
        return hm.OPWKinematics(params, lower, upper), "OPW"
    raise ValueError(
        f"no verified analytic preset for {model}; supported: "
        f"{', '.join((*UR_GEOMETRY, 'IRB1200_5_90'))}"
    )


def pose_from_control(control) -> np.ndarray:
    pose = trimesh.transformations.quaternion_matrix(control.wxyz)
    pose[:3, 3] = control.position
    return pose


def branch_label(joints: np.ndarray) -> str:
    signs = tuple("+" if joints[index] >= 0.0 else "-" for index in (0, 2, 4))
    return f"S/E/W={signs[0]}/{signs[1]}/{signs[2]}"


def rotation_error_degrees(actual: np.ndarray, desired: np.ndarray) -> float:
    relative = actual[:3, :3].T @ desired[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def periodic_joint_distance(first: np.ndarray, second: np.ndarray) -> float:
    delta = first - second
    return float(np.linalg.norm(np.arctan2(np.sin(delta), np.cos(delta))))


def validate_solver(robot, solver, lower, upper, samples: int) -> dict:
    """Cross-check analytical FK/IK against the URDF tree over the workspace."""
    if samples < 1:
        raise ValueError("validation sample count must be positive")
    rng = np.random.default_rng(6305)
    sample_lower = np.maximum(lower, -1.25)
    sample_upper = np.minimum(upper, 1.25)
    if np.any(sample_lower > sample_upper):
        raise ValueError("joint limits do not contain a finite validation range")
    fk_errors, position_errors, rotation_errors = [], [], []
    solve_times, branch_counts = [], []
    for joints in rng.uniform(sample_lower, sample_upper, size=(samples, 6)):
        analytic = solver.forward(joints)
        numerical = robot.kinematics.forward(joints)
        fk_errors.append(np.max(np.abs(analytic - numerical)))
        started = time.perf_counter()
        solutions = solver.solve_all(analytic, joints)
        solve_times.append((time.perf_counter() - started) * 1e3)
        if not solutions:
            raise RuntimeError("FK-generated reachable pose returned no IK")
        branch_counts.append(len(solutions))
        recovered = solver.forward(solutions[0])
        position_errors.append(np.linalg.norm(
            recovered[:3, 3] - analytic[:3, 3]
        ))
        rotation_errors.append(rotation_error_degrees(recovered, analytic))
    return {
        "samples": samples,
        "fk_max": float(np.max(fk_errors)),
        "position_max": float(np.max(position_errors)),
        "rotation_max": float(np.max(rotation_errors)),
        "solve_median_ms": float(np.median(solve_times)),
        "solve_p95_ms": float(np.percentile(solve_times, 95)),
        "branches_min": int(np.min(branch_counts)),
        "branches_max": int(np.max(branch_counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-samples", type=int, default=64,
                        help="deterministic URDF/analytic cross-check count")
    args = parser.parse_args()

    urdf = args.urdf.resolve()
    robot = hm.Robot(str(urdf))
    if robot.dof != 6:
        raise SystemExit("OPW/UR demo requires a six-axis URDF")
    solver, family = make_solver(robot, urdf)
    lower, upper = robot.kinematics.joint_limits
    try:
        validation = validate_solver(
            robot, solver, lower, upper, args.validation_samples
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"analytic validation failed: {error}") from error
    seed = np.clip(np.array([0.25, -0.65, 0.75, -0.4, 0.6, -0.2]),
                   lower, upper)
    analytic_pose = solver.forward(seed)
    numerical_pose = robot.kinematics.forward(seed)
    fk_error = float(np.max(np.abs(analytic_pose - numerical_pose)))
    start = time.perf_counter()
    solutions = solver.solve_all(analytic_pose, seed)
    solve_ms = (time.perf_counter() - start) * 1e3
    if fk_error > 1e-7 or not solutions:
        raise RuntimeError(
            f"{family} preset validation failed: FK={fk_error:.3e}, "
            f"solutions={len(solutions)}"
        )
    if args.validate_only:
        print(
            f"{family} {robot.name}: samples={validation['samples']}, "
            f"FK max={validation['fk_max']:.3e}, "
            f"IK max={validation['position_max'] * 1e3:.3e}mm/"
            f"{validation['rotation_max']:.3e}deg, branches="
            f"{validation['branches_min']}-{validation['branches_max']}, "
            f"solve median/p95={validation['solve_median_ms']:.3f}/"
            f"{validation['solve_p95_ms']:.3f} ms"
        )
        return

    robot.load_visuals()
    topology = _tree_topology(robot)
    joints = seed.copy()
    transforms = _link_transforms(robot, joints, topology)
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid(
        "/world/grid", width=3.0, height=3.0, cell_size=0.1,
        section_size=0.5, plane_opacity=0.08, shadow_opacity=0.15,
    )
    server.scene.add_frame(
        "/world/base", axes_length=0.16, axes_radius=0.006,
    )
    mesh_handles = []
    for link, transform in zip(robot.links, transforms):
        handles = []
        for index, visual in enumerate(link.visuals):
            mesh = visual_mesh(hm, visual, urdf.parent)
            wxyz, position = pose_components(transform @ visual.origin)
            handle = server.scene.add_mesh_trimesh(
                f"/robot/{link.name}/{index}", mesh,
                scale=tuple(visual.scale), wxyz=wxyz, position=position,
            )
            handles.append((handle, visual.origin))
        mesh_handles.append(handles)

    wxyz, position = pose_components(analytic_pose)
    target = server.scene.add_transform_controls(
        "/target", scale=0.18, wxyz=wxyz, position=position,
    )
    target_marker = server.scene.add_icosphere(
        "/markers/target", radius=0.014, color=(255, 155, 45),
        material="toon5", position=position,
    )
    actual_frame = server.scene.add_frame(
        "/frames/actual_tcp", axes_length=0.13, axes_radius=0.005,
        origin_radius=0.012, origin_color=(45, 175, 255),
        wxyz=wxyz, position=position,
    )
    actual_marker = server.scene.add_icosphere(
        "/markers/actual_tcp", radius=0.012, color=(45, 175, 255),
        material="toon5", position=position,
    )
    error_line = add_line_segments(
        server.scene, "/diagnostics/position_error",
        np.array([[position, position]]), (235, 70, 70), line_width=3.0,
    )
    solution_index = server.gui.add_slider(
        "Solution", min=1, max=max(1, len(solutions)), step=1,
        initial_value=1,
    )
    status = server.gui.add_markdown("")
    performance = ViserPerformanceMonitor(server.gui)
    state = {
        "solutions": solutions, "solve_ms": solve_ms,
        "position_error": 0.0, "rotation_error": 0.0,
        "target": analytic_pose.copy(), "busy": False,
        "validation": validation,
    }

    def update_scene() -> None:
        started = time.perf_counter()
        index = min(int(solution_index.value) - 1,
                    len(state["solutions"]) - 1)
        joints[:] = state["solutions"][index]
        for transform, handles in zip(
                _link_transforms(robot, joints, topology), mesh_handles):
            for handle, origin in handles:
                handle.wxyz, handle.position = pose_components(
                    transform @ origin
                )
        actual = solver.forward(joints)
        actual_wxyz, actual_position = pose_components(actual)
        desired_position = state["target"][:3, 3]
        actual_frame.wxyz, actual_frame.position = actual_wxyz, actual_position
        actual_marker.position = actual_position
        target_marker.position = desired_position
        error_line.points = np.array([[actual_position, desired_position]])
        state["position_error"] = np.linalg.norm(
            actual_position - desired_position
        )
        state["rotation_error"] = rotation_error_degrees(
            actual, state["target"]
        )
        branch_rows = []
        for branch_index, solution in enumerate(state["solutions"]):
            marker = "→" if branch_index == index else ""
            branch_rows.append(
                f"| {marker} {branch_index + 1} | "
                f"{branch_label(solution)} | "
                f"{periodic_joint_distance(solution, joints):.3f} |"
            )
        joint_header = " / ".join(
            f"J{joint + 1}={np.rad2deg(value):+.1f}°"
            for joint, value in enumerate(joints)
        )
        status.content = (
            f"### {family} analytic IK — solved\n"
            "🟠 Target TCP · 🔵 Actual TCP · 🔴 Position residual\n\n"
            "| Metric | Value |\n|:--|--:|\n"
            f"| Model | **{robot.name}** |\n"
            f"| Active branch | **{index + 1}/{len(state['solutions'])} · "
            f"{branch_label(joints)}** |\n"
            f"| Residual | **{state['position_error'] * 1e3:.3f} mm / "
            f"{state['rotation_error']:.4f}°** |\n"
            f"| Solve | **{state['solve_ms']:.3f} ms** |\n\n"
            f"Startup validation: **{validation['samples']} poses**, FK max "
            f"**{validation['fk_max']:.2e}**, solve p95 "
            f"**{validation['solve_p95_ms']:.3f} ms**\n\n"
            f"`{joint_header}`\n\n"
            "| # | Configuration | Δq from active [rad] |\n"
            "|--:|:--:|--:|\n" + "\n".join(branch_rows)
        )
        performance.record(started)

    def solve_target() -> None:
        if state["busy"]:
            return
        state["busy"] = True
        try:
            desired = pose_from_control(target)
            state["target"] = desired
            target_marker.position = desired[:3, 3]
            started = time.perf_counter()
            found = solver.solve_all(desired, joints.copy())
            state["solve_ms"] = (time.perf_counter() - started) * 1e3
            if not found:
                status.content = (
                    f"### {family} analytic IK — unreachable\n"
                    "🟠 The target remains visible. Move the gizmo back into "
                    "the workspace; the robot keeps its last valid pose.\n\n"
                    f"Solve attempt: **{state['solve_ms']:.3f} ms**"
                )
                return
            state["solutions"] = found
            solution_index.max = len(found)
            solution_index.value = 1
            update_scene()
        finally:
            state["busy"] = False

    @target.on_update
    def _on_target(_event):
        solve_target()

    @solution_index.on_update
    def _on_solution(_event):
        update_scene()

    update_scene()
    print(f"Viewing {family} solver for {urdf}; press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
