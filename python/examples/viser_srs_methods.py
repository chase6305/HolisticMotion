#!/usr/bin/env python3
"""Compare SRS inverse-kinematics strategies with an end-effector gizmo."""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import trimesh
import viser

from _bootstrap import import_holistic_motion
from _viser_utils import (
    ViserPerformanceMonitor,
    add_line_segments,
    pose_components,
    visual_mesh,
)
from viser_dual_arm_line import _tree_topology, _tree_transforms
from viser_robot import DEFAULT_ASSET_DIR


hm = import_holistic_motion()

METHODS = {
    "Closed-form analytic": None,
    "Seeded numerical": hm.SRSSolveMethod.SEEDED_NUMERICAL,
    "Fixed configuration": hm.SRSSolveMethod.CONFIGURATION,
    "All configurations": hm.SRSSolveMethod.ALL_CONFIGURATIONS,
    "Nearest redundancy": hm.SRSSolveMethod.NEAREST_REDUNDANCY,
}
DEFAULT_SEED = np.array([0.15, -0.35, 0.25, -0.7, 0.2, 0.3, -0.15])


def _control_transform(control) -> np.ndarray:
    transform = trimesh.transformations.quaternion_matrix(control.wxyz)
    transform[:3, 3] = np.asarray(control.position)
    return transform


def _pose_errors(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    position = np.linalg.norm(actual[:3, 3] - target[:3, 3])
    relative = actual[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return position, float(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf", type=Path, default=DEFAULT_ASSET_DIR / "left_arm.urdf"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gizmo-scale", type=float, default=0.22)
    parser.add_argument(
        "--trail-length", type=float, default=0.15,
        help="maximum visible target trail length in metres",
    )
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.gizmo_scale) or args.gizmo_scale <= 0.0:
        parser.error("--gizmo-scale must be finite and positive")
    if not np.isfinite(args.trail_length) or args.trail_length < 0.0:
        parser.error("--trail-length must be finite and non-negative")

    urdf_path = args.urdf.resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF does not exist: {urdf_path}")
    robot = hm.Robot(str(urdf_path))
    solver = robot.kinematics
    if not isinstance(solver, hm.SRSKinematics):
        parser.error("the URDF must contain a serial seven-revolute chain")
    geometry = solver.analyze_geometry()
    if not geometry.closed_form_compatible:
        parser.error("the 7R chain does not satisfy ideal SRS geometry")

    lower, upper = solver.joint_limits
    seed = np.clip(DEFAULT_SEED.copy(), lower + 1e-6, upper - 1e-6)
    target = solver.forward(seed)
    if args.validate_only:
        counts = {}
        for label, method in METHODS.items():
            try:
                if method is None:
                    configuration = solver.configuration(seed)
                    solutions = [
                        solver.analytic_solution(target, configuration, seed)
                    ]
                else:
                    solutions = solver.solve(target, seed, method)
            except ValueError:
                counts[label] = "unavailable"
                continue
            counts[label] = len(solutions)
            for solution in solutions:
                position, angle = _pose_errors(solver.forward(solution), target)
                if position > 1e-3 or angle > 1e-3:
                    raise RuntimeError(f"{label} returned an invalid solution")
        summary = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(
            f"validated {robot.name}: SRS upper/forearm="
            f"{geometry.upper_arm_length:.4f}/{geometry.forearm_length:.4f} m; "
            f"{summary}"
        )
        return

    robot.load_visuals()
    server = viser.ViserServer(port=args.port)
    performance = ViserPerformanceMonitor(server.gui)
    lock = threading.RLock()
    joints = seed.copy()
    candidates: list[np.ndarray] = []
    candidate_index = 0
    active_configuration = solver.configuration(joints)
    topology = _tree_topology(robot)
    joint_names = [joint.name for joint in robot.actuated_joints]

    def robot_transforms() -> dict[str, np.ndarray]:
        return _tree_transforms(
            robot, dict(zip(joint_names, joints)), topology
        )

    transforms = robot_transforms()
    handles: dict[str, list[tuple[object, np.ndarray]]] = {}
    for link in robot.links:
        link_handles = []
        for index, visual in enumerate(link.visuals):
            mesh = visual_mesh(hm, visual, urdf_path.parent)
            origin = np.asarray(visual.origin)
            wxyz, position = pose_components(transforms[link.name] @ origin)
            handle = server.scene.add_mesh_trimesh(
                f"/robot/{link.name}/visual_{index}", mesh,
                scale=tuple(visual.scale), wxyz=wxyz, position=position,
            )
            link_handles.append((handle, origin))
        handles[link.name] = link_handles

    initial_pose = solver.forward(joints)
    wxyz, position = pose_components(initial_pose)
    control = server.scene.add_transform_controls(
        "/target/end_effector", scale=args.gizmo_scale, line_width=4.0,
        wxyz=wxyz, position=position,
        disable_rotations=args.position_only,
    )
    target_frame = server.scene.add_frame(
        "/target/frame", axes_length=0.11, axes_radius=0.006,
        origin_radius=0.012, origin_color=(255, 155, 40),
        wxyz=wxyz, position=position,
    )
    target_marker = server.scene.add_icosphere(
        "/target/marker", radius=0.018, color=(255, 155, 40),
        material="toon5", position=position,
    )
    actual_frame = server.scene.add_frame(
        "/actual/frame", axes_length=0.085, axes_radius=0.004,
        origin_radius=0.009, origin_color=(40, 190, 255),
        wxyz=wxyz, position=position,
    )
    actual_marker = server.scene.add_icosphere(
        "/actual/marker", radius=0.013, color=(40, 190, 255),
        material="toon5", position=position,
    )
    error_line = add_line_segments(
        server.scene,
        "/guides/target_error", np.array([[position, position]]),
        (255, 220, 40), 3.0,
    )
    trail_points = deque([np.asarray(position).copy()], maxlen=32)
    trail = add_line_segments(
        server.scene,
        "/guides/target_trail", np.empty((0, 2, 3)),
        (255, 155, 40), 2.0,
    )
    target_label = server.scene.add_label(
        "/labels/target", "TARGET", position=np.asarray(position) + [0, 0, 0.04],
        anchor="bottom-center", font_screen_scale=0.8,
    )
    actual_label = server.scene.add_label(
        "/labels/actual", "ACTUAL TCP", position=np.asarray(position) + [0, 0, 0.04],
        anchor="top-center", font_screen_scale=0.75,
    )
    method = server.gui.add_dropdown(
        "SRS solve method", tuple(METHODS), initial_value="All configurations"
    )
    solve_button = server.gui.add_button("Solve target")
    next_button = server.gui.add_button("Next configuration")
    reset_button = server.gui.add_button("Reset")
    status = server.gui.add_markdown(
        "### SRS Solver\n"
        "🟠 Target · 🔵 Actual TCP · yellow line = residual\n\n"
        "Drag the gizmo to begin solving. Analytical mode stays closed-form "
        "during the drag."
    )
    pending_event = threading.Event()
    pending_lock = threading.Lock()
    pending_target: np.ndarray | None = None
    pending_final = False
    last_dashboard_publish = 0.0

    def update_scene(started: float | None = None) -> None:
        frame_started = started if started is not None else time.perf_counter()
        for link_name, transform in robot_transforms().items():
            for handle, origin in handles[link_name]:
                handle.wxyz, handle.position = pose_components(
                    transform @ origin
                )
        actual_pose = solver.forward(joints)
        actual_wxyz, actual_position = pose_components(actual_pose)
        actual_frame.wxyz, actual_frame.position = actual_wxyz, actual_position
        actual_marker.position = actual_position
        actual_label.position = actual_position + np.array([0.0, 0.0, 0.04])
        target_position = np.asarray(control.position)
        error_line.points = np.array([[actual_position, target_position]])
        performance.record(frame_started)

    def update_target_guides(add_trail_point: bool = True) -> None:
        target_pose = _control_transform(control)
        target_wxyz, target_position = pose_components(target_pose)
        target_frame.wxyz, target_frame.position = target_wxyz, target_position
        target_marker.position = target_position
        target_label.position = target_position + np.array([0.0, 0.0, 0.04])
        if add_trail_point and (
            not trail_points or
            np.linalg.norm(target_position - trail_points[-1]) > 0.003
        ):
            trail_points.append(target_position.copy())
            visible_length = 0.0
            for index in range(len(trail_points) - 1, 0, -1):
                visible_length += np.linalg.norm(
                    trail_points[index] - trail_points[index - 1]
                )
                if visible_length > args.trail_length:
                    for _ in range(index):
                        trail_points.popleft()
                    break
        if len(trail_points) >= 2:
            points = np.asarray(trail_points)
            trail.points = np.stack((points[:-1], points[1:]), axis=1)

    def follow_target(requested: np.ndarray) -> tuple[np.ndarray, float]:
        """Advance toward a target through small, locally solvable SE(3) steps."""
        start_pose = solver.forward(joints)
        start_quaternion = trimesh.transformations.quaternion_from_matrix(
            start_pose
        )
        target_quaternion = trimesh.transformations.quaternion_from_matrix(
            requested
        )
        solution = joints.copy()
        reached = 0.0
        for fraction in np.linspace(1.0 / 16.0, 1.0, 16):
            intermediate = np.eye(4)
            quaternion = trimesh.transformations.quaternion_slerp(
                start_quaternion, target_quaternion, fraction
            )
            intermediate[:3, :3] = (
                trimesh.transformations.quaternion_matrix(quaternion)[:3, :3]
            )
            intermediate[:3, 3] = (
                start_pose[:3, 3]
                + fraction * (requested[:3, 3] - start_pose[:3, 3])
            )
            try:
                local = solver.solve(
                    intermediate, solution,
                    hm.SRSSolveMethod.SEEDED_NUMERICAL,
                )
            except ValueError:
                break
            solution = np.asarray(local[0])
            reached = float(fraction)
        return solution, reached

    def solver_dashboard(
        title: str,
        requested: np.ndarray,
        position_error: float,
        angle_error: float,
        elapsed_ms: float | None,
        reached: float,
    ) -> str:
        actual = solver.forward(joints)
        configuration = solver.configuration(joints)
        singular_values = np.linalg.svd(
            solver.jacobian(joints), compute_uv=False
        )
        target_xyz = requested[:3, 3]
        actual_xyz = actual[:3, 3]
        reach_text = "Exact" if reached >= 1.0 else f"{reached * 100.0:.0f}%"
        solve_time_text = (
            "n/a" if elapsed_ms is None else f"{elapsed_ms:.3f} ms"
        )
        return (
            "### SRS Solver\n"
            "🟠 Target · 🔵 Actual TCP · yellow line = residual\n\n"
            "| Metric | Value |\n"
            "|:--|--:|\n"
            f"| Method | **{title}** |\n"
            f"| Solution | **{candidate_index + 1} / {len(candidates)}** |\n"
            f"| S / E / W | **{configuration.shoulder:+d} / "
            f"{configuration.elbow:+d} / {configuration.wrist:+d}** |\n"
            f"| Redundancy | **{np.degrees(configuration.redundancy):.2f}°** |\n"
            f"| Position error | **{position_error * 1000.0:.3f} mm** |\n"
            f"| Orientation error | **{np.degrees(angle_error):.3f}°** |\n"
            f"| IK core time | **{solve_time_text}** |\n"
            f"| Reach | **{reach_text}** |\n"
            f"| Jacobian σ min | **{singular_values[-1]:.3e}** |\n\n"
            f"Target XYZ: `{target_xyz[0]:+.3f}, {target_xyz[1]:+.3f}, "
            f"{target_xyz[2]:+.3f}` m  \n"
            f"Actual XYZ: `{actual_xyz[0]:+.3f}, {actual_xyz[1]:+.3f}, "
            f"{actual_xyz[2]:+.3f}` m"
        )

    def solve_target(
        requested: np.ndarray | None = None,
        selected_method=None,
        preview: bool = False,
    ) -> None:
        nonlocal candidates, candidate_index, last_dashboard_publish
        nonlocal active_configuration
        with lock:
            started = time.perf_counter()
            if requested is None:
                requested = _control_transform(control)
            if args.position_only:
                requested[:3, :3] = solver.forward(joints)[:3, :3]
            reached = 1.0
            solve_method = (
                selected_method
                if selected_method is not None
                else METHODS[method.value]
            )
            ik_started = time.perf_counter()
            try:
                if solve_method is None:
                    candidates = [np.asarray(solver.analytic_solution(
                        requested, active_configuration, joints
                    ))]
                else:
                    candidates = [
                        np.asarray(value) for value in solver.solve(
                            requested, joints, solve_method
                        )
                    ]
            except ValueError:
                if solve_method is None:
                    candidates = []
                    update_scene(started)
                    now = time.monotonic()
                    if not preview or now - last_dashboard_publish >= 0.1:
                        status.content = (
                            "### SRS Solver\n"
                            "| Metric | Value |\n|:--|--:|\n"
                            "| Method | **Closed-form analytic** |\n"
                            "| Result | **No feasible strict closed-form solution** |\n\n"
                            "The target, branch, joint limits, or robot geometry "
                            "does not admit an exact analytical result. No numerical "
                            "fallback was used."
                        )
                        last_dashboard_publish = now
                    return
                closest, reached = follow_target(requested)
                if reached == 0.0:
                    candidates = []
                    status.content = (
                        "**No local IK step found.** The target remains "
                        "movable—drag it back toward the blue TCP marker."
                    )
                    update_scene(started)
                    return
                candidates = [closest]
            solve_elapsed = (time.perf_counter() - ik_started) * 1000.0
            candidate_index = 0
            joints[:] = candidates[0]
            active_configuration = solver.configuration(joints)
            update_scene(started)
            position_error, angle_error = _pose_errors(
                solver.forward(joints), requested
            )
            now = time.monotonic()
            if not preview or now - last_dashboard_publish >= 0.1:
                if preview:
                    title = (
                        "Closed-form analytic preview"
                        if solve_method is None else "Seeded preview"
                    )
                else:
                    title = method.value
                status.content = solver_dashboard(
                    title, requested, position_error, angle_error,
                    solve_elapsed, reached,
                )
                last_dashboard_publish = now

    @control.on_update
    def _on_target(event) -> None:
        nonlocal pending_target, pending_final
        # Keep this callback lightweight so Viser can accept unconstrained
        # six-axis drag events even when IK needs several continuation steps.
        update_target_guides()
        with pending_lock:
            pending_target = _control_transform(control)
            pending_final = getattr(event, "phase", None) == "end"
        pending_event.set()

    def _solver_worker() -> None:
        nonlocal pending_target, pending_final
        while True:
            pending_event.wait()
            # Coalesce a burst of mouse events and solve only the newest pose.
            time.sleep(1.0 / 60.0)
            with pending_lock:
                requested = pending_target
                final = pending_final
                pending_target = None
                pending_final = False
                pending_event.clear()
            if requested is None:
                continue
            if final:
                solve_target(requested)
            elif METHODS[method.value] is None:
                # Strict analytical mode remains analytical while dragging;
                # do not hide numerical preview latency in its timing/feel.
                solve_target(requested, preview=True)
            else:
                solve_target(
                    requested,
                    hm.SRSSolveMethod.SEEDED_NUMERICAL,
                    preview=True,
                )

    threading.Thread(target=_solver_worker, daemon=True).start()

    @solve_button.on_click
    def _on_solve(_event) -> None:
        solve_target()

    @next_button.on_click
    def _on_next(_event) -> None:
        nonlocal candidate_index, active_configuration
        with lock:
            if not candidates:
                solve_target()
                return
            candidate_index = (candidate_index + 1) % len(candidates)
            joints[:] = candidates[candidate_index]
            active_configuration = solver.configuration(joints)
            update_scene()
            requested = _control_transform(control)
            position_error, angle_error = _pose_errors(
                solver.forward(joints), requested
            )
            status.content = solver_dashboard(
                method.value, requested, position_error, angle_error,
                None, 1.0,
            )

    @reset_button.on_click
    def _on_reset(_event) -> None:
        nonlocal candidates, candidate_index, active_configuration
        with lock:
            joints[:] = seed
            active_configuration = solver.configuration(joints)
            candidates = []
            candidate_index = 0
            update_scene()
            control.wxyz, control.position = pose_components(
                solver.forward(joints)
            )
            trail_points.clear()
            trail_points.append(np.asarray(control.position).copy())
            trail.points = np.empty((0, 2, 3))
            update_target_guides(add_trail_point=False)
            status.content = "Reset to the reference SRS configuration."

    update_scene()
    print(f"Viser SRS methods demo: http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
