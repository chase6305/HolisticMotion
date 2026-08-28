#!/usr/bin/env python3
"""Interactive Viser viewer for a serial robot loaded from an external URDF."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from _bootstrap import import_holistic_motion
from _viser_utils import (
    ViserPerformanceMonitor, add_line_segments, pose_components, visual_mesh,
)


hm = import_holistic_motion()


def _motion(joint, value: float) -> np.ndarray:
    transform = np.eye(4)
    axis = np.asarray(joint.axis, dtype=float)
    if joint.joint_type in (hm.JointType.REVOLUTE, hm.JointType.CONTINUOUS):
        norm = np.linalg.norm(axis)
        if norm > 0.0:
            transform = trimesh.transformations.rotation_matrix(
                value, axis / norm
            )
    elif joint.joint_type == hm.JointType.PRISMATIC:
        transform[:3, 3] = axis * value
    return transform


def _tree_topology(robot) -> list:
    ordered, pending = [], [robot.root_link_name]
    while pending:
        parent_name = pending.pop()
        for joint_name in robot.get_link(parent_name).child_joints:
            joint = robot.get_joint(joint_name)
            ordered.append(joint)
            pending.append(joint.child_link)
    if len(ordered) != len(robot.joints):
        raise RuntimeError("URDF contains disconnected links")
    return ordered


def _link_transforms(
    robot: hm.Robot, joints: np.ndarray, topology: list
) -> list[np.ndarray]:
    values = {
        joint.name: float(value)
        for joint, value in zip(robot.actuated_joints, joints)
    }
    transforms = {robot.root_link_name: np.eye(4)}
    for joint in topology:
        value = values.get(joint.name, 0.0)
        if joint.mimic_joint:
            value = (values.get(joint.mimic_joint, 0.0)
                     * joint.mimic_multiplier + joint.mimic_offset)
        transforms[joint.child_link] = (
            transforms[joint.parent_link] @ np.asarray(joint.origin)
            @ _motion(joint, value)
        )
    return [transforms[link.name] for link in robot.links]


def _limit_aware_posture(robot):
    """Return a neutral pose and signed travel toward the roomier limit side."""
    dof = robot.dof
    home = np.zeros(dof)
    signed_travel = np.empty(dof)
    for index, joint in enumerate(robot.actuated_joints):
        lower, upper = float(joint.limit.lower), float(joint.limit.upper)
        if np.isfinite(lower) and np.isfinite(upper):
            home[index] = np.clip(0.0, lower, upper)
            positive_room = max(0.0, upper - home[index])
            negative_room = max(0.0, home[index] - lower)
            direction = 1.0 if positive_room >= negative_room else -1.0
            signed_travel[index] = direction * min(
                0.45, 0.28 * max(positive_room, negative_room)
            )
        else:
            signed_travel[index] = 0.35
    return home, signed_travel


def create_demo_trajectory(robot):
    """Create a trajectory biased toward each joint's roomier limit side."""
    home, signed_travel = _limit_aware_posture(robot)
    dof = robot.dof
    phase = 0.43 * np.arange(dof)
    first = home + signed_travel * (0.55 + 0.30 * np.sin(phase + 0.7))
    second = home + signed_travel * (0.35 + 0.25 * np.sin(phase - 0.9))
    for index, joint in enumerate(robot.actuated_joints):
        lower, upper = float(joint.limit.lower), float(joint.limit.upper)
        if np.isfinite(lower):
            first[index], second[index] = max(first[index], lower), max(second[index], lower)
        if np.isfinite(upper):
            first[index], second[index] = min(first[index], upper), min(second[index], upper)
    waypoints = np.vstack((home, first, second, home))
    return hm.RnTrajectory(
        waypoints, np.full(dof, 0.8), np.full(dof, 1.5),
        np.full(dof, 4.0), blend_tolerance=0.01,
    )


def create_cartesian_line_trajectory(robot, distance: float = 0.35):
    """Sample a TCP line uniformly, solve IK in order, then time-parameterize."""
    home, signed_travel = _limit_aware_posture(robot)
    seed = home + 0.15 * signed_travel
    if robot.dof < 4:
        raise RuntimeError("Cartesian line demo requires at least four joints")
    joint4 = robot.actuated_joints[3]
    lower4, upper4 = float(joint4.limit.lower), float(joint4.limit.upper)
    elbow_candidates = [
        angle for angle in (-0.5 * np.pi, 0.5 * np.pi)
        if lower4 - 1e-9 <= angle <= upper4 + 1e-9
    ]
    if not elbow_candidates:
        raise RuntimeError("joint 4 cannot reach either +90 or -90 degrees")
    # Prefer the 90-degree elbow posture on the side with more remaining room.
    seed[3] = max(
        elbow_candidates,
        key=lambda angle: min(angle - lower4, upper4 - angle),
    )
    solver = robot.kinematics
    start = np.asarray(solver.forward(seed))
    candidates = []
    offsets = np.linspace(0.0, distance, 51)
    for axis in range(3):
        for direction in (-1.0, 1.0):
            solutions, targets = [], []
            previous = seed.copy()
            valid = True
            for offset in offsets:
                target = start.copy()
                target[axis, 3] += direction * offset
                try:
                    solution = np.asarray(
                        solver.inverse(target, previous), dtype=float
                    )
                except ValueError:
                    valid = False
                    break
                actual = np.asarray(solver.forward(solution))
                if (np.linalg.norm(actual[:3, 3] - target[:3, 3]) > 1e-3
                        or np.max(np.abs(solution - previous)) > 0.35):
                    valid = False
                    break
                solutions.append(solution)
                targets.append(target[:3, 3].copy())
                previous = solution
            if valid:
                candidates.append((axis, direction, np.asarray(solutions),
                                   np.asarray(targets)))
    if not candidates:
        raise RuntimeError(
            f"no continuous {distance:.3f} m Cartesian line is reachable "
            "from the 90-degree joint-4 posture"
        )
    axis, direction, waypoints, targets = min(
        candidates,
        key=lambda item: np.sum(np.abs(item[2][-1] - seed)),
    )
    # IK is solved at all 51 Cartesian samples. A uniformly decimated subset
    # becomes the time-parameterization control polygon; dense FK below still
    # validates the executed curve against the original line.
    control_indices = np.unique(np.r_[np.arange(0, len(waypoints), 5),
                                      len(waypoints) - 1])
    control_waypoints = waypoints[control_indices]
    segment_motion = np.linalg.norm(np.diff(control_waypoints, axis=0), axis=1)
    blend = min(0.005, 0.25 * float(np.min(segment_motion[segment_motion > 0])))
    limits = np.ones(robot.dof)
    trajectory = hm.RnTrajectory(
        control_waypoints, 0.7 * limits, 1.4 * limits, 4.0 * limits,
        blend_tolerance=blend,
    )
    # Dense FK validation catches joint interpolation that bows away from the
    # requested TCP line even when all IK waypoints themselves are accurate.
    _, sampled_q, _, _, _ = trajectory.sample_uniform(501)
    sampled_tcp = np.asarray([solver.forward(q)[:3, 3] for q in sampled_q])
    origin, unit = targets[0], np.zeros(3)
    unit[axis] = direction
    along = (sampled_tcp - origin) @ unit
    projected = origin + along[:, None] * unit
    deviation = float(np.max(np.linalg.norm(sampled_tcp - projected, axis=1)))
    if deviation > 0.003:
        raise RuntimeError(
            f"Cartesian line interpolation deviation {deviation * 1e3:.2f} mm"
        )
    return trajectory, "xyz"[axis], direction, distance, deviation, seed[3]


def _sample_for_charts(trajectory, count: int = 1601):
    """Sample densely and preserve both sides of every time-law boundary."""
    uniform = np.linspace(0.0, trajectory.duration, count)
    breakpoints = np.asarray(trajectory.breakpoints)
    epsilon = max(trajectory.duration, 1.0) * 1e-8
    boundary_neighbours = np.concatenate((
        np.clip(breakpoints - epsilon, 0.0, trajectory.duration),
        breakpoints,
        np.clip(breakpoints + epsilon, 0.0, trajectory.duration),
    ))
    times = np.unique(np.concatenate((uniform, boundary_neighbours)))
    return times, trajectory.sample(times)


def load_robot(urdf_path: Path):
    robot = hm.Robot(str(urdf_path))
    if not robot.has_kinematics:
        raise RuntimeError("URDF does not contain a supported serial chain")
    robot.load_visuals()
    topology = _tree_topology(robot)
    transforms = _link_transforms(robot, np.zeros(robot.dof), topology)
    return robot, transforms, topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--trajectory", choices=("joint", "cartesian-line"),
                        default="joint")
    parser.add_argument("--line-distance", type=float, default=0.35,
                        help="exact requested TCP line length [m]")
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    robot, transforms, topology = load_robot(urdf_path)
    mesh_count = sum(len(link.visuals) for link in robot.links)
    line_info = None
    if args.trajectory == "cartesian-line":
        if args.line_distance <= 0.0:
            raise SystemExit("--line-distance must be positive")
        trajectory, axis, direction, line_length, line_error, elbow_angle = \
            create_cartesian_line_trajectory(robot, args.line_distance)
        line_info = (axis, direction, line_length, line_error, elbow_angle)
    else:
        trajectory = create_demo_trajectory(robot)
    if args.validate_only:
        for link in robot.links:
            for visual in link.visuals:
                visual_mesh(hm, visual, urdf_path.parent)
        report = trajectory.constraint_report(samples=1601)
        line_text = ""
        if line_info is not None:
            axis, direction, line_length, line_error, elbow_angle = line_info
            line_text = (
                f", TCP line={axis}{direction:+.0f} {line_length:.3f}m, "
                f"uniform IK samples=51, J4={np.rad2deg(elbow_angle):+.0f}deg, "
                f"deviation={line_error * 1e3:.2f}mm"
            )
        print(
            f"validated {robot.name}: {len(robot.links)} links, "
            f"{robot.dof} DoF, {mesh_count} visuals, "
            f"trajectory {trajectory.duration:.3f}s, "
            f"velocity continuity="
            f"{'OK' if report['velocity_continuous'] else 'JUMP'}, "
            f"acceleration continuity="
            f"{'OK' if report['acceleration_continuous'] else 'JUMP'}"
            f"{line_text}"
        )
        return

    server = viser.ViserServer(port=args.port)
    handles = []
    for link, transform in zip(robot.links, transforms):
        if not link.visuals:
            handles.append([])
            continue
        link_handles = []
        for index, visual in enumerate(link.visuals):
            mesh = visual_mesh(hm, visual, urdf_path.parent)
            visual_pose = transform @ visual.origin
            wxyz, position = pose_components(visual_pose)
            handle = server.scene.add_mesh_trimesh(
                f"/robot/{link.name}/visual_{index}",
                mesh,
                scale=tuple(visual.scale),
                wxyz=wxyz,
                position=position,
            )
            link_handles.append((handle, visual.origin))
        handles.append(link_handles)

    initial_joints = np.asarray(trajectory.position(0.0), dtype=float)
    joints = initial_joints.copy()
    sliders = []
    performance = ViserPerformanceMonitor(server.gui)
    tcp_pose = robot.kinematics.forward(joints)
    tcp_wxyz, tcp_position = pose_components(tcp_pose)
    tcp_frame = server.scene.add_frame(
        "/frames/actual_tcp", axes_length=0.09, axes_radius=0.004,
        origin_radius=0.01, origin_color=(40, 190, 255),
        wxyz=tcp_wxyz, position=tcp_position,
    )
    tcp_marker = server.scene.add_icosphere(
        "/markers/actual_tcp", radius=0.012, color=(40, 190, 255),
        material="toon5", position=tcp_position,
    )
    if line_info is not None:
        _, path_q, _, _, _ = trajectory.sample_uniform(200)
        tcp_path = np.asarray([
            robot.kinematics.forward(q)[:3, 3] for q in path_q
        ])
        add_line_segments(
            server.scene, "/trajectory/tcp_line",
            np.stack((tcp_path[:-1], tcp_path[1:]), axis=1),
            (245, 165, 45), line_width=3.0,
        )
    dashboard = server.gui.add_markdown("### Robot State")
    state = {
        "mode": "Manual", "time": 0.0, "duration": 0.0,
        "last_gui_publish": -np.inf,
    }

    chart_time, chart_states = _sample_for_charts(trajectory)
    chart_position, chart_velocity, chart_acceleration, _ = chart_states
    palette = (
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab",
    )
    joint_labels = tuple(
        f"J{index + 1}" for index in range(len(robot.actuated_joints))
    )
    chart_series = ({"label": "time [s]"},) + tuple(
        {"label": joint_labels[index],
         "stroke": palette[index % len(palette)],
         "width": 1.7, "pxAlign": False, "cap": "round",
         "points": {"show": False}}
        for index, joint in enumerate(robot.actuated_joints)
    )
    chart_legend = {
        "show": True, "live": True, "isolate": True,
        "markers": {"show": True, "width": 3.0},
    }
    chart_cursor = {
        "show": True, "x": True, "y": True,
        "sync": {"key": "robot-trajectory"},
    }
    joint_table = [
        "| Label | URDF joint | Color |\n|:--:|:--|:--:|"
    ]
    joint_table.extend(
        f"| **{label}** | `{joint.name}` | "
        f"<span style='color:{palette[index % len(palette)]}'>■</span> |"
        for index, (label, joint) in enumerate(
            zip(joint_labels, robot.actuated_joints)
        )
    )
    with server.gui.add_folder("Trajectory curves"):
        server.gui.add_markdown("\n".join(joint_table))
        trajectory_cursor = server.gui.add_slider(
            "Shared trajectory time [s]", min=0.0,
            max=trajectory.duration, step=trajectory.duration / 400.0,
            initial_value=0.0, disabled=True,
        )
        server.gui.add_uplot(
            (chart_time, *chart_position.T), chart_series,
            title="Position q [rad]", height=280,
            axes=({"label": "time [s]"}, {"label": "q [rad]"}),
            legend=chart_legend, cursor=chart_cursor,
        )
        server.gui.add_uplot(
            (chart_time, *chart_velocity.T), chart_series,
            title="Velocity dq/dt [rad/s]", height=280,
            axes=({"label": "time [s]"}, {"label": "dq/dt [rad/s]"}),
            legend=chart_legend, cursor=chart_cursor,
        )
        server.gui.add_uplot(
            (chart_time, *chart_acceleration.T), chart_series,
            title="Acceleration d²q/dt² [rad/s²]", height=280,
            axes=({"label": "time [s]"},
                  {"label": "d²q/dt² [rad/s²]"}),
            legend=chart_legend, cursor=chart_cursor,
        )
        server.gui.add_markdown(
            "The time slider above is the shared cursor for all three plots. "
            "Hover a chart for per-joint values and zoom controls."
        )

    def update_dashboard() -> None:
        tcp = robot.kinematics.forward(joints)
        xyz = tcp[:3, 3]
        progress = (
            state["time"] / state["duration"] * 100.0
            if state["duration"] > 0.0 else 0.0
        )
        dashboard.content = (
            "### Robot State\n"
            "🔵 Actual TCP\n\n"
            "| Metric | Value |\n|:--|--:|\n"
            f"| Mode | **{state['mode']}** |\n"
            f"| Trajectory | **{state['time']:.2f} / "
            f"{state['duration']:.2f} s ({progress:.1f}%)** |\n"
            f"| ‖q‖ | **{np.linalg.norm(joints):.3f} rad** |\n"
            f"| TCP X / Y / Z | **{xyz[0]:+.3f} / {xyz[1]:+.3f} / "
            f"{xyz[2]:+.3f} m** |"
        )

    def update_scene(frame_started=None) -> None:
        if frame_started is None:
            frame_started = time.perf_counter()
        transforms_now = _link_transforms(robot, joints, topology)
        for transform, link_handles in zip(transforms_now, handles):
            for handle, visual_origin in link_handles:
                wxyz, position = pose_components(transform @ visual_origin)
                handle.wxyz = wxyz
                handle.position = position
        tcp_wxyz, tcp_position = pose_components(
            robot.kinematics.forward(joints)
        )
        tcp_frame.wxyz, tcp_frame.position = tcp_wxyz, tcp_position
        tcp_marker.position = tcp_position
        now = time.monotonic()
        if now - state["last_gui_publish"] >= 0.05:
            trajectory_cursor.value = min(
                float(state["time"]), trajectory.duration
            )
            update_dashboard()
            state["last_gui_publish"] = now
        performance.record(frame_started)

    for index, joint in enumerate(robot.actuated_joints):
        slider = server.gui.add_slider(
            joint_labels[index],
            min=float(joint.limit.lower),
            max=float(joint.limit.upper),
            step=0.01,
            initial_value=float(joints[index]),
            hint=f"{joint.name} · radians",
        )

        @slider.on_update
        def _on_joint_update(event, joint_index=index):
            joints[joint_index] = event.target.value
            update_scene()

        sliders.append(slider)

    reset = server.gui.add_button("Reset")
    play = server.gui.add_button("Play constrained trajectory")
    stop = server.gui.add_button("Stop trajectory")
    animation_stop = threading.Event()
    animation_lock = threading.Lock()

    @reset.on_click
    def _on_reset(_event):
        animation_stop.set()
        joints[:] = initial_joints
        state.update(mode="Manual", time=0.0, duration=0.0)
        for slider, value in zip(sliders, initial_joints):
            slider.value = float(value)
        update_scene()

    def run_trajectory() -> None:
        if not animation_lock.acquire(blocking=False):
            return
        try:
            animation_stop.clear()
            mode = ("Cartesian TCP line" if line_info is not None
                    else "Limit-aware joint trajectory")
            state.update(mode=mode, duration=trajectory.duration)
            while True:
                start_time = time.monotonic()
                next_frame = start_time
                last_gui_update = start_time - 1.0
                while not animation_stop.is_set():
                    frame_started = time.perf_counter()
                    now = time.monotonic()
                    elapsed = min(now - start_time, trajectory.duration)
                    state["time"] = elapsed
                    joints[:] = trajectory.position(elapsed)
                    update_scene(frame_started)
                    # Joint sliders are diagnostic controls. Updating them less
                    # frequently avoids flooding the websocket and keeps mesh
                    # animation cadence stable.
                    if now - last_gui_update >= 0.05:
                        for slider, value in zip(sliders, joints):
                            slider.value = float(value)
                        last_gui_update = now
                    if elapsed >= trajectory.duration:
                        break
                    next_frame += 1.0 / 60.0
                    time.sleep(max(0.0, next_frame - time.monotonic()))
                if animation_stop.is_set() or not args.loop:
                    break
        finally:
            animation_lock.release()

    def start_trajectory() -> None:
        threading.Thread(target=run_trajectory, daemon=True).start()

    @play.on_click
    def _on_play(_event):
        start_trajectory()

    @stop.on_click
    def _on_stop(_event):
        animation_stop.set()

    if args.autoplay:
        start_trajectory()

    update_scene()

    print(f"Viewing {robot.name} from {urdf_path}; press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        animation_stop.set()
        server.stop()


if __name__ == "__main__":
    main()
