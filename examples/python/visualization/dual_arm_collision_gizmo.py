#!/usr/bin/env python3
"""Interactively pose both humanoid arms with independent Viser gizmos."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import trimesh
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/examples"))
from _bootstrap import import_holistic_motion

hm = import_holistic_motion()

from holistic_motion.visualization.viser import (
    ViserPerformanceMonitor,
    add_line_segments,
    chain_joint_names,
    pose_components,
    tree_topology,
    tree_transforms,
    visual_mesh,
)

METHODS = {
    "Closed-form analytic": None,
    "Seeded numerical": hm.SRSSolveMethod.SEEDED_NUMERICAL,
    "Fixed configuration": hm.SRSSolveMethod.CONFIGURATION,
    "All configurations": hm.SRSSolveMethod.ALL_CONFIGURATIONS,
    "Nearest redundancy": hm.SRSSolveMethod.NEAREST_REDUNDANCY,
}

REPOSITORY_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = REPOSITORY_DIR / "examples/configs/marvin.json"


def _csv_links(value: str) -> list[str]:
    links = [item.strip() for item in value.split(",") if item.strip()]
    if not links:
        raise argparse.ArgumentTypeError("expected comma-separated link names")
    return links


def _descendant_links(robot, root: str) -> list[str]:
    """Return root and every descendant link in deterministic tree order."""
    result = []
    pending = [root]
    while pending:
        link_name = pending.pop(0)
        result.append(link_name)
        pending.extend(
            robot.get_joint(name).child_link
            for name in robot.get_link(link_name).child_joints
        )
    return result


def _arm_end_link(robot, side: str, requested: str | None) -> str:
    """Resolve common full-robot arm TCP names without asset-specific code."""
    link_names = {link.name for link in robot.links}
    if requested:
        if requested not in link_names:
            raise ValueError(f"link does not exist: {requested}")
        return requested
    candidates = (
        f"{side}_ee",
        f"{side}_hand_tcp",
        f"{side}_tcp",
        f"{side}_hand_base",
    )
    for candidate in candidates:
        if candidate in link_names:
            return candidate
    raise ValueError(
        f"could not find the {side} arm TCP; tried {', '.join(candidates)}"
    )


def _solve(solver, target, configuration, seed):
    """Prefer strict SRS IK, retaining its corrected compatibility path."""
    if not isinstance(solver, hm.SRSKinematics):
        return np.asarray(solver.inverse(target, seed)), "numerical"
    try:
        return np.asarray(
            solver.analytic_solution(target, configuration, seed)
        ), "closed-form"
    except ValueError:
        return np.asarray(
            solver.solve_configuration(target, configuration, seed)
        ), "analytic + correction"


def _control_transform(control) -> np.ndarray:
    transform = trimesh.transformations.quaternion_matrix(
        np.asarray(control.wxyz)
    )
    transform[:3, 3] = np.asarray(control.position)
    return transform


def _rotation_error(actual: np.ndarray, target: np.ndarray) -> float:
    relative = actual[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    assets = parser.add_mutually_exclusive_group(required=True)
    assets.add_argument("--asset-root", type=Path)
    assets.add_argument("--urdf", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--left-base")
    parser.add_argument("--right-base")
    parser.add_argument("--left-ee", help="override the auto-detected left TCP link")
    parser.add_argument("--right-ee", help="override the auto-detected right TCP link")
    parser.add_argument("--gizmo-scale", type=float, default=0.18)
    parser.add_argument(
        "--left-collision-links", type=_csv_links,
        help="comma-separated left-arm links; defaults to the IK chain links",
    )
    parser.add_argument(
        "--right-collision-links", type=_csv_links,
        help="comma-separated right-arm links; defaults to the IK chain links",
    )
    parser.add_argument(
        "--arm-self-collision", action="store_true",
        help="also check collision within each arm group",
    )
    parser.add_argument(
        "--no-collision", action="store_true",
        help="disable live Pinocchio/Coal collision queries",
    )
    parser.add_argument(
        "--disable-link-pair", action="append", default=[],
        metavar="LINK1:LINK2",
        help="disable every geometry pair between two links (repeatable)",
    )
    parser.add_argument(
        "--keep-marvin-wrist-pairs", action="store_true",
        help="do not apply the demo's Marvin wrist-pair exclusions",
    )
    parser.add_argument(
        "--position-only", action="store_true",
        help="disable gizmo rotation and solve translation only",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--benchmark-samples", type=int, default=0,
        help="with --validate-only, benchmark this many collision queries",
    )
    args = parser.parse_args()
    if not np.isfinite(args.gizmo_scale) or args.gizmo_scale <= 0.0:
        parser.error("--gizmo-scale must be finite and positive")
    if args.benchmark_samples < 0:
        parser.error("--benchmark-samples must be non-negative")
    if args.benchmark_samples and not args.validate_only:
        parser.error("--benchmark-samples requires --validate-only")

    profile_path = args.profile.expanduser().resolve()
    if not profile_path.is_file():
        parser.error(f"profile does not exist: {profile_path}")
    try:
        profile = json.loads(profile_path.read_text())
        profile_arms = profile["arms"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(f"invalid robot profile {profile_path}: {error}")
    urdf_path = (
        args.urdf.expanduser().resolve() if args.urdf else
        (args.asset_root.expanduser().resolve() / profile["urdf"])
    )
    if not urdf_path.is_file():
        parser.error(f"URDF does not exist: {urdf_path}")

    robot = hm.Robot(str(urdf_path))
    bases = {
        "left": args.left_base or profile_arms["left"]["base"],
        "right": args.right_base or profile_arms["right"]["base"],
    }
    try:
        end_links = {
            "left": _arm_end_link(
                robot, "left", args.left_ee or profile_arms["left"].get("ee")
            ),
            "right": _arm_end_link(
                robot, "right", args.right_ee or profile_arms["right"].get("ee")
            ),
        }
    except ValueError as error:
        parser.error(str(error))
    solvers = {
        side: robot.create_kinematics(bases[side], end_links[side])
        for side in ("left", "right")
    }
    if any(solver is None for solver in solvers.values()):
        parser.error("could not construct both arm chains from the URDF")

    topology = tree_topology(robot)
    zero_tree = tree_transforms(robot, {}, topology)
    base_world = {
        side: zero_tree[bases[side]] for side in solvers
    }
    world_base = {
        side: np.linalg.inv(transform)
        for side, transform in base_world.items()
    }
    joint_names = {
        side: chain_joint_names(
            robot, bases[side], end_links[side]
        )
        for side in solvers
    }
    current = {
        side: np.asarray(profile_arms[side]["seed"], dtype=float)
        for side in ("left", "right")
    }
    for side, solver in solvers.items():
        lower, upper = solver.joint_limits
        current[side] = np.clip(current[side], lower + 1e-6, upper - 1e-6)
    configurations = {
        side: solver.configuration(current[side])
        for side, solver in solvers.items()
        if isinstance(solver, hm.SRSKinematics)
    }
    candidates = {"left": [], "right": []}
    candidate_indices = {"left": 0, "right": 0}
    collision_model = None
    collision_links = {"left": [], "right": []}
    arm_chain_collision_links = {"left": [], "right": []}
    configure_collision_pairs = None
    if not args.no_collision:
        if not hasattr(hm, "CollisionModel"):
            parser.error("collision support is disabled; run ./scripts/build.sh")
        collision_model = hm.CollisionModel(
            str(urdf_path), [str(urdf_path.parent)]
        )
        available = set(collision_model.collision_link_names)
        for side in ("left", "right"):
            requested = (
                args.left_collision_links if side == "left"
                else args.right_collision_links
            )
            automatic_chain = [
                robot.get_joint(name).child_link for name in joint_names[side]
            ]
            automatic = automatic_chain + _descendant_links(
                robot, end_links[side]
            )
            collision_links[side] = list(dict.fromkeys(
                link for link in (requested or automatic) if link in available
            ))
            arm_chain_collision_links[side] = list(dict.fromkeys(
                link for link in (requested or automatic_chain)
                if link in available
            ))
        custom_disabled_pairs = []
        for value in args.disable_link_pair:
            first, separator, second = value.partition(":")
            if not separator or not first or not second:
                parser.error(f"invalid --disable-link-pair value: {value}")
            custom_disabled_pairs.append((first, second))

        def _configure_collision_pairs(
            include_self: bool, include_fixtures: bool = True
        ) -> int:
            enabled_groups = [("left_arm", "right_arm")]
            if include_self:
                enabled_groups.extend([
                    ("left_arm", "left_arm"),
                    ("right_arm", "right_arm"),
                ])
            selected_links = (
                collision_links if include_fixtures
                else arm_chain_collision_links
            )
            collision_model.set_collision_groups(
                {
                    "left_arm": selected_links["left"],
                    "right_arm": selected_links["right"],
                },
                enabled_groups,
            )
            disabled = list(custom_disabled_pairs)
            if include_self and not args.keep_marvin_wrist_pairs:
                disabled.extend(
                tuple(pair) for pair in profile.get("collision", {}).get(
                    "disabled_self_link_pairs", []
                )
                    if pair[0] in available and pair[1] in available
                )
            for first, second in disabled:
                collision_model.remove_collision_pairs_by_links(first, second)
            return collision_model.pair_count

        configure_collision_pairs = _configure_collision_pairs
        try:
            configure_collision_pairs(args.arm_self_collision)
        except ValueError as error:
            parser.error(f"invalid collision group: {error}")

    def tcp_world(side: str) -> np.ndarray:
        return base_world[side] @ solvers[side].forward(current[side])

    if args.validate_only:
        for side in solvers:
            target = solvers[side].forward(current[side])
            solver = solvers[side]
            configuration = configurations.get(side)
            solution, _ = _solve(
                solver, target, configuration, current[side]
            )
            actual = solvers[side].forward(solution)
            position_error = np.linalg.norm(
                actual[:3, 3] - target[:3, 3]
            )
            orientation_error = _rotation_error(actual, target)
            if position_error > 5e-4 or orientation_error > 1e-4:
                raise RuntimeError(
                    f"{side} arm IK round-trip failed: "
                    f"{position_error * 1000.0:.3f} mm, "
                    f"{np.degrees(orientation_error):.3f} deg"
                )
        collision_summary = "disabled"
        if collision_model:
            joint_positions = {}
            for side in ("left", "right"):
                joint_positions.update(zip(joint_names[side], current[side]))
            q = collision_model.configuration_from_joint_positions(
                joint_positions
            )
            report = collision_model.evaluate(q)
            nearest = report.minimum_distance
            collision_summary = (
                f"{collision_model.pair_count} pairs, "
                f"minimum={nearest.distance * 1000.0:.2f} mm, "
                f"query={report.query_time_ms:.3f} ms"
            )
            if args.benchmark_samples:
                for _ in range(10):
                    collision_model.evaluate(q)
                timings = np.asarray([
                    collision_model.evaluate(q).query_time_ms
                    for _ in range(args.benchmark_samples)
                ])
                print(
                    f"collision benchmark ({args.benchmark_samples} samples): "
                    f"avg={np.mean(timings):.3f} ms, "
                    f"P95={np.percentile(timings, 95):.3f} ms, "
                    f"max={np.max(timings):.3f} ms, "
                    f"rate={1000.0 / np.mean(timings):.1f} Hz"
                )
        print(
            f"validated {robot.name}: independent left/right gizmo IK, "
            f"DoF={len(joint_names['left'])}/{len(joint_names['right'])}, "
            f"arm collision={collision_summary}"
        )
        return

    robot.load_visuals()
    server = viser.ViserServer(port=args.port)
    performance = ViserPerformanceMonitor(server.gui)
    scene_lock = threading.RLock()
    positions = dict.fromkeys(
        joint_names["left"] + joint_names["right"], 0.0
    )
    positions.update(zip(joint_names["left"], current["left"]))
    positions.update(zip(joint_names["right"], current["right"]))
    handles: dict[str, list[tuple[object, np.ndarray]]] = {}
    initial_tree = tree_transforms(robot, positions, topology)
    for link in robot.links:
        handles[link.name] = []
        for index, visual in enumerate(link.visuals):
            mesh = visual_mesh(hm, visual, urdf_path.parent)
            origin = np.asarray(visual.origin)
            pose = initial_tree[link.name] @ origin
            wxyz, position = pose_components(pose)
            handle = server.scene.add_mesh_trimesh(
                f"/robot/{link.name}/visual_{index}", mesh,
                scale=tuple(visual.scale), wxyz=wxyz, position=position,
            )
            handles[link.name].append((handle, origin))

    controls = {}
    for side in ("left", "right"):
        pose = tcp_world(side)
        wxyz, position = pose_components(pose)
        controls[side] = server.scene.add_transform_controls(
            f"/targets/{side}_ee",
            scale=args.gizmo_scale,
            line_width=4.0,
            wxyz=wxyz,
            position=position,
            disable_rotations=args.position_only,
        )

    guides = {}
    guide_colors = {"left": (255, 155, 40), "right": (210, 90, 255)}
    for side in ("left", "right"):
        wxyz, position = pose_components(tcp_world(side))
        color = guide_colors[side]
        guides[side] = {
            "target": server.scene.add_frame(
                f"/guides/{side}/target", axes_length=0.09,
                axes_radius=0.004, origin_radius=0.009,
                origin_color=color, wxyz=wxyz, position=position,
            ),
            "actual": server.scene.add_icosphere(
                f"/guides/{side}/actual", radius=0.012,
                color=(40, 190, 255), material="toon5", position=position,
            ),
            "error": add_line_segments(
                server.scene,
                f"/guides/{side}/error", np.array([[position, position]]),
                color, 3.0,
            ),
        }
    method = server.gui.add_dropdown(
        "SRS solve method", tuple(METHODS),
        initial_value="Nearest redundancy",
    )
    active_arm = server.gui.add_dropdown(
        "Candidate arm", ("left", "right"), initial_value="left"
    )
    next_solution = server.gui.add_button("Next configuration")
    status = server.gui.add_markdown(
        "### Dual-arm SRS Solver\n"
        "Drag either 6D gizmo. Each arm keeps its own branch and candidates."
    )
    collision_status = server.gui.add_markdown(
        "### Arm Collision\nWaiting for the first scene update."
        if collision_model else "### Arm Collision\nDisabled."
    )
    reject_unsafe = server.gui.add_checkbox(
        "Reject unsafe IK", initial_value=False
    )
    safety_margin = server.gui.add_number(
        "Safety margin (mm)", initial_value=20.0,
        min=0.0, max=200.0, step=1.0,
    )
    exact_distance_rate = server.gui.add_number(
        "Exact distance rate (Hz)", initial_value=4.0,
        min=0.5, max=30.0, step=0.5,
    )
    collision_mode = server.gui.add_dropdown(
        "Collision mode",
        (
            "Between complete arms",
            "Between arm chains only",
            "Complete arms + self",
            "Disabled",
        ),
        initial_value=(
            "Complete arms + self" if args.arm_self_collision
            else "Between complete arms"
        ) if collision_model else "Disabled",
    )
    pair_selector = server.gui.add_dropdown(
        "Active collision pair", ("No active pairs",),
        initial_value="No active pairs",
    )
    disable_selected_pair = server.gui.add_button("Disable selected pair")
    restore_mode_pairs = server.gui.add_button("Restore mode pairs")
    collision_markers = {
        "first": server.scene.add_icosphere(
            "/collision/nearest_first", radius=0.018,
            color=(255, 50, 50), material="toon5", visible=False,
        ),
        "second": server.scene.add_icosphere(
            "/collision/nearest_second", radius=0.018,
            color=(255, 50, 50), material="toon5", visible=False,
        ),
    }
    collision_line = add_line_segments(
        server.scene, "/collision/nearest_segment",
        np.zeros((1, 2, 3)), (255, 50, 50), 5.0,
    )
    collision_line.visible = False
    collision_query_times = deque(maxlen=120)
    fast_query_times = deque(maxlen=120)
    exact_cache = {"report": None, "timestamp": 0.0}
    visible = server.gui.add_checkbox("Show robot", initial_value=True)
    reset = server.gui.add_button("Reset both arms")

    def update_collision(report=None) -> None:
        if collision_model is None:
            return
        if not collision_model.pair_count:
            collision_markers["first"].visible = False
            collision_markers["second"].visible = False
            collision_line.visible = False
            collision_status.content = "### Arm Collision — Disabled"
            return
        q = collision_model.configuration_from_joint_positions(positions)
        margin = float(safety_margin.value) / 1000.0
        fast_started = time.perf_counter()
        in_collision = collision_model.is_within_distance(q, 0.0)
        unsafe = (
            in_collision or
            collision_model.is_within_distance(q, margin)
        )
        fast_ms = (time.perf_counter() - fast_started) * 1000.0
        fast_query_times.append(fast_ms)
        now = time.perf_counter()
        exact_period = 1.0 / float(exact_distance_rate.value)
        if (report is not None or exact_cache["report"] is None or
                now - exact_cache["timestamp"] >= exact_period):
            report = report or collision_model.evaluate(q)
            exact_cache["report"] = report
            exact_cache["timestamp"] = now
            collision_query_times.append(float(report.query_time_ms))
        else:
            report = exact_cache["report"]
        collisions = report.collisions
        nearest = report.minimum_distance
        average_ms = float(np.mean(collision_query_times))
        p95_ms = float(np.percentile(collision_query_times, 95))
        fast_average_ms = float(np.mean(fast_query_times))
        first_point = np.asarray(nearest.nearest_point_first)
        second_point = np.asarray(nearest.nearest_point_second)
        collision_markers["first"].position = first_point
        collision_markers["second"].position = second_point
        collision_markers["first"].visible = True
        collision_markers["second"].visible = True
        collision_line.points = np.array([[first_point, second_point]])
        collision_line.visible = True
        state = (
            "🔴 COLLISION" if in_collision else
            "🟠 BELOW MARGIN" if unsafe else "🟢 CLEAR"
        )
        detail = ""
        if collisions:
            detail = "\n" + "\n".join(
                f"- `{item.first_geometry}` ↔ `{item.second_geometry}`: "
                f"**{item.distance * 1000.0:.2f} mm**"
                for item in collisions[:8]
            )
        collision_status.content = (
            f"### Arm Collision — {state}\n"
            f"- Active pairs: **{collision_model.pair_count}**\n"
            f"- Minimum distance: **{nearest.distance * 1000.0:.2f} mm**\n"
            f"- Fast threshold: **{fast_ms:.3f} ms** latest, "
            f"**{fast_average_ms:.3f} ms** avg\n"
            f"- Exact distance: **{report.query_time_ms:.3f} ms** latest, "
            f"**{average_ms:.3f} ms** avg, **{p95_ms:.3f} ms** P95"
            f"{detail}"
        )

    def arm_is_unsafe(margin: float) -> bool:
        if collision_model is None or not collision_model.pair_count:
            return False
        joint_positions = {}
        for side in ("left", "right"):
            joint_positions.update(zip(joint_names[side], current[side]))
        q = collision_model.configuration_from_joint_positions(joint_positions)
        return collision_model.is_within_distance(q, margin)

    pair_lookup = {}

    def refresh_pair_selector() -> None:
        nonlocal pair_lookup
        pair_lookup = {}
        if collision_model is not None:
            for index, pair in enumerate(collision_model.collision_pairs):
                label = (
                    f"[{index}] {pair.first_link}/{pair.first_geometry} ↔ "
                    f"{pair.second_link}/{pair.second_geometry}"
                )
                pair_lookup[label] = pair
        options = tuple(pair_lookup) or ("No active pairs",)
        pair_selector.options = options
        pair_selector.value = options[0]

    def update_scene(frame_started: float | None = None,
                     collision_report=None) -> None:
        started = frame_started or time.perf_counter()
        positions.update(zip(joint_names["left"], current["left"]))
        positions.update(zip(joint_names["right"], current["right"]))
        tree = tree_transforms(robot, positions, topology)
        for link_name, transform in tree.items():
            for handle, origin in handles[link_name]:
                handle.wxyz, handle.position = pose_components(
                    transform @ origin
                )
        update_collision(collision_report)
        performance.record(started)

    def sync_control(side: str) -> None:
        controls[side].wxyz, controls[side].position = pose_components(
            tcp_world(side)
        )

    def update_guides(side: str, target_world: np.ndarray) -> None:
        target_wxyz, target_position = pose_components(target_world)
        actual_world = tcp_world(side)
        _, actual_position = pose_components(actual_world)
        guides[side]["target"].wxyz = target_wxyz
        guides[side]["target"].position = target_position
        guides[side]["actual"].position = actual_position
        guides[side]["error"].points = np.array(
            [[actual_position, target_position]]
        )

    def dashboard(
        side: str, backend: str, target_base: np.ndarray,
        elapsed_ms: float, position_error: float, orientation_error: float,
    ) -> str:
        solver = solvers[side]
        count = len(candidates[side])
        index = candidate_indices[side] + 1 if count else 0
        configuration = configurations.get(side)
        branch = "n/a"
        redundancy = "n/a"
        if configuration is not None:
            branch = (
                f"{configuration.shoulder:+d} / {configuration.elbow:+d} / "
                f"{configuration.wrist:+d}"
            )
            redundancy = f"{np.degrees(configuration.redundancy):.2f}°"
        singular_values = np.linalg.svd(
            solver.jacobian(current[side]), compute_uv=False
        )
        xyz = target_base[:3, 3]
        return (
            "### Dual-arm SRS Solver\n"
            f"| Metric | {side.title()} arm |\n|:--|--:|\n"
            f"| Method | **{backend}** |\n"
            f"| Solution | **{index} / {count}** |\n"
            f"| S / E / W | **{branch}** |\n"
            f"| Redundancy | **{redundancy}** |\n"
            f"| Position error | **{position_error * 1000.0:.3f} mm** |\n"
            f"| Orientation error | **{np.degrees(orientation_error):.3f}°** |\n"
            f"| IK core time | **{elapsed_ms:.3f} ms** |\n"
            f"| Jacobian σ min | **{singular_values[-1]:.3e}** |\n\n"
            f"Target XYZ: `{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}` m"
        )

    def solve_arm(side: str, target_base: np.ndarray) -> tuple[str, float]:
        solver = solvers[side]
        started = time.perf_counter()
        if isinstance(solver, hm.SRSKinematics):
            selected = METHODS[method.value]
            if selected is None:
                values = [solver.analytic_solution(
                    target_base, configurations[side], current[side]
                )]
            else:
                values = solver.solve(target_base, current[side], selected)
            candidates[side] = [np.asarray(value) for value in values]
            backend = method.value
        else:
            candidates[side] = [
                np.asarray(solver.inverse(target_base, current[side]))
            ]
            backend = "Seeded numerical"
        if not candidates[side]:
            raise ValueError("no IK solution")
        candidate_indices[side] = 0
        current[side] = candidates[side][0]
        if isinstance(solver, hm.SRSKinematics):
            configurations[side] = solver.configuration(current[side])
        return backend, (time.perf_counter() - started) * 1000.0

    def follow_target(side: str, requested: np.ndarray) -> tuple[float, float]:
        """Follow a large drag using small seeded SE(3) continuation steps."""
        solver = solvers[side]
        start_pose = solver.forward(current[side])
        start_quaternion = trimesh.transformations.quaternion_from_matrix(
            start_pose
        )
        target_quaternion = trimesh.transformations.quaternion_from_matrix(
            requested
        )
        solution = current[side].copy()
        reached = 0.0
        started = time.perf_counter()
        for fraction in np.linspace(1.0 / 12.0, 1.0, 12):
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
                if isinstance(solver, hm.SRSKinematics):
                    values = solver.solve(
                        intermediate, solution,
                        hm.SRSSolveMethod.SEEDED_NUMERICAL,
                    )
                    if not values:
                        break
                    solution = np.asarray(values[0])
                else:
                    solution = np.asarray(solver.inverse(intermediate, solution))
            except ValueError:
                break
            reached = float(fraction)
        if reached > 0.0:
            current[side] = solution
            candidates[side] = [solution.copy()]
            candidate_indices[side] = 0
            if isinstance(solver, hm.SRSKinematics):
                configurations[side] = solver.configuration(solution)
        return reached, (time.perf_counter() - started) * 1000.0

    pending = {"left": None, "right": None}
    pending_lock = threading.Lock()
    pending_event = threading.Event()

    def process_target(arm: str, target_world: np.ndarray) -> None:
        with scene_lock:
            started = time.perf_counter()
            previous = current[arm].copy()
            target_base = world_base[arm] @ target_world
            if args.position_only:
                target_base[:3, :3] = solvers[arm].forward(
                    current[arm]
                )[:3, :3]
            reached = 1.0
            try:
                backend, solve_ms = solve_arm(arm, target_base)
            except ValueError:
                reached, solve_ms = follow_target(arm, target_base)
                backend = f"Seeded continuation ({reached * 100.0:.0f}%)"
                if reached == 0.0:
                    status.content = (
                        "### Dual-arm SRS Solver\n"
                        f"**{arm.title()} arm: no local IK step.**\n\n"
                        "The gizmo is still movable. Bring it toward the blue "
                        "actual-TCP marker, reduce rotation, or select "
                        "Seeded numerical."
                    )
                    update_guides(arm, target_world)
                    return
            margin = float(safety_margin.value) / 1000.0
            unsafe = arm_is_unsafe(margin)
            if reject_unsafe.value and unsafe:
                current[arm] = previous
                candidates[arm] = []
                candidate_indices[arm] = 0
                if isinstance(solvers[arm], hm.SRSKinematics):
                    configurations[arm] = solvers[arm].configuration(previous)
                update_scene(started)
                update_guides(arm, target_world)
                status.content = (
                    "### Dual-arm SRS Solver\n"
                    f"**Rejected {arm} IK solution:** it violates the "
                    f"{margin * 1000.0:.2f} mm safety margin."
                )
                return
            actual = solvers[arm].forward(current[arm])
            update_scene(started)
            update_guides(arm, target_world)
            position_error = np.linalg.norm(
                actual[:3, 3] - target_base[:3, 3]
            )
            orientation_error = _rotation_error(actual, target_base)
            active_arm.value = arm
            status.content = dashboard(
                arm, backend, target_base, solve_ms,
                position_error, orientation_error,
            )

    def solver_worker() -> None:
        while True:
            pending_event.wait()
            time.sleep(1.0 / 60.0)
            with pending_lock:
                requests = pending.copy()
                pending["left"] = None
                pending["right"] = None
                pending_event.clear()
            for arm, target_world in requests.items():
                if target_world is not None:
                    process_target(arm, target_world)

    threading.Thread(target=solver_worker, daemon=True).start()

    for side in ("left", "right"):
        control = controls[side]

        @control.on_update
        def _solve_control(event, arm=side) -> None:
            target_world = _control_transform(event.target)
            update_guides(arm, target_world)
            with pending_lock:
                pending[arm] = target_world
            pending_event.set()

    @next_solution.on_click
    def _next_solution(_event) -> None:
        side = active_arm.value
        with scene_lock:
            if not candidates[side]:
                return
            candidate_indices[side] = (
                candidate_indices[side] + 1
            ) % len(candidates[side])
            current[side] = candidates[side][candidate_indices[side]].copy()
            solver = solvers[side]
            if isinstance(solver, hm.SRSKinematics):
                configurations[side] = solver.configuration(current[side])
            update_scene()
            target_world = _control_transform(controls[side])
            target_base = world_base[side] @ target_world
            actual = solver.forward(current[side])
            position_error = np.linalg.norm(
                actual[:3, 3] - target_base[:3, 3]
            )
            orientation_error = _rotation_error(actual, target_base)
            update_guides(side, target_world)
            status.content = dashboard(
                side, method.value, target_base, 0.0,
                position_error, orientation_error,
            )

    @visible.on_update
    def _set_visibility(event) -> None:
        for link_handles in handles.values():
            for handle, _ in link_handles:
                handle.visible = event.target.value

    @collision_mode.on_update
    def _set_collision_mode(event) -> None:
        if collision_model is None or configure_collision_pairs is None:
            return
        with scene_lock:
            if event.target.value == "Disabled":
                collision_model.clear_collision_pairs()
            else:
                configure_collision_pairs(
                    event.target.value == "Complete arms + self",
                    event.target.value != "Between arm chains only",
                )
            refresh_pair_selector()
            update_scene()

    @disable_selected_pair.on_click
    def _disable_selected_pair(_event) -> None:
        if collision_model is None:
            return
        pair = pair_lookup.get(pair_selector.value)
        if pair is None:
            return
        with scene_lock:
            collision_model.remove_collision_pair(
                pair.first_geometry, pair.second_geometry
            )
            refresh_pair_selector()
            update_scene()

    @restore_mode_pairs.on_click
    def _restore_mode_pairs(_event) -> None:
        if collision_model is None or configure_collision_pairs is None:
            return
        with scene_lock:
            if collision_mode.value == "Disabled":
                collision_model.clear_collision_pairs()
            else:
                configure_collision_pairs(
                    collision_mode.value == "Complete arms + self",
                    collision_mode.value != "Between arm chains only",
                )
            refresh_pair_selector()
            update_scene()

    @reset.on_click
    def _reset(_event) -> None:
        with scene_lock:
            current["left"] = np.asarray(
                profile_arms["left"]["seed"], dtype=float
            )
            current["right"] = np.asarray(
                profile_arms["right"]["seed"], dtype=float
            )
            for side, solver in solvers.items():
                lower, upper = solver.joint_limits
                current[side] = np.clip(
                    current[side], lower + 1e-6, upper - 1e-6
                )
                if isinstance(solver, hm.SRSKinematics):
                    configurations[side] = solver.configuration(current[side])
            update_scene()
            sync_control("left")
            sync_control("right")
            for side in ("left", "right"):
                candidates[side] = []
                candidate_indices[side] = 0
                update_guides(side, tcp_world(side))
            status.content = "Both arms reset to their initial poses."

    refresh_pair_selector()
    update_scene()
    print(f"Viser dual-arm gizmo demo: http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
