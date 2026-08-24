#!/usr/bin/env python3
"""Interactive Pink-style retargeting on a URDF robot in Viser."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _enter_python_toolkit_environment() -> None:
    if os.environ.get("HOLISTICMOTION_PURE_PYTHON") == "1":
        return
    repository = Path(__file__).resolve().parents[3]
    runner = repository / "scripts/run-python-toolkit.sh"
    if not runner.is_file() or os.environ.get("HOLISTICMOTION_DEMO_REEXEC"):
        return
    os.environ["HOLISTICMOTION_DEMO_REEXEC"] = "1"
    os.execv(
        str(runner),
        [str(runner), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )


_enter_python_toolkit_environment()

import numpy as np
import trimesh
from holistic_motion.kit.retargeting import (
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)
from holistic_motion.visualization.viser import (
    ViserPerformanceMonitor,
    pose_components,
)


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=repository / "examples/configs/marvin.json",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/home/ubuntu/workspace/chase/HumanoidAssets"),
    )
    parser.add_argument("--urdf", "--urdf-path", "--urdf_path", dest="urdf", type=Path)
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--gizmo-scale", type=float, default=0.18)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def control_pose(control) -> np.ndarray:
    transform = trimesh.transformations.quaternion_matrix(
        np.asarray(control.wxyz, dtype=float)
    )
    transform[:3, 3] = np.asarray(control.position, dtype=float)
    return transform


def load_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if not meshes:
            raise ValueError(f"visual mesh is empty: {path}")
        return trimesh.util.concatenate(meshes)
    return loaded


def build_solver(urdf: Path, profile: dict) -> PinkRetargetingSolver:
    retargeting = profile["retargeting"]
    acceleration_limits = {}
    configured_acceleration = retargeting["trajectory_limits"]["max_acceleration"]
    for group in ("left_arm", "right_arm"):
        acceleration_limits.update(
            zip(retargeting["joint_groups"][group], configured_acceleration)
        )
    return PinkRetargetingSolver(
        urdf,
        frames=retargeting["frames"],
        joint_groups=retargeting["joint_groups"],
        frame_tasks={
            "left_hand": FrameTask(position_cost=1.0, orientation_cost=0.25),
            "right_hand": FrameTask(position_cost=1.0, orientation_cost=0.25),
            "head": FrameTask(position_cost=0.25, orientation_cost=0.1),
        },
        posture_task=PostureTask(cost=1e-3),
        damping=1e-5,
        step_size=0.7,
        tolerance=2e-3,
        position_tolerance=2.5e-3,
        orientation_tolerance=3e-3,
        acceleration_limits=acceleration_limits,
        max_iterations=80,
    )


def frame_poses(solver, q: np.ndarray) -> dict[str, np.ndarray]:
    solver.pin.forwardKinematics(solver.model, solver.data, q)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    return {
        name: np.asarray(solver.data.oMf[frame_id].homogeneous).copy()
        for name, frame_id in solver._frame_ids.items()
    }


def main() -> None:
    args = parse_args()
    if args.rate <= 0.0 or args.gizmo_scale <= 0.0:
        raise SystemExit("--rate and --gizmo-scale must be positive")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    urdf = (args.urdf or (args.asset_root / profile["urdf"])).resolve()
    solver = build_solver(urdf, profile)
    q = np.asarray(solver.pin.neutral(solver.model), dtype=float)
    initial_targets = frame_poses(solver, q)

    model = solver.model
    visual_model = solver.pin.buildGeomFromUrdf(
        model,
        str(urdf),
        solver.pin.GeometryType.VISUAL,
        package_dirs=[str(urdf.parent)],
    )
    visual_data = visual_model.createData()
    solver.pin.updateGeometryPlacements(
        model, solver.data, visual_model, visual_data, q
    )
    print(
        f"validated Pink scene: robot={model.name}, nq={model.nq}, "
        f"visuals={len(visual_model.geometryObjects)}, modes="
        f"{','.join(mode.value for mode in solver.mode_manager.available_modes)}"
    )
    if args.validate_only:
        return

    try:
        import viser
    except ImportError as error:
        raise SystemExit(
            "install examples with `pip install -e '.[examples]'`"
        ) from error

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=3.0, height=3.0)
    mesh_handles = []
    for index, geometry in enumerate(visual_model.geometryObjects):
        mesh = load_mesh(geometry.meshPath)
        placement = visual_data.oMg[index]
        pose = np.asarray(placement.homogeneous)
        wxyz, position = pose_components(pose)
        handle = server.scene.add_mesh_trimesh(
            f"/robot/{geometry.name}",
            mesh,
            scale=tuple(np.asarray(geometry.meshScale, dtype=float)),
            wxyz=wxyz,
            position=position,
        )
        mesh_handles.append(handle)

    controls = {}
    colors = {
        "left_hand": (255, 155, 40),
        "right_hand": (210, 90, 255),
        "head": (70, 180, 255),
    }
    for name, pose in initial_targets.items():
        wxyz, position = pose_components(pose)
        controls[name] = server.scene.add_transform_controls(
            f"/targets/{name}",
            scale=args.gizmo_scale,
            line_width=4.0,
            wxyz=wxyz,
            position=position,
        )
        server.scene.add_frame(
            f"/references/{name}",
            axes_length=0.07,
            axes_radius=0.003,
            origin_radius=0.008,
            origin_color=colors[name],
            wxyz=wxyz,
            position=position,
        )

    mode = server.gui.add_dropdown(
        "Retargeting mode",
        ("left_arm", "right_arm", "dual_arm", "whole_body"),
        initial_value="dual_arm",
    )
    solve_enabled = server.gui.add_checkbox("Solve continuously", initial_value=True)
    strategy = server.gui.add_dropdown(
        "Control strategy",
        ("Iterative solve", "Single QP step"),
        initial_value="Iterative solve",
    )
    solve_once = server.gui.add_button("Solve once")
    reset = server.gui.add_button("Reset targets and posture")
    status = server.gui.add_markdown("### Pink retargeting\nWaiting for target input.")
    performance = ViserPerformanceMonitor(server.gui, target_fps=args.rate)
    dirty = {"value": True}
    solve_requested = {"value": False}
    step_progress = {"best": float("inf"), "stagnant": 0}

    def mark_dirty() -> None:
        dirty["value"] = True
        step_progress["best"] = float("inf")
        step_progress["stagnant"] = 0

    def update_control_visibility() -> None:
        selected = mode.value
        controls["left_hand"].visible = selected in (
            "left_arm",
            "dual_arm",
            "whole_body",
        )
        controls["right_hand"].visible = selected in (
            "right_arm",
            "dual_arm",
            "whole_body",
        )
        controls["head"].visible = selected == "whole_body"

    update_control_visibility()

    for control in controls.values():

        @control.on_update
        def _(_event) -> None:
            mark_dirty()

    @mode.on_update
    def _(_event) -> None:
        update_control_visibility()
        mark_dirty()

    @strategy.on_update
    def _(_event) -> None:
        mark_dirty()

    @solve_once.on_click
    def _(_event) -> None:
        solve_requested["value"] = True

    @reset.on_click
    def _(_event) -> None:
        nonlocal q
        q = np.asarray(solver.pin.neutral(solver.model), dtype=float)
        solver.reset(q)
        solver.set_posture_target(q)
        for name, pose in initial_targets.items():
            wxyz, position = pose_components(pose)
            controls[name].wxyz = wxyz
            controls[name].position = position
        mark_dirty()

    period = 1.0 / args.rate
    last_result = None
    while True:
        frame_started = time.perf_counter()
        if (solve_enabled.value and dirty["value"]) or solve_requested["value"]:
            solver.set_mode(mode.value)
            targets = {
                name: control_pose(control) for name, control in controls.items()
            }
            if strategy.value == "Single QP step":
                last_result = solver.step(targets, seed=q)
            else:
                last_result = solver.solve(targets, seed=q)
            q = last_result.configuration.copy()
            if strategy.value == "Single QP step" and not last_result.success:
                if last_result.residual < step_progress["best"] - 1e-7:
                    step_progress["best"] = last_result.residual
                    step_progress["stagnant"] = 0
                else:
                    step_progress["stagnant"] += 1
                dirty["value"] = step_progress["stagnant"] < 40
            else:
                dirty["value"] = False
            solve_requested["value"] = False

        solver.pin.forwardKinematics(model, solver.data, q)
        solver.pin.updateGeometryPlacements(
            model, solver.data, visual_model, visual_data, q
        )
        for index, handle in enumerate(mesh_handles):
            pose = np.asarray(visual_data.oMg[index].homogeneous)
            handle.wxyz, handle.position = pose_components(pose)

        if last_result is not None:
            state = (
                "🟢 converged"
                if last_result.success
                else "🔵 tracking"
                if dirty["value"]
                else "🟠 stopped"
            )
            target_lines = "\n".join(
                f"- `{name}`: position **{position * 1000.0:.2f} mm**, "
                f"orientation **{np.rad2deg(orientation):.2f}°**"
                for name, (
                    position,
                    orientation,
                ) in last_result.target_residuals.items()
            )
            status.content = (
                f"### Pink retargeting — {state}\n"
                f"- Mode: **{last_result.mode.value}**\n"
                f"- Strategy: **{strategy.value}**\n"
                f"- Termination: **{last_result.termination_reason}**\n"
                f"- Position residual: **{last_result.position_residual * 1000.0:.2f} mm**\n"
                f"- Orientation residual: "
                f"**{np.rad2deg(last_result.orientation_residual):.2f}°**\n"
                f"- Iterations: **{last_result.iterations}**\n"
                f"- Accepted steps / clipped axes: "
                f"**{last_result.accepted_steps} / {last_result.limit_hits}**\n"
                f"- Solve time: **{last_result.solve_ms:.3f} ms**\n"
                "#### Active targets\n"
                f"{target_lines}"
            )
        performance.record(frame_started)
        time.sleep(max(0.0, period - (time.perf_counter() - frame_started)))


if __name__ == "__main__":
    main()
