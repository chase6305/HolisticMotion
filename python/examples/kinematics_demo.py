#!/usr/bin/env python3
"""FK, IK, Jacobian, and null-space demo for an external 7R arm URDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import import_holistic_motion


hm = import_holistic_motion()

DEFAULT_ASSET_DIR = Path(
    "/home/ubuntu/workspace/chase/HumanoidAssets/"
    "Marvin_M6_S_CCS_696_V4.0"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--null-space-steps", type=int, default=5)
    parser.add_argument(
        "--solve-method",
        choices=("analytic", "seeded", "configuration", "all", "nearest"),
        default="nearest",
    )
    args = parser.parse_args()

    urdf = args.urdf or DEFAULT_ASSET_DIR / f"{args.arm}_arm.urdf"
    robot = hm.Robot(str(urdf.resolve()))
    if not isinstance(robot.kinematics, hm.SRSKinematics):
        raise RuntimeError("example requires a serial seven-revolute chain")

    solver = robot.kinematics
    geometry = solver.analyze_geometry()
    solver.set_tcp(np.eye(4))
    solver.set_user_frame(np.eye(4))

    # A non-singular configuration inside both arm limit sets.
    reference = np.array([0.15, -0.35, 0.25, -0.7, 0.2, 0.3, -0.15])
    target = solver.forward(reference)
    methods = {
        "seeded": hm.SRSSolveMethod.SEEDED_NUMERICAL,
        "configuration": hm.SRSSolveMethod.CONFIGURATION,
        "all": hm.SRSSolveMethod.ALL_CONFIGURATIONS,
        "nearest": hm.SRSSolveMethod.NEAREST_REDUNDANCY,
    }
    if args.solve_method == "analytic":
        configuration = solver.configuration(reference)
        solutions = [
            solver.analytic_solution(target, configuration, reference)
        ]
    else:
        solutions = solver.solve(target, reference, methods[args.solve_method])
    solution = solutions[0]
    jacobian = solver.jacobian(solution)

    preferred_direction = np.zeros(7)
    preferred_direction[2] = 1.0
    null_velocity = solver.null_space_velocity(
        solution, preferred_direction
    )

    planner = hm.NullSpacePlanner(solver)
    null_path = planner.plan(
        solution,
        preferred_direction,
        steps=args.null_space_steps,
        step_size=0.01,
    )

    max_pose_error = max(
        np.max(np.abs(solver.forward(point) - target)) for point in null_path
    )
    print(f"robot: {robot.name}")
    print(f"URDF: {urdf.resolve()}")
    print(f"DoF: {robot.dof}")
    print(
        "SRS geometry: "
        f"closed-form={geometry.closed_form_compatible}, "
        f"upper/forearm={geometry.upper_arm_length:.4f}/"
        f"{geometry.forearm_length:.4f} m, "
        f"shoulder/wrist residual={geometry.shoulder_axis_residual:.2e}/"
        f"{geometry.wrist_axis_residual:.2e} m"
    )
    print(f"FK target:\n{target}")
    print(f"IK solution: {solution}")
    print(f"SRS solve method: {args.solve_method} ({len(solutions)} solution(s))")
    print(f"IK reference error: {np.linalg.norm(solution - reference):.3e}")
    print(f"Jacobian shape: {jacobian.shape}")
    print(f"null-space residual: {np.linalg.norm(jacobian @ null_velocity):.3e}")
    print(f"null-space path points: {len(null_path)}")
    print(f"maximum TCP matrix error along path: {max_pose_error:.3e}")


if __name__ == "__main__":
    main()
