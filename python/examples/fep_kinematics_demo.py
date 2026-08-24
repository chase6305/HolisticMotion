#!/usr/bin/env python3
"""Validate the selectable FEP kinematics backends on a 7R URDF chain."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import import_holistic_motion


hm = import_holistic_motion()
METHODS = {
    "seeded": hm.FEPSolveMethod.SEEDED_NUMERICAL,
    "configuration": hm.FEPSolveMethod.CONFIGURATION,
    "all": hm.FEPSolveMethod.ALL_CONFIGURATIONS,
    "nearest": hm.FEPSolveMethod.NEAREST_REDUNDANCY,
    "compatible": hm.FEPSolveMethod.COMPATIBILITY,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--base", default="left_arm_base")
    parser.add_argument("--tip", default="left_ee")
    parser.add_argument("--method", choices=tuple(METHODS) + ("every",),
                        default="every")
    args = parser.parse_args()
    path = args.urdf.resolve()
    if not path.is_file():
        parser.error(f"URDF does not exist: {path}")
    robot = hm.Robot(str(path))
    solver = robot.create_fep_kinematics(args.base, args.tip)
    if solver is None:
        parser.error("FEP requires a serial seven-revolute chain")
    lower, upper = solver.joint_limits
    reference = np.clip(
        np.array([0.15, -0.35, 0.25, -0.7, 0.2, 0.3, -0.15]),
        lower + 1e-6, upper - 1e-6,
    )
    target = solver.forward(reference)
    selected = METHODS.items() if args.method == "every" else (
        (args.method, METHODS[args.method]),
    )
    for name, method in selected:
        try:
            solutions = solver.solve(target, reference, method)
        except ValueError:
            print(f"{name}: unavailable")
            continue
        errors = [
            np.linalg.norm(solver.forward(q)[:3, 3] - target[:3, 3])
            for q in solutions
        ]
        print(
            f"{name}: {len(solutions)} solution(s), "
            f"best TCP error={min(errors) * 1000.0:.3f} mm"
        )


if __name__ == "__main__":
    main()
