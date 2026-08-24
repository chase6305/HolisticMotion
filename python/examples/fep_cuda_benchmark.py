#!/usr/bin/env python3
"""Benchmark FEP batch FK and verify CPU/CUDA numerical agreement."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _bootstrap import import_holistic_motion


hm = import_holistic_motion()


def measure(solver, joints, backend, repeats):
    solver.forward_batch(joints, backend)
    started = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = solver.forward_batch(joints, backend)
    elapsed = (time.perf_counter() - started) / repeats
    return result, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--base", default="left_arm_base")
    parser.add_argument("--tip", default="left_ee")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error("batch size and repeats must be positive")
    robot = hm.Robot(str(args.urdf.resolve()))
    solver = robot.create_fep_kinematics(args.base, args.tip)
    if solver is None:
        parser.error("FEP requires a serial seven-revolute chain")
    lower, upper = solver.joint_limits
    joints = np.random.default_rng(42).uniform(
        lower, upper, size=(args.batch_size, 7)
    )
    cpu, cpu_time = measure(
        solver, joints, hm.FEPBackend.CPU, args.repeats
    )
    print(
        f"CPU: {cpu_time * 1000.0:.3f} ms/batch, "
        f"{args.batch_size / cpu_time:.0f} poses/s"
    )
    if not solver.cuda_available:
        print("CUDA runtime unavailable; GPU comparison skipped.")
        return
    cuda, cuda_time = measure(
        solver, joints, hm.FEPBackend.CUDA, args.repeats
    )
    error = float(np.max(np.abs(cpu - cuda)))
    print(
        f"CUDA: {cuda_time * 1000.0:.3f} ms/batch, "
        f"{args.batch_size / cuda_time:.0f} poses/s, "
        f"speedup={cpu_time / cuda_time:.2f}x, max error={error:.3e}"
    )
    if error > 1e-10:
        raise RuntimeError("CPU/CUDA FK mismatch exceeds tolerance")


if __name__ == "__main__":
    main()
