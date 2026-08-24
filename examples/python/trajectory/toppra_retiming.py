#!/usr/bin/env python3
"""Retime a multi-joint waypoint path with HolisticMotion TOPPRA."""

from __future__ import annotations

import argparse

import numpy as np
from holistic_motion.trajectory import ToppraTrajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    path = np.array([[0.0, 0.0], [0.45, -0.3], [1.0, 0.5]])
    trajectory = ToppraTrajectory(
        path, max_velocity=[1.0, 0.8], max_acceleration=[2.0, 1.5]
    )
    times, position, velocity, acceleration = trajectory.sample_uniform(args.samples)
    print(f"duration: {trajectory.duration:.6f} s")
    print(f"peak velocity: {np.max(np.abs(velocity), axis=0)}")
    print(f"peak sampled acceleration: {np.max(np.abs(acceleration), axis=0)}")

    if args.plot:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 1, sharex=True)
        for axis, values, label in zip(
            axes, (position, velocity, acceleration), ("q", "dq", "ddq")
        ):
            axis.plot(times, values)
            axis.set_ylabel(label)
            axis.grid(True)
        axes[-1].set_xlabel("time [s]")
        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
