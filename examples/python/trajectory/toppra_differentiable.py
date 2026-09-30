"""Differentiate batched TOPPRA durations and samples on CPU or CUDA."""

from __future__ import annotations

import argparse

import torch
from holistic_motion.trajectory import retime_path_torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requires a compatible GPU and CUDA-enabled PyTorch")
    if args.samples < 2:
        parser.error("--samples must be at least two")

    waypoints = torch.tensor(
        [[[0.0, 0.0], [0.45, -0.3], [1.0, 0.5]], [[0.1, 0.2], [0.3, -0.4], [0.8, 0.7]]],
        dtype=torch.float64,
        device=args.device,
        requires_grad=True,
    )
    velocity = torch.tensor(
        [1.0, 0.8], dtype=torch.float64, device=args.device, requires_grad=True
    )
    acceleration = torch.tensor(
        [2.0, 1.5], dtype=torch.float64, device=args.device, requires_grad=True
    )
    trajectory = retime_path_torch(waypoints, velocity, acceleration, grid_size=61)
    times, position, _, _ = trajectory.sample_uniform(args.samples)
    loss = trajectory.duration.sum() + 0.01 * position.square().mean()
    loss.backward()
    print(f"device: {position.device}; positions: {tuple(position.shape)}")
    print(f"durations: {trajectory.duration.detach().cpu().tolist()}")
    print(f"final sample times: {times[:, -1].detach().cpu().tolist()}")
    print(
        f"waypoint gradient norms: {waypoints.grad.flatten(1).norm(dim=1).cpu().tolist()}"
    )
    print(f"velocity limit gradients: {velocity.grad.cpu().tolist()}")
    print(f"acceleration limit gradients: {acceleration.grad.cpu().tolist()}")


if __name__ == "__main__":
    main()
