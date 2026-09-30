"""Differentiate batched TOPPRA durations and samples on CPU or CUDA."""

from __future__ import annotations

import argparse

import torch
from holistic_motion.trajectory import retime_path_torch


def optimize_waypoints(waypoints, velocity, acceleration, steps):
    """Reduce duration while keeping endpoints fixed and interior changes bounded."""
    reference = waypoints.detach()
    velocity, acceleration = velocity.detach(), acceleration.detach()
    offsets = torch.zeros_like(reference[:, 1:-1], requires_grad=True)
    optimizer = torch.optim.Adam([offsets], lr=0.05)
    best_loss = float("inf")
    best_points = reference
    initial_loss = None
    for step in range(steps + 1):
        # For these example paths, this bound keeps adjacent points distinct.
        interior = reference[:, 1:-1] + 0.1 * offsets.tanh()
        candidate = torch.cat((reference[:, :1], interior, reference[:, -1:]), dim=1)
        trajectory = retime_path_torch(candidate, velocity, acceleration, grid_size=61)
        loss = trajectory.duration.sum() + 0.1 * (candidate - reference).square().mean()
        value = loss.item()
        if initial_loss is None:
            initial_loss = value
        if value < best_loss:
            best_loss, best_points = value, candidate.detach().clone()
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not bool(torch.isfinite(offsets.grad).all()):
            raise RuntimeError("waypoint optimization produced non-finite gradients")
        optimizer.step()
    # Active constraints can switch during training; report the best iterate.
    with torch.no_grad():
        result = retime_path_torch(best_points, velocity, acceleration, grid_size=61)
    print(f"optimization objective: {initial_loss:.6f} -> {best_loss:.6f}")
    print(f"optimized durations: {result.duration.cpu().tolist()}")
    print(
        f"largest waypoint adjustment: {(best_points - reference).abs().max().item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--optimize-steps", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requires a compatible GPU and CUDA-enabled PyTorch")
    if args.samples < 2:
        parser.error("--samples must be at least two")
    if args.optimize_steps < 0 or args.threads < 1:
        parser.error("--optimize-steps must be non-negative and --threads positive")
    torch.set_num_threads(args.threads)

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
    if args.optimize_steps:
        optimize_waypoints(waypoints, velocity, acceleration, args.optimize_steps)


if __name__ == "__main__":
    main()
