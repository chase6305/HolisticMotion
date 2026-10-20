"""Compare Torch spline and complete trajectory costs against a source snapshot.

Run with --device cpu or cuda and an explicit --baseline module. The spline
scope isolates coefficient construction; trajectory includes timing, sampling
and optional reverse-mode gradients. Timings alternate baseline/candidate order.
CUDA peak added bytes exclude the already allocated inputs and model weights.
Saved bytes count tensor references (including aliases), not allocator memory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import torch
from compare_toppra_torch import load_module


def inputs_for(scope, batch, count, dof, device, seed):
    generator = torch.Generator().manual_seed(seed + count + dof)
    if scope == "spline":
        h = 0.5 + torch.rand(batch, count - 1, generator=generator, dtype=torch.float64)
        h = h / h.sum(1, keepdim=True)
        slopes = torch.randn(
            batch, count - 1, dof, generator=generator, dtype=torch.float64
        )
        return tuple(v.to(device).requires_grad_() for v in (h, slopes))
    s = torch.linspace(0, 1, count, dtype=torch.float64)
    axes = torch.arange(dof, dtype=torch.float64)
    points = s[:, None] + 0.07 * torch.sin(
        (1 + axes.remainder(4)) * s[:, None] + 0.3 * axes
    )
    points = points[None].repeat(batch, 1, 1)
    points = (
        points * torch.linspace(0.7, 1.0, batch, dtype=torch.float64)[:, None, None]
    )
    points += torch.arange(batch, dtype=torch.float64)[:, None, None] * 0.01
    velocity = torch.linspace(0.8, 1.3, dof, dtype=torch.float64)
    acceleration = torch.linspace(1.5, 2.5, dof, dtype=torch.float64)
    return tuple(
        v.to(device).requires_grad_() for v in (points, velocity, acceleration)
    )


def execute(module, inputs, scope, count, grid_size, backward, end_speed):
    context = contextlib.nullcontext() if backward else torch.no_grad()
    with context:
        profile = ()
        if scope == "spline":
            values = module.TorchToppraTrajectory._derivative_coefficients(*inputs)
            loss_values = tuple(v / count**order for order, v in enumerate(values))
        else:
            trajectory = module.TorchToppraTrajectory(
                *inputs, grid_size=grid_size, end_path_velocity=end_speed
            )
            fractions = torch.linspace(
                0.0037, 0.9961, 101, dtype=torch.float64, device=inputs[0].device
            )
            query = trajectory.duration[..., None] * fractions
            values = (trajectory.duration, *trajectory.sample(query))
            loss_values = values
            result = trajectory.result
            profile = (
                result.gridpoints,
                result.path_speeds,
                result.path_accelerations,
                result.times,
            )
        if backward:
            loss = sum(value.square().mean() for value in loss_values)
            gradients = torch.autograd.grad(loss, inputs)
            return (*profile, *values, *gradients)
        return (*profile, *values)


def saved_references(call):
    calls = 0
    referenced_bytes = 0

    def save(value):
        nonlocal calls, referenced_bytes
        calls += 1
        referenced_bytes += value.numel() * value.element_size()
        return value.detach()

    with torch.autograd.graph.saved_tensors_hooks(save, lambda value: value):
        call()
    return {"save_calls": calls, "referenced_bytes": referenced_bytes}


def measure(call, device):
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        start_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start_event.record()
    start = time.perf_counter_ns()
    outputs = call()
    if device == "cuda":
        end_event.record()
        end_event.synchronize()
    result = {"wall_ms": (time.perf_counter_ns() - start) / 1e6}
    if device == "cuda":
        result.update(
            event_ms=start_event.elapsed_time(end_event),
            peak_added_bytes=torch.cuda.max_memory_allocated() - allocated_before,
        )
    del outputs
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "python/holistic_motion/trajectory/toppra_torch.py",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=("spline", "trajectory"),
        default=["spline", "trajectory"],
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--dofs", type=int, nargs="+", default=[7])
    parser.add_argument(
        "--waypoint-counts", type=int, nargs="+", default=[8, 16, 32, 128, 512]
    )
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--end-speeds", type=float, nargs="+", default=[0.0, 0.005])
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batches + args.dofs) < 1 or min(args.waypoint_counts) < 2:
        parser.error("batches/DOFs must be positive and waypoint counts at least two")
    if args.rounds < 2 or args.grid_size < 2:
        parser.error("rounds and grid size must be at least two")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    modules = [
        load_module(path, name)
        for path, name in (
            (args.baseline, "spline_baseline"),
            (args.candidate, "spline_candidate"),
        )
    ]
    cases = []
    for scope, batch, count, dof in itertools.product(
        args.scopes, args.batches, args.waypoint_counts, args.dofs
    ):
        inputs = inputs_for(scope, batch, count, dof, args.device, args.seed)
        for backward, end_speed in itertools.product(
            (False, True), [0.0] if scope == "spline" else args.end_speeds
        ):
            calls = [
                partial(
                    execute,
                    module,
                    inputs,
                    scope,
                    count,
                    args.grid_size,
                    backward,
                    end_speed,
                )
                for module in modules
            ]
            outputs = [call() for call in calls]
            for old, new in zip(*outputs):
                torch.testing.assert_close(new, old, rtol=1e-7, atol=1e-8)
            del old, new
            actual_grid_size = (
                outputs[0][0].shape[-1] if scope == "trajectory" else None
            )
            del outputs
            for call in calls:
                for _ in range(3):
                    call()
            measurements = [[], []]
            for repetition in range(args.rounds):
                for index in (0, 1) if repetition % 2 == 0 else (1, 0):
                    measurements[index].append(measure(calls[index], args.device))
            medians = [
                statistics.median(v["wall_ms"] for v in group) for group in measurements
            ]
            case = {
                "scope": scope,
                "batch": batch,
                "waypoint_count": count,
                "dof": dof,
                "grid_size": args.grid_size,
                "actual_grid_size": actual_grid_size,
                "end_speed": end_speed,
                "backward": backward,
                "baseline_ms": medians[0],
                "candidate_ms": medians[1],
                "speedup": medians[0] / medians[1],
                "paired_speedup": statistics.median(
                    old["wall_ms"] / new["wall_ms"] for old, new in zip(*measurements)
                ),
                "measurements": measurements,
            }
            if backward:
                case["saved_references"] = [saved_references(call) for call in calls]
            cases.append(case)
            print(
                f"{args.device} {scope} B={batch} N={count} D={dof} "
                f"end={end_speed} backward={backward}: "
                f"{medians[0]:.3f} -> {medians[1]:.3f} ms",
                flush=True,
            )
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "rounds": args.rounds,
        "seed": args.seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "threads": torch.get_num_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "autograd_multithreading": torch.autograd.is_multithreading_enabled(),
        "sources": [
            {"path": str(path.resolve()), "sha256": module.source_sha256}
            for path, module in zip((args.baseline, args.candidate), modules)
        ],
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    with torch.autograd.set_multithreading_enabled(False):
        main()
