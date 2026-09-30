"""Compare tensor TOPPRA values, gradients, timings, and saved tensor storage.

Extract a baseline module with git show, then run for either CPU or CUDA::

    git show f2e2b17:python/holistic_motion/trajectory/toppra_torch.py > /tmp/toppra.py
    python benchmarks/compare_toppra_torch.py --baseline /tmp/toppra.py --device cuda

No native extension or robot assets are required. Torch must be installed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source = path.read_bytes()
    # Explicit CLI paths select executable modules. Execute and hash the same
    # bytes even if a checkout is edited while the benchmark is running.
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    module.source_sha256 = hashlib.sha256(source).hexdigest()
    return module


def run(module, inputs, grid_size, backward, end_speed=0.0):
    context = contextlib.nullcontext() if backward else torch.no_grad()
    with context:
        trajectory = module.TorchToppraTrajectory(
            *inputs, grid_size=grid_size, end_path_velocity=end_speed
        )
        motion = trajectory.sample_uniform(101)
        gradients = ()
        if backward:
            loss = trajectory.duration.sum() + sum(
                weight * value.square().mean()
                for weight, value in zip((0.01, 0.03, 0.001), motion[1:])
            )
            gradients = torch.autograd.grad(loss, inputs)
        result = trajectory.result
        return (
            result.gridpoints,
            result.path_speeds,
            result.path_accelerations,
            result.times,
            result.duration,
            *motion,
            *gradients,
        )


def saved_storage(module, inputs, grid_size, end_speed=0.0):
    storages = {}
    calls = 0
    referenced = 0

    def pack(tensor):
        nonlocal calls, referenced
        calls += 1
        referenced += tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        run(module, inputs, grid_size, True, end_speed)
    return {
        "save_calls": calls,
        "referenced_bytes": referenced,
        "unique_storage_bytes": sum(storages.values()),
    }


def measure(module, inputs, grid_size, backward, device, end_speed=0.0):
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
    outputs = run(module, inputs, grid_size, backward, end_speed)
    if device == "cuda":
        end_event.record()
        end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - start) / 1e6
    result = {"wall_ms": wall_ms}
    if device == "cuda":
        result["event_ms"] = start_event.elapsed_time(end_event)
        result["peak_added_bytes"] = (
            torch.cuda.max_memory_allocated() - allocated_before
        )
    del outputs
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "python/holistic_motion/trajectory/toppra_torch.py",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--dofs", type=int, nargs="+", default=[1, 7, 32])
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[31, 101])
    parser.add_argument("--end-speeds", type=float, nargs="+", default=[0.0, 0.005])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 2 or args.threads < 1:
        parser.error("rounds must be >= 2 and threads >= 1")
    if min(args.batches + args.dofs) < 1 or min(args.grid_sizes) < 2:
        parser.error("batch sizes and DOFs must be positive, grid sizes >= 2")
    if any(not math.isfinite(speed) or speed < 0 for speed in args.end_speeds):
        parser.error("end speeds must be finite and non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(args.threads)
    modules = [
        load_module(path, name)
        for path, name in (
            (args.baseline, "toppra_baseline"),
            (args.candidate, "toppra_candidate"),
        )
    ]
    cases = []
    for batch in args.batches:
        for dof in args.dofs:
            generator = torch.Generator().manual_seed(20260930 + dof)
            points = torch.randn(
                batch, 8, dof, generator=generator, dtype=torch.float64
            )
            inputs = tuple(
                value.to(args.device).requires_grad_()
                for value in (
                    points,
                    torch.linspace(0.8, 1.3, dof, dtype=torch.float64),
                    torch.linspace(1.5, 2.5, dof, dtype=torch.float64),
                )
            )
            for grid_size, end_speed in (
                (grid_size, end_speed)
                for grid_size in args.grid_sizes
                for end_speed in args.end_speeds
            ):
                for backward in (False, True):
                    expected = tuple(
                        value.detach().cpu()
                        for value in run(
                            modules[0], inputs, grid_size, backward, end_speed
                        )
                    )
                    actual = run(modules[1], inputs, grid_size, backward, end_speed)
                    for old, new in zip(expected, actual):
                        torch.testing.assert_close(
                            new.detach().cpu(), old, rtol=1e-7, atol=1e-8
                        )
                    actual_grid_size = expected[0].shape[-1]
                    del expected, actual
                    for module in modules:
                        for _ in range(2):
                            run(module, inputs, grid_size, backward, end_speed)
                    measurements = [[], []]
                    for repetition in range(args.rounds):
                        for index in (0, 1) if repetition % 2 == 0 else (1, 0):
                            measurements[index].append(
                                measure(
                                    modules[index],
                                    inputs,
                                    grid_size,
                                    backward,
                                    args.device,
                                    end_speed,
                                )
                            )
                    medians = [
                        statistics.median(item["wall_ms"] for item in values)
                        for values in measurements
                    ]
                    case = {
                        "batch": batch,
                        "dof": dof,
                        "grid_size": grid_size,
                        "actual_grid_size": actual_grid_size,
                        "end_speed": end_speed,
                        "backward": backward,
                        "baseline_ms": medians[0],
                        "candidate_ms": medians[1],
                        "speedup": medians[0] / medians[1],
                        "measurements": measurements,
                    }
                    if backward:
                        case["saved_tensors"] = [
                            saved_storage(module, inputs, grid_size, end_speed)
                            for module in modules
                        ]
                    cases.append(case)
                    print(
                        f"{args.device} B={batch} D={dof} G={grid_size} end={end_speed} backward={backward}: {medians[0]:.2f} -> {medians[1]:.2f} ms",
                        file=sys.stderr,
                        flush=True,
                    )
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "threads": args.threads,
        "rounds": args.rounds,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "sources": [
            {
                "path": str(path.resolve()),
                "sha256": module.source_sha256,
            }
            for path, module in zip((args.baseline, args.candidate), modules)
        ],
        "cases": cases,
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
