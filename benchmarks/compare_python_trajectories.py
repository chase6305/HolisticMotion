"""Compare two installed packages with alternating, persistent Python workers.

Both installations must support the interpreter selected by --python. Results
include every timing repetition, checksums, and loaded extension identities.
Use the same installation for both arguments to estimate measurement noise.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def serve(installation, dimensions, cpu):
    if cpu != "none":
        os.sched_setaffinity(0, {int(cpu)})

    import gc

    import holistic_motion as hm
    import holistic_motion._holistic_motion as native
    import numpy as np
    from python_trajectory_benchmark import checksum, fixtures

    root = Path(installation).resolve()
    extension = Path(native.__file__).resolve()
    if root not in extension.parents:
        raise RuntimeError(f"expected extension under {root}, loaded {extension}")
    hm.setup_logging("ERROR", handlers=[])
    gc.disable()
    trajectories = list(fixtures(tuple(map(int, dimensions.split(",")))))
    metadata = [
        {
            "group": group,
            "dof": dof,
            "waypoints": count,
            "profile": profile,
            "duration": trajectory.duration,
        }
        for group, dof, count, profile, trajectory in trajectories
    ]
    identity = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "extension": str(extension),
        "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "fixtures": metadata,
    }
    print(json.dumps(identity), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            return
        trajectory = trajectories[request["fixture"]][-1]
        operation, samples = request["operation"], request["samples"]
        if operation == "state":
            query = trajectory.duration * 0.37
            call = partial(trajectory.state, query)
            batch = 256
        else:
            method = (
                trajectory.sample_uniform
                if operation == "uniform"
                else trajectory.constraint_report
            )
            call = partial(method, samples)
            batch = {2: 128, 65: 16, 2001: 2, 20001: 1}[samples]
        for _ in range(3):
            result = call()
        wall, cpu_time = [], []
        for _ in range(request["repeats"]):
            wall_begin = time.perf_counter_ns()
            cpu_begin = time.thread_time_ns()
            for _ in range(batch):
                result = call()
            cpu_time.append((time.thread_time_ns() - cpu_begin) / batch / 1000)
            wall.append((time.perf_counter_ns() - wall_begin) / batch / 1000)
        print(
            json.dumps(
                {
                    "batch": batch,
                    "wall_us": wall,
                    "cpu_us": cpu_time,
                    "checksum": checksum(result),
                }
            ),
            flush=True,
        )


def receive(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("benchmark worker stopped; see its stderr output")
    return json.loads(line)


@contextlib.contextmanager
def worker(args, installation):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(installation), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--worker",
        str(installation),
        ",".join(map(str, args.dofs)),
        str(args.cpu) if args.cpu is not None else "none",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        yield process, receive(process)
    finally:
        try:
            process.communicate('{"stop": true}\n', timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()


def summarize(records, fixtures):
    groups = {}
    for record in records:
        config = record["config"]
        key = (config["fixture"], config["operation"], config["samples"])
        groups.setdefault(key, []).append(record["results"])
    summaries = []
    for (fixture, operation, samples), pairs in sorted(groups.items()):
        checksums = {value["checksum"] for pair in pairs for value in pair.values()}
        if len(checksums) != 1:
            raise RuntimeError("results changed between rounds or installations")
        row = dict(
            fixtures[fixture],
            operation=operation,
            samples=samples,
            checksum=next(iter(checksums)),
        )
        for metric in ("wall_us", "cpu_us"):
            values = {
                mode: [statistics.median(pair[mode][metric]) for pair in pairs]
                for mode in ("baseline", "candidate")
            }
            paired_changes = [
                100 * (candidate / baseline - 1)
                for baseline, candidate in zip(values["baseline"], values["candidate"])
            ]
            row[metric] = {
                mode: {
                    "median": statistics.median(timings),
                    "minimum": min(timings),
                    "maximum": max(timings),
                    "round_medians": timings,
                }
                for mode, timings in values.items()
            }
            row[metric].update(
                paired_changes=paired_changes,
                paired_change_median=statistics.median(paired_changes),
            )
        summaries.append(row)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter compatible with both installations",
    )
    parser.add_argument("--rounds", type=positive_int, default=8)
    parser.add_argument("--repeats", type=positive_int, default=9)
    parser.add_argument("--cpu", type=int, help="pin both workers to this logical CPU")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument(
        "--dofs", type=int, nargs="+", choices=range(1, 33), default=(1, 3, 7, 14, 32)
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("state", "uniform", "report"),
        default=("state", "uniform", "report"),
    )
    args = parser.parse_args()
    for name in ("baseline", "candidate"):
        path = getattr(args, name).resolve()
        if not path.is_dir():
            parser.error(f"{name} installation is not a directory: {path}")
        setattr(args, name, path)
    if args.cpu is not None:
        if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
            parser.error("CPU affinity is not supported on this platform")
        if args.cpu not in os.sched_getaffinity(0):
            parser.error("CPU is outside the current process affinity")
    records = []
    with contextlib.ExitStack() as stack:
        workers = {
            mode: stack.enter_context(worker(args, getattr(args, mode)))
            for mode in ("baseline", "candidate")
        }
        identities = {mode: identity for mode, (_, identity) in workers.items()}
        fixtures = identities["baseline"]["fixtures"]
        if fixtures != identities["candidate"]["fixtures"]:
            raise RuntimeError("fixture metadata differs between installations")
        configs = [
            {
                "fixture": index,
                "operation": operation,
                "samples": samples,
                "repeats": args.repeats,
            }
            for index in range(len(fixtures))
            for operation in dict.fromkeys(args.operations)
            for samples in ((1,) if operation == "state" else (2, 65, 2001, 20001))
        ]
        for round_index in range(args.rounds):
            random.Random(args.seed + round_index).shuffle(configs)
            for index, config in enumerate(configs):
                pair = {}
                order = ("baseline", "candidate")
                if (round_index + index) % 2:
                    order = order[::-1]
                for mode in order:
                    process = workers[mode][0]
                    process.stdin.write(json.dumps(config) + "\n")
                    process.stdin.flush()
                    pair[mode] = receive(process)
                if pair["baseline"]["checksum"] != pair["candidate"]["checksum"]:
                    raise RuntimeError(f"result checksum mismatch for {config}")
                records.append(
                    {
                        "round": round_index,
                        "order": list(order),
                        "config": config.copy(),
                        "results": pair,
                    }
                )
            print(f"Completed round {round_index + 1}/{args.rounds}", file=sys.stderr)
    output = {
        "schema_version": 1,
        "sources": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("python_trajectory_benchmark.py"),
            )
        },
        "identities": identities,
        "settings": {
            "rounds": args.rounds,
            "repeats": args.repeats,
            "cpu": args.cpu,
            "seed": args.seed,
            "dofs": args.dofs,
            "operations": args.operations,
        },
        "results": summarize(records, fixtures),
        "raw_rounds": records,
    }
    json.dump(output, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        serve(*sys.argv[2:])
    else:
        main()
