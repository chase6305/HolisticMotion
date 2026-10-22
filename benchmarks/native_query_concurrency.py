"""Measure equal native query workloads sequentially and across Python threads.

Uses temporary URDFs and shared, unchanged native solvers. Complete calls include
Python validation/conversion and owned output creation. The IK target is known
to be unreachable, exercising the iteration budget; FK output is checked against
a scalar reference. Pool creation and model preparation are outside timing.
"""

import argparse
import json
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import holistic_motion as hm
import numpy as np


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def write_chain(path, count):
    links = "".join(f'<link name="link{i}"/>' for i in range(count + 1))
    joints = "".join(
        f'<joint name="joint{i}" type="revolute">'
        f'<parent link="link{i}"/><child link="link{i + 1}"/>'
        f'<origin xyz="0.02 -0.01 0.1"/>'
        f'<axis xyz="{("1 0 0", "0 1 0", "0 0 1")[i % 3]}"/>'
        '<limit lower="-2" upper="2" velocity="1" effort="1"/></joint>'
        for i in range(count)
    )
    path.write_text('<robot name="threads">' + links + joints + "</robot>")


def measure(pool, query, validate, workers, repeats):
    validate([query(0)])
    samples = {"sequential": [], "threads": []}
    for repeat in range(repeats):
        order = (
            ("sequential", "threads") if repeat % 2 == 0 else ("threads", "sequential")
        )
        for kind in order:
            started = perf_counter()
            if kind == "sequential":
                results = [query(i) for i in range(workers)]
            else:
                results = list(pool.map(query, range(workers)))
            samples[kind].append((perf_counter() - started) * 1000)
            validate(results)
    medians = {kind: statistics.median(values) for kind, values in samples.items()}
    return {
        "workers": workers,
        "samples_ms": samples,
        "median_ms": medians,
        "speedup": medians["sequential"] / medians["threads"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--ik-calls", type=positive_int, default=128)
    parser.add_argument("--batch-calls", type=positive_int, default=16)
    parser.add_argument("--seven-axis-calls", type=positive_int, default=16)
    parser.add_argument(
        "--optimization-calls",
        type=positive_int,
        help="also compare retained/released GIL on native path optimization",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not hm.runtime_info().native_loaded:
        parser.error("requires the native core")
    with tempfile.TemporaryDirectory(prefix="hm_concurrency_") as folder:
        path = Path(folder) / "model.urdf"
        write_chain(path, 14)
        solver = hm.Robot(str(path)).kinematics
        seed = np.full(14, 0.1)
        target = solver.forward(seed)
        target[:3, 3] += 100.0

        def ik_chunk(_):
            failures = 0
            for _ in range(args.ik_calls):
                report = solver.solve_ik(target, seed)
                failures += report.status == hm.IKSolveStatus.NO_SOLUTION
            return failures

        def validate_ik(results):
            assert all(failures == args.ik_calls for failures in results)

        write_chain(path, 7)
        robot = hm.Robot(str(path))
        srs = robot.kinematics
        fep = robot.create_fep_kinematics("link0", "link7")
        batch = np.full((4096, 7), 0.1, order="F")
        reference = fep.forward(batch[0])

        def batch_chunk(_):
            for _ in range(args.batch_calls):
                poses = fep.forward_batch(batch, hm.FEPBackend.CPU)
            return poses

        def validate_batch(results):
            for poses in results:
                assert poses.shape == (4096, 4, 4)
                np.testing.assert_allclose(
                    poses, np.broadcast_to(reference, poses.shape), atol=1e-12
                )

        seven_seed = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])
        seven_target = srs.forward(seven_seed)

        def srs_chunk(_):
            for _ in range(args.seven_axis_calls):
                report = srs.solve_detailed(
                    seven_target, seven_seed, hm.SRSSolveMethod.ALL_CONFIGURATIONS
                )
                assert report.status == hm.SRSSolveStatus.SUCCESS
            return report.solutions

        def fep_chunk(_):
            for _ in range(args.seven_axis_calls):
                solutions = fep.solve(
                    seven_target, seven_seed, hm.FEPSolveMethod.ALL_CONFIGURATIONS
                )
            return solutions

        def validate_seven(results):
            for solutions in results:
                assert solutions
                for solution in solutions:
                    np.testing.assert_allclose(
                        srs.forward(solution), seven_target, atol=1e-3
                    )

        optimization_rows = []
        if args.optimization_calls is not None:
            optimizer = hm.PathOptimizer(-np.ones(7) * 3.0, np.ones(7) * 3.0)
            path = np.array(
                [
                    -0.8 + 1.6 * row / 255 + 0.3 * np.sin(0.8 * row + np.arange(7))
                    for row in range(256)
                ]
            )
            options = hm.PathOptimizationOptions()
            options.max_iterations = 24
            options.timeout_seconds = 30.0
            options.step_size = 0.2
            reference_optimization = optimizer.optimize(path, options)
            assert reference_optimization.success

            def optimization_chunk(release_gil):
                def query(_):
                    for _ in range(args.optimization_calls):
                        result = optimizer.optimize(
                            path, options, release_gil=release_gil
                        )
                    return result

                return query

            def validate_optimization(results):
                for result in results:
                    assert result.success
                    np.testing.assert_array_equal(
                        result.path, reference_optimization.path
                    )
                    assert (
                        result.statistics.final_objective
                        == reference_optimization.statistics.final_objective
                    )

        with ThreadPoolExecutor(args.workers) as pool:
            ik = measure(pool, ik_chunk, validate_ik, args.workers, args.repeats)
            fk = measure(pool, batch_chunk, validate_batch, args.workers, args.repeats)
            srs_report = measure(
                pool, srs_chunk, validate_seven, args.workers, args.repeats
            )
            fep_solve = measure(
                pool, fep_chunk, validate_seven, args.workers, args.repeats
            )
            if args.optimization_calls is not None:
                for release_gil in (False, True):
                    row = measure(
                        pool,
                        optimization_chunk(release_gil),
                        validate_optimization,
                        args.workers,
                        args.repeats,
                    )
                    row.update(
                        operation="path_optimization",
                        release_gil=release_gil,
                        calls_per_worker=args.optimization_calls,
                        dof=7,
                        waypoints=256,
                        checksum=float(np.asarray(reference_optimization.path).sum()),
                    )
                    optimization_rows.append(row)
        ik.update(
            operation="solve_ik",
            calls_per_worker=args.ik_calls,
            outcome="expected_no_solution",
            dof=14,
        )
        fk.update(
            operation="forward_batch",
            calls_per_worker=args.batch_calls,
            batch_size=4096,
            checksum=float(reference.sum() * 4096),
        )
        srs_report.update(operation="srs_solve_detailed")
        fep_solve.update(operation="fep_solve")
        for result in (srs_report, fep_solve):
            result.update(
                calls_per_worker=args.seven_axis_calls,
                method="ALL_CONFIGURATIONS",
                dof=7,
            )
    rows = [ik, fk, srs_report, fep_solve, *optimization_rows]
    print(json.dumps(rows, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
