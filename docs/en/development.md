# Development and documentation

<div class="language-switcher">English · <a href="../zh_CN/development.html">简体中文</a></div>

## Repository layout

- Public headers: `include/holistic_motion/`
- C++ implementations: `src/`
- Python bindings: `bindings/python/`
- Python package: `python/holistic_motion/`
- C++ tests: `tests/cpp/`
- Python tests: `tests/python/`
- Conan consumer test: `test_package/`

## Native build parallelism

`scripts/build.sh` chooses its native compile job count from CPU affinity,
visible CPU quotas and available memory. It budgets approximately 2 GiB per
compiler process and always allows at least one job. Linux detection considers
host available memory and visible cgroup v1/v2 hard limits, including parent
groups; see the [kernel cgroup documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).
Mount locations and subtree roots are resolved from
[`/proc/self/mountinfo`](https://www.kernel.org/doc/html/latest/filesystems/proc.html#proc-pid-mountinfo-information-about-mounts),
including escaped paths; conventional locations remain a fallback.
Missing resource files fall back to the available CPU information. An explicit
positive `JOBS` takes priority, for example `JOBS=2 ./scripts/build.sh --tests`.
The estimate applies to this script's CMake build; Conan dependency build
parallelism remains controlled by the selected Conan profiles.

## C++ diagnostics and benchmarks

Place explicit class-template instantiations after their member definitions.
Instantiating earlier can omit constructor and method symbols on Clang even
when GCC emits them. The optional Clang 14 core job builds its own Conan
dependency cohort and links the Python bindings across all trajectory dimensions.
Separate optional jobs cover PyPy 3.10 and both CPython 3.13 ABI layouts.

Optional runtimes are tested in separate processes. After a CUDA-enabled native
build with tests and Python bindings, run the hardware check with:

```bash
bash scripts/test-cuda.sh --build-dir build/cmake --python /path/to/python -q
# Also exercise the optional Torch CUDA trajectory backend:
bash scripts/test-cuda.sh --build-dir build/cmake --python /path/to/python --torch -q
```

This check requires a usable GPU and fails if a CUDA backend silently falls back
to CPU. It covers both FEP kernels, batch layouts, and candidate scoring across
thread-block boundaries. CI separately compiles CUDA and tests the no-device
fallback on a CPU runner. The CUDA job also exports the Conan package and links
a real FEP consumer, which exercises transitive runtime linkage even without a GPU.
For Python Pinocchio retargeting, use
`PYTHON_EXECUTABLE=/path/to/python bash scripts/test-retargeting.sh -q`; this
selects toolkit mode and checks that Pinocchio is installed before running.
The optional workflow also builds portable toolkit wheels and exercises the
supported Python Pinocchio versions and the hand backend.
The native core job covers CPython 3.9 and 3.12 and exports Python-enabled Conan
packages whose installed consumer imports the recorded interpreter ABI. The
3.9 job checks NumPy 1.23; the 3.12 job checks sanitizers and the supported
core-plus-Python-Pinocchio runtime. The main collision job uses CPython 3.11.
After its collision tests, that job also installs Python Pinocchio and checks
both import orders in fresh processes with the default native package. Core
FK/retargeting must work together while collision remains unloaded; explicit
collision access must enforce the runtime boundary.

`concurrent_queries_smoke` shares an immutable mapping, a read-only FEP solver,
and planners with stateless callbacks across four workers. Each worker owns its
seeds and outputs; CUDA batches run when a device is available. History-bearing
solvers and mutable collision workspaces retain the per-worker ownership
contract in [Concepts](concepts.md).

`jacobian_benchmark` and `srs_null_space_benchmark` report native query timing
and allocation counts with caller-owned output storage. `planning_context_benchmark`
compares callback snapshots containing copied metadata with snapshots sharing a
prepared joint context. Linux static builds count wrapped `malloc` calls;
other builds label their narrower ordinary C++ allocation counter explicitly.
Counting runs separately from timing.

`benchmarks/collision_worker_benchmark.py` compares full worker construction
with `CollisionModel.clone()`, including fresh workspaces and configured pairs.
`benchmarks/planning_snapshot_benchmark.py` measures complete short planning and
optimization calls, including Python conversion, request snapshots, collision
validation and owned results. Both use temporary synthetic URDFs. Compare
separate coherent SDK installs using `PYTHONPATH`, and retain the raw samples
and output checksums when reporting a performance change.

`benchmarks/joint_mapping_benchmark.py` compares per-call named collision-model
mapping, a reusable native map, and a prepared NumPy map. For example:

```bash
PYTHONPATH=python:build/cmake python benchmarks/joint_mapping_benchmark.py \
  --joints 64 --selected 7 --continuous --iterations 10000 --repeats 7
```

The synthetic URDF is temporary; input parsing, map construction and output
equivalence checks stay outside timing. CSV reports alternating-order median
timings and an output checksum. Public input validation remains in each timed
call; model convenience calls additionally check the complete manifold. This
measures complete coordinate conversion, not collision or FK throughput.

`benchmarks/runtime_import_benchmark.py` compares fresh-process package imports
and the first collision-type access. It alternates package order and reports
median times plus raw samples. On Linux it also records resident memory and
mapped Pinocchio/Coal libraries after the core import:

```bash
python benchmarks/runtime_import_benchmark.py --repeats 11 \
  --package baseline=/path/to/baseline/install \
  --package current=/path/to/current/install --json /tmp/imports.json
```

Use the interpreter that matches both extensions. Interpreter startup and the
worker's standard-library imports are outside the timed interval. This measures
module initialization with the operating system's current file cache, rather
than a cold disk read; timings are observational, with no CI threshold.

`benchmarks/native_query_concurrency.py` compares equal query workloads in one
thread and a persistent Python thread pool. It uses unchanged shared solvers,
14-joint unreachable IK targets, 4096-row CPU FK batches, SRS detailed reports
and FEP branch enumeration. `--null-space-calls 16` adds SRS null-space planning with 512 steps per request.
`--optimization-calls 16` also compares geometry-only
path optimization with `release_gil=False` and `True` on a shared seven-coordinate,
256-waypoint optimizer. Model construction,
pool creation and output equivalence checks are outside timing; each complete
call still includes input conversion and owned outputs:

```bash
PYTHONPATH=python:build/cmake python benchmarks/native_query_concurrency.py \
  --workers 4 --repeats 7 --optimization-calls 16 --json /tmp/native-threads.json
```

`test_native_concurrency.py` checks independent reports/batches and logging
exception recovery. A separate process also verifies that another Python thread
can progress during a long native IK call, without using a speedup threshold.
Planning regressions cover both GIL modes, callback exceptions and reference
cleanup. An external SDK plugin checks the actual GIL state seen by native
callbacks, including the compatible default mode.

ASan, UBSan, and Eigen assertions can be enabled in a separate CPU-only C++
development build. The option defaults to off and requires GCC or Clang.
Instrumentation covers project sources; prebuilt Conan dependencies remain
uninstrumented. This configuration is for testing, not package distribution.
The core CI job runs it after the regular Python build.

```bash
conan install . --output-folder=build-core --build=missing \
  -o '&:with_python=False' -o '&:with_tests=True' \
  -o '&:with_cuda=False' -o '&:with_collision=False'
cmake -S . -B build-sanitizers \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/build-core/build/Release/generators/conan_toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHOLISTICMOTION_BUILD_PYTHON=OFF -DHOLISTICMOTION_BUILD_TESTS=ON \
  -DHOLISTICMOTION_ENABLE_CUDA=OFF -DHOLISTICMOTION_ENABLE_COLLISION=OFF \
  -DHOLISTICMOTION_ENABLE_SANITIZERS=ON
cmake --build build-sanitizers --parallel 2
ctest --test-dir build-sanitizers --output-on-failure --no-tests=error
```

This helper supports a CPU core build. Mixing local ASan-instrumented Eigen
allocation/deallocation with the default prebuilt Pinocchio libraries produced
an allocation-ABI mismatch during collision workspace destruction; CMake rejects
that combination before dependency discovery. Collision sanitizer work requires
a separately instrumented, matching dependency cohort rather than this helper.

For performance measurements, enable `HOLISTICMOTION_BUILD_BENCHMARKS=ON` in a
regular Release build with sanitizers disabled, then run
`<build-directory>/sampling_planner_benchmark`. It emits CSV for weighted RRT*
searches with 2, 7, and 14 joints, including periodic joints. Each case uses a
fixed seed and 4000 iterations, with one warmup and five timed repetitions.
The `success` column distinguishes solved paths from exhausted searches; a zero
path length on an unsolved case is not a quality improvement. No external robot
asset or benchmark dependency is required. Timing is observational and is not
a CI pass/fail threshold.

Pass `[goal_bias [dof continuous]]` to isolate a workload, for example
`sampling_planner_benchmark 0.05 7 1`. Explicit dimensions range from 2 to 32;
`continuous=1` makes the first joint periodic. CSV includes a path checksum to
compare seeded results before timing. Indexed queries cache bounded-coordinate
boxes and use conservative roundoff margins to retain boundary ties.
Distance queries cache one byte per joint for topology checks, preserving the
weighted sum order while avoiding packed-bit decoding in candidate scans.

The same option also builds `<build-directory>/fep_batch_benchmark`. Its default CPU
benchmark uses a synthetic seven-joint chain, nonidentity origins and TCP, seed 7,
and batch sizes 1/32/128/1024. Each case has three warmup calls followed by seven
timed groups of `max(1, 16384 / batch_size)` calls. CSV output reports the median
microseconds per call, maximum scalar-FK error, and a pose checksum. Output
capacity is reused; Python conversion is outside this timing. Pass `CUDA` or
`AUTO` to select another backend, with 100 calls per timed group. Explicit CUDA
requires an available device and includes host packing, device allocation,
transfers, the kernel and output unpacking.

`fep_selection_benchmark [CPU|CUDA]` measures complete seven-coordinate candidate
scoring against a CPU reference, with counts 1/32/129/1024 and seed 82. It uses
three warmups and seven groups; CUDA groups contain 100 calls and CPU groups
contain `max(1, 16384 / count)`. CSV includes maximum score error and a checksum.
The two CUDA operations use one request-local device allocation each. Model
constants for batch FK share one upload; buffers remain independent between
queries. The Linux `cuda_workspace_smoke` test injects allocation and transfer
failures, including output download after kernel launch, and checks cleanup and
successful retries. Without a GPU it still checks size rejection and allocation
failure handling.
Candidate scoring also checks nonunit coordinate periods against an independent
trigonometric reference on CPU and CUDA. The seed and seven periods share one
upload, retaining three transfers and one allocation per scoring request.

`<build-directory>/forward_kinematics_benchmark` measures scalar terminal FK on
synthetic 0/1/3/7/14/64-coordinate mixed chains and seven-axis SRS/FEP solvers.
Origins and TCP are nonidentity, and axes include nonunit scales. After checking
each state against the full frame trace, it runs seven groups of 65536 calls and
reports median nanoseconds, ordinary C++ `operator new` calls per query, and a
translation checksum. The allocation counter excludes aligned allocations and
direct `malloc` calls. Built-in numerical, SRS and FEP terminal queries traverse
the chain without constructing a frame list; unknown numerical subclasses retain
virtual `GetAllFK()` dispatch. `forward_kinematics_smoke` also compares against an
independent matrix chain and checks invalid inputs and custom dispatch.

`joint_space_snapshot_benchmark` measures immutable-space copies, direct plans
and two-waypoint optimization for 2/7/64/256 coordinates with long joint names.
Seven groups contain 32768 calls each; allocation counting uses a separate
128-call pass. CSV records nanoseconds, allocations, counter kind and a checksum.
Linux GCC/Clang builds wrap linked `malloc` calls, including Eigen allocations;
other platforms count ordinary C++ `new` only. Snapshot copies retain immutable
data without allocation. `joint_space_snapshot_smoke` checks copy/move validity,
builder isolation, retained results and concurrent snapshot construction.

`<build-directory>/numerical_ik_benchmark` measures synthetic 1/3/6/7/14-joint
chains with a nonidentity TCP. It covers an exact seed, a nearby reachable target,
and an unreachable target with a 60-iteration budget. One warmup group precedes
seven timed groups: respectively 16384, 1024, and 128 calls per group for those
three scenarios. CSV includes median microseconds per call, success, FK
evaluation count, pose residual, and joint checksum. Compare success and FK
counts before comparing times; zero residual on a failed solve is a placeholder.
The `blocked_by_limits` case uses a one-axis prismatic joint and identity TCP,
with 1024 calls per group. Numerical IK stops when clamped coordinates are
exactly unchanged; tiny representable steps continue normally.
The `damped_ik_workspace_smoke` test separately checks rank-deficient and
rectangular solves against regularized normal equations, while an Eigen guard
forbids heap allocation during the workspace's solve step, including its first use.

`kinematic_model_smoke` covers rejected constructor and model updates, preservation
of home/weight/limit state, invalid frames, fixed-only FK/IK, and FK overflow.
Subclass fault injection also checks runtime guards for empty models, mismatched
DOF, active terminal nodes, and empty FK output. Python regressions exercise
finite values whose FK arithmetic overflows through all three numerical FK and
Jacobian entry points, then verify that an ordinary query still works.

`concurrent_queries_smoke` starts four C++ workers together. They share a
read-only FEP model, a named map with distinct configuration/tangent layouts,
and generic sampling/path optimizers, while owning their seeds and outputs.
FK is compared with an independent matrix chain, IK round trips are checked,
and concurrent planning results must match the sequential seeded reference.
A CUDA-enabled build with an available device also checks concurrent batch FK;
otherwise it exercises CPU batches. It runs under the CPU sanitizer build too.

`<build-directory>/robot_model_benchmark` measures URDF loading and explicit
base-to-tip solver construction. It writes temporary synthetic prismatic chains
with 7/64/256/1024 joints, plus a 256-joint trunk with 1024 fixed leaves. After one
warmup group it records seven groups, with 8 model loads or 128 chain constructions
per group. CSV reports median microseconds per call, DOF, and zero-configuration
tip height. URDF generation and FK validation are outside timing; loading includes
file reads, parsing, construction, and destruction. Large cases expose scaling
costs and do not represent typical seven-axis arm latency.

The `robot_model_smoke` regression checks imported unit-axis semantics, preserved
model/solver state after failed reloads, C++ model renaming, first-match duplicate
name resolution, and termination after a parent edit introduces a cycle. Python
tests cover default-chain selection across supported, mimic, and planar branches.
`compiler_check_smoke` checks compiler-family and version comparisons for GCC,
Clang, Apple Clang, and MSVC using simulated CMake values; it does not replace
building on each platform. Profile setup is described in [Installation](installation.md).

`<build-directory>/path_optimizer_benchmark` compares synthetic paths with
2/7/14 joints and 32/256 waypoints. Cases use bounded joints or a continuous first
joint crossing the angle seam, with and without a lightweight C++ validator.
Every case must complete 12 sweeps and accept updates. After one warmup group,
seven timed groups contain 32 calls for short paths or 4 calls for long paths.
CSV includes median microseconds, iteration/update/validation counts, objectives,
final length, and a path checksum. Compare all result columns before timing;
this benchmark excludes Python callback overhead and native mesh collision costs.
Four optional positional arguments select one case, for example
`path_optimizer_benchmark 7 256 1 0`, to alternate versions within each case.
Append `--finite-cost` or `--analytic-cost` to include a quadratic state cost
with numerical or analytic gradients. CSV also records cost mode and callback
counts so reused workspaces can be compared without changing the amount of work.

`benchmarks/continuous_tracker_benchmark.py` measures complete Python tracking
steps with 1 or 8 candidates using fixed-transform/Jacobian fixtures. It includes
NumPy SVD and continuity selection but excludes native FK/IK work. Run with
`HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python`; use `--baseline` to compare an
earlier `srs_tracker.py`. Candidate counts and result checks accompany timings.
`benchmarks/sphere_fit_benchmark.py --selection-workloads` covers small, large,
and duplicate-heavy candidate sets in both fitting modes. Its existing
`--baseline` option checks output equivalence and measures allocations separately.

`benchmarks/urdf_geometry_benchmark.py` measures complete URDF loading with
1/32/128 links containing unique or repeated box, cylinder, and sphere geometry.
Use the same pure-Python environment and `--baseline /path/to/old/urdf_sphere_fit.py`
for paired timings with exact vertex/face checks. Input generation is excluded.

`benchmarks/collision_reconfiguration_benchmark.py` measures public collision-pair
resets, group changes, and obstacle add/remove cycles on synthetic 7/32/128-joint
models with two sphere geometries. It warms each operation three times, then
reports seven groups of 100 calls, including the complete Python/native call.
Pair counts and physical distances are checked after timing. This measures
geometry workspace updates, not end-to-end planning throughput; the immutable
kinematic model's `pinocchio::Data` remains reusable across those edits.

`path_geometry_workspace_smoke` checks local objective changes against full path
objectives and analytic gradients against central finite differences, including
periodic seams, reverse sweeps, multiple trials, and coincident waypoints. Its
Eigen allocation guard covers the workspace's first and subsequent evaluations.

`<build-directory>/ik_limit_benchmark` measures limit expansion for 1/7/14
revolute joints. Narrow ranges retain one solution; wide ranges produce 2, 128,
and 16384 configurations. Each case warms one group and records seven groups,
with 8192 calls for narrow ranges, 1024 for wide 1/7-joint cases, and 8 for the
wide 14-joint case. CSV contains median microseconds, result count, and checksum.
Input/result preparation is included, while the independent phase/range checks
run outside timing. `ik_limit_smoke` has a ten-second CTest timeout and covers
extreme seeds, inclusive endpoints, independent turn enumeration, mixed-radix
output order, budgets, and empty results after failure.

`<build-directory>/path_construction_benchmark` measures native fifth-degree
Bezier path construction for 2/7 dimensions and 32/512/4096 synthetic waypoints,
with zero blend tolerance and warning-level logging. CSV contains the median
of 30 measured calls after one warmup, plus a path-length checksum. It isolates
path construction and disabled debug formatting, excluding time parameterization,
Python bindings, and robot assets. Compare versions under the same build and
machine conditions; timing is not a CI threshold.

`<build-directory>/trajectory_construction_benchmark` covers 2/7/20/32 dimensions,
4/64 waypoints, both timing profiles, and paths with and without blends.
`trajectory_query_benchmark` compares separate position/derivative queries and
combined state queries at the same dimensions and waypoint counts. Both emit CSV with timing,
duration, and checksums; use paired runs on the same machine and build settings.

`trajectory_cap_benchmark` isolates preliminary curve speed-cap sampling and
compares it with complete trapezoidal construction and a subsequent composed-limit
check. The repeated check must preserve the constructed duration. It covers 2/7/20/32 dimensions,
64 ordinary-scale waypoints and 4 waypoints scaled by `1e4`, with geometry, limits,
and blend tolerance scaled together. Two additional SE3 workloads keep rotation
amplitudes fixed while scaling translation, blend tolerance, and all six limits;
this defines a workload rather than a pose unit conversion. Each CSV row identifies
the group and includes median `cap_us`, `limit_us`, and `construction_us`,
cap checksum, and duration from 21 measured calls after one warmup. Timing ratios
are diagnostic: the operations are measured separately, without instrumentation
inside the constructor. Geometry creation is excluded.

`trajectory_report_benchmark` measures native Double-S constraint reports for 2/7/20/32
dimensions, 4/64 waypoints, and 2/2001/20001 requested uniform samples. Reports
also inspect phase endpoints. Its 24 workloads emit median microseconds from
21 measured calls after one warmup, plus checksums. It excludes construction
and Python binding overhead; timing is not a CI threshold.

`benchmarks/python_trajectory_benchmark.py` measures complete Python calls,
including result assignment and release of the previous result. By default it
emits 110 state/uniform-sampling configurations. Use `--operations report` for
88 constraint-report configurations, or combine `state uniform report`.
`--dofs 1 4 32` selects joint dimensions; Cartesian fixtures remain included.
`--repeats` controls repetitions within each configuration. Report checksums
include all diagnostic fields and array shape, dtype, strides, ownership and
writability. For example:

```bash
PYTHONPATH=build/install python benchmarks/python_trajectory_benchmark.py \
  --operations report --dofs 1 4 32 --repeats 9
```

`benchmarks/compare_python_trajectories.py` compares two installation trees in
separate persistent workers. It alternates each paired call, shuffles workload
order, and verifies all result checksums. JSON output includes wall/thread-CPU
measurements, every repetition, and the loaded extension paths and hashes.
The default covers 198 state, uniform-sampling and report configurations.
Use the same installation for both arguments to estimate host noise; persistent
workers reduce process-startup gaps but do not eliminate shared-host contention.
Negative paired percentage changes mean less elapsed time for the candidate.

```bash
python benchmarks/compare_python_trajectories.py \
  --baseline /tmp/baseline-install --candidate build/install \
  --python /path/to/compatible/python --rounds 8 > comparison.json
```

`--cpu` optionally pins both workers to one allowed logical CPU on platforms
supporting affinity. `--dofs`, `--operations`, and `--repeats` narrow the same
fixtures used by the single-installation benchmark. Both installations must be
compatible with the selected interpreter. Timings include result assignment,
previous-result release and timer overhead; checksum work stays outside timing.

`benchmarks/trajectory_audit.py` generates native trajectories with an explicit
seed and checks finite samples, endpoints, and derivative limits at uniform
times plus every breakpoint. For example, run
`PYTHONPATH=build/install python benchmarks/trajectory_audit.py --cases 3000`.
JSON reports rejected inputs separately from invariant failures and includes
the first ten failures. Replay one with the same `--seed` and `--case-index`.
Use `--distribution stress` for unequal legs, near-collinear points, near
reversals, and wider anisotropic limits up to 32 dimensions. Each shape covers
both profiles. Include the same distribution when replaying a case.
Any rejection or invariant failure returns status 1; construction success alone
is insufficient.
This diagnostic does not prove continuous collision freedom or derivative
limits between samples, and its elapsed time is not an isolated benchmark.

## Generated numerical regression tests

The optional `test` extra adds pytest and Hypothesis. To test an existing Conan
install without triggering another native package build, install only the test
dependencies into the Python environment used by that install:

```bash
python -m pip install 'pytest>=7' 'hypothesis>=6.112,<7'
PYTHONPATH=build/install python -m pytest tests/python -q
HYPOTHESIS_PROFILE=deep PYTHONPATH=build/install \
  python -m pytest tests/python -m property --hypothesis-show-statistics
```

The default `ci` profile runs 40 generated examples per property with deterministic
generation. The `deep` profile runs 300 and retains Hypothesis's local example
database under the ignored `.hypothesis/` directory. Convert relevant failures
into explicit examples or ordinary regression tests before committing a fix.
Without Hypothesis, only the generated-test module is skipped; CI installs it.
Use an up-to-date Python patch release: Python 3.10.0 has a
[slotted dataclass initialization defect](https://github.com/python/cpython/issues/88815)
that prevents current Hypothesis versions from running.

Properties cover whole-turn equivalence in weighted joint spaces, mixed-joint
FK/IK and geometric Jacobians under base transforms, TOPPRA spatial/time scaling,
and preservation of the best RRT* path as its iteration budget increases.
Named-coordinate properties generate branched mixed-joint URDFs, shuffle joint
selection, preserve a floating-base reference, and compare native chain point
Jacobians with full-model sphere-distance tangent gradients.
Path smoothing properties independently recompute the full weighted objective
and check objective decrease, exact endpoint preservation, and joint bounds for
mixed periodic/bounded paths with optional quadratic state costs.
Planning checks disable shortcutting and require completion of the fixed iteration
budget; equal wall-clock budgets do not guarantee the same explored samples.
Trajectory sampling properties complement the existing analytic polynomial
extrema tests and do not prove collision or dynamic feasibility between samples.

## Viser scene environment

Repository Viser applications call `configure_scene(server.scene)` from
`holistic_motion.visualization.viser`. It establishes Z-up camera controls, a
blurred studio background, environment/fill/key lighting, shadows, and a solid
metric floor grid. Pass `ground_width`, `ground_height`, and `ground_z` for the
scene scale and `ground_center_xy` for an offset scene. Mesh inspection tools
can call `configure_scene_from_bounds()` to center and place the floor just
below finite bounds. Grid cells automatically follow a decimal 1-2-5 scale;
pass `grid_cell_size` only when a scene needs a fixed spacing. The helper uses
richer Viser APIs when available, reduces optional environment and light
parameters across transitional releases, and preserves floor position whenever
the grid API supports it. Bounds whose span or padded floor cannot be
represented by finite coordinates are rejected before scene creation. Keep
application geometry outside the reserved `/environment` scene subtree.

## Documentation build

```bash
python -m pip install '.[docs]'
./scripts/docs.sh
```

The build treats Sphinx warnings and invalid API references as errors. Generated
files remain under `docs/_build/` and must not be committed.

When changing public behavior:

1. Update the relevant public header or Python docstring.
2. Add or update a focused test.
3. Update both English and Chinese guide pages.
4. Run `./scripts/docs.sh` and the affected test suite.
