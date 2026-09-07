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

## C++ diagnostics and benchmarks

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

For performance measurements, enable `HOLISTICMOTION_BUILD_BENCHMARKS=ON` in a
regular Release build with sanitizers disabled, then run
`<build-directory>/sampling_planner_benchmark`. It emits CSV for weighted RRT*
searches with 2, 7, and 14 joints, including periodic joints. Each case uses a
fixed seed and 4000 iterations, with one warmup and five timed repetitions.
The `success` column distinguishes solved paths from exhausted searches; a zero
path length on an unsolved case is not a quality improvement. No external robot
asset or benchmark dependency is required. Timing is observational and is not
a CI pass/fail threshold.

The same option also builds `<build-directory>/fep_batch_benchmark`. This CPU
benchmark uses a synthetic seven-joint chain, nonidentity origins and TCP, seed 7,
and batch sizes 1/32/128/1024. Each case has three warmup calls followed by seven
timed groups of `max(1, 16384 / batch_size)` calls. CSV output reports the median
microseconds per call, maximum scalar-FK error, and a pose checksum. Output
capacity is reused; Python conversion and GPU transfers are outside this timing.

`<build-directory>/numerical_ik_benchmark` measures synthetic 1/3/6/7/14-joint
chains with a nonidentity TCP. It covers an exact seed, a nearby reachable target,
and an unreachable target with a 60-iteration budget. One warmup group precedes
seven timed groups: respectively 16384, 1024, and 128 calls per group for those
three scenarios. CSV includes median microseconds per call, success, FK
evaluation count, pose residual, and joint checksum. Compare success and FK
counts before comparing times; zero residual on a failed solve is a placeholder.
The `damped_ik_workspace_smoke` test separately checks rank-deficient and
rectangular solves against regularized normal equations, while an Eigen guard
forbids heap allocation during the workspace's solve step, including its first use.

`kinematic_model_smoke` covers rejected constructor and model updates, preservation
of home/weight/limit state, invalid frames, fixed-only FK/IK, and FK overflow.
Subclass fault injection also checks runtime guards for empty models, mismatched
DOF, active terminal nodes, and empty FK output. Python regressions exercise
finite values whose FK arithmetic overflows through all three numerical FK and
Jacobian entry points, then verify that an ordinary query still works.

`<build-directory>/robot_model_benchmark` measures URDF loading and explicit
base-to-tip solver construction. It writes temporary synthetic prismatic chains
with 7/64/256/1024 joints, plus a 256-joint trunk with 1024 fixed leaves. After one
warmup group it records seven groups, with 8 model loads or 128 chain constructions
per group. CSV reports median microseconds per call, DOF, and zero-configuration
tip height. URDF generation and FK validation are outside timing; loading includes
file reads, parsing, construction, and destruction. Large cases expose scaling
costs and do not represent typical seven-axis arm latency.

The `robot_model_smoke` regression checks C++ model renaming, first-match duplicate
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
Path smoothing properties independently recompute the full weighted objective
and check objective decrease, exact endpoint preservation, and joint bounds for
mixed periodic/bounded paths with optional quadratic state costs.
Planning checks disable shortcutting and require completion of the fixed iteration
budget; equal wall-clock budgets do not guarantee the same explored samples.
Trajectory sampling properties complement the existing analytic polynomial
extrema tests and do not prove collision or dynamic feasibility between samples.

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
