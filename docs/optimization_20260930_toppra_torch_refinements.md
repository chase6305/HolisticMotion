# Differentiable TOPPRA numerical and performance refinements

Baseline: `f2e2b17`. The main changes are in `5cc6c1e` and `ab848ad`; the
additional spline-indexing refinement is included with this report.
This report covers the follow-up two-hour session on 2026-09-30. The optional
PyTorch CPU/CUDA API introduced in the earlier report stays unchanged.

## Numerical changes

- Normalize each path's speed units before constructing LP half-planes. The
  scale accounts for both velocity and acceleration limits and is detached;
  it is a change of variables whose derivative cancels. Timing is converted
  back to physical units after solving. Checks cover time scales from 1e-120
  to 1e120 and acceleration-to-velocity ratios through 1e±240.
- Scale chord vectors before taking norms, avoiding a square overflow when
  the length is representable. Batched spatial-unit tests include 1e300.
- Apply incoming adjoints before the quotient factor in division backward,
  so unused near-singular bounds do not turn zero adjoints into NaNs. Skip
  denominator adjoints and saved quotients when only the numerator needs a
  gradient. First-order derivatives remain piecewise with active choices fixed.
- Keep a projected stop exactly zero when its constraint residual is within
  the existing eight-epsilon roundoff bound. Previously, adding that bound
  created a tiny speed whose square root amplified roundoff in durations and
  gradients. An independent 80-digit half-plane solve confirms the regression's
  zero speed and duration 16.6798165269817138078... . Both backends agree after
  the fix. A grid producing adjacent stops now asks the caller to refine it;
  one coarse-grid case previously returned about 861,713 seconds from roundoff.
- NumPy endpoint and controllability feasibility tolerances scale with squared
  speed rather than absolute units. Infeasible requests remain rejected after
  changing time units, instead of passing an early check and failing later.
- The tensor API rejects complex tensor/array limits, boundaries, and sampling times before a
  real cast can discard their imaginary components.

## Work and memory reductions

Symmetric acceleration strips need only distinct pairs among 3D coefficients,
not a square over 6D+2 half-planes. Pair selection stays outside autograd;
selected bounds are re-evaluated with gradients. Scratch arrays are reused
within a chunk and released before the next chunk. No persistent CUDA cache or
new dependency is introduced.

For a batch ending entirely at rest, controllable lower bounds are identically
zero. Positive affine intercepts and slopes are computed once for the upper
recursion. Mixed/nonzero terminal speeds preserve both bounds and precompute
coefficient masks. Interval unbinding avoids a full zero gradient buffer for
every repeated slice. Constraint construction skips an unused position graph.

At 16 paths, 8 waypoints, 7 DOF and 106 actual grid points, the final profiler
comparison counts 17,179 baseline CUDA launches, 8,744 with the new rest pass,
and 12,444 with a nonzero terminal speed. These are operation-count observations,
not a promise of proportional speedup. The original NumPy solver is never called
by the tensor path.

### Long spline sequences

A subsequent indexing change also unbinds spline interval widths and slopes
once before the Thomas solve. Forward arithmetic stays unchanged; backward
collects interval gradients once instead of allocating full-sized zero buffers
for each slice. A 12-case comparison against `ab848ad`, with 16 alternating
rounds and 8/32/128 waypoints, verifies values and gradients before timing.
At 128 waypoints, 7 DOF, and 255 grid points, forward + sampling + backward is:

| Device | Batch | Before ms | After ms | Reduction |
| --- | --- | --- | --- | --- |
| CPU | 1 | 46.36 | 42.47 | 8.4% |
| CPU | 16 | 78.69 | 74.42 | 5.4% |
| CUDA | 1 | 112.26 | 104.66 | 6.8% |
| CUDA | 16 | 110.52 | 102.59 | 7.2% |

The full 72-case matrices below remain explicitly tied to `ab848ad`, before
this indexing-only refinement. Tests add a 128-waypoint directional derivative
check and 128-DOF independent half-plane comparisons spanning multiple chunks.

## Validation

- 213 combined tensor, NumPy and native-import tests pass locally, including
  128 tensor cases (63 CPU, 65 actual CUDA).
- 1,200 seeded paths per device check same-grid NumPy duration, finite gradients,
  unit metamorphisms, and directional finite differences at two step sizes.
  This campaign found the projected-stop issue; both complete reruns pass.
- 600 boundary cases check feasibility and continuous bounds; 412 are feasible.
  The scaled NumPy rerun additionally checks 1e-8 and 1e8 time-unit changes.
  Degenerate coarse rest grids are explicitly refined before boundary trials.
- The deep Hypothesis profile passes 300 generated NumPy scaling examples.
- Both the CMake install and a freshly built local CPython 3.10 Linux wheel
  pass 209 numerical tests against their installed modules. The combined
  command passes 213 including import checks; two import checks deliberately
  spawn source-tree subprocesses. The wheel contains the exact tensor source
  and declares `torch>=2.2` only in its optional extras. Native import also works.
  This is local packaging validation using the existing Conan toolchain.
- Without Torch, NumPy 1.23.5 passes 85 numerical/import checks, including
  81 installed NumPy tests; the optional tensor test module skips.
- CPU and CUDA Adam examples reduce the same objective from 5.720031 to
  5.193538 in 50 steps, holding endpoints and limits fixed.
- A 501-call CUDA forward/backward check keeps post-release allocated memory
  constant. A subsequent 2,000-step, eight-path CUDA Adam loop reduces loss
  from 162.552972 to 156.281246, retains finite gradients, and checks endpoints
  and continuous motion limits every 100 steps. Post-release allocated memory
  stays at 13,824 bytes after warmup; reserved memory stays at 28 MiB. These
  measurements include the live parameters and optimizer state, not CUDA context
  or other processes' memory.

The GPU is an RTX 5090 D v2 shared with other workloads. PyTorch is 2.13.0+cu130,
Python 3.10.0, and NumPy 2.2.6. CI checks CPU autograd on PyTorch 2.2.2 and current
Torch; local CUDA cases run on real hardware. No robot assets are required.

## Measurement protocol

`benchmarks/compare_toppra_torch.py` compares values and input gradients before
measurement. Each case uses eight seeded waypoints, 101 samples, float64,
one CPU thread, and alternating AB/BA order after warmup. Forward mode uses
`no_grad`; training mode includes forward, sampling, a scalar loss, and backward.
Requested grids 31 and 101 yield 36 and 106 actual points for these paths.

CUDA wall timings synchronize completion and also record CUDA events. Peak
added allocation is the allocator peak above live input storage. Saved-tensor
storage is measured separately, with reference-counted wrappers that track live
storages; addresses can be reused after release. This replaces an early
cumulative-address statistic that was not a reliable retained-memory measure.
The wrappers return detached views and must all be released after the call.

Final matrices compare `f2e2b17` with the frozen implementation in `ab848ad`.
Each device has 72 cases: batches 1/16/64, DOF 1/7/32, requested grids 31/101,
terminal speeds 0/0.005, and forward/training modes. CUDA uses 30 alternating
rounds per case; CPU uses 10 and is pinned to logical CPU 15. Each table row
summarizes 18 cases using the median of candidate/baseline median-time ratios.
A ratio below 1 means faster. Raw repetitions and paired ratios are also saved.

| Device | End speed | Mode | Median time reduction | Per-case time ratio range |
| --- | --- | --- | --- | --- |
| CPU | 0 | Forward | 56.0% | 0.20–0.69 |
| CPU | 0 | Forward + backward | 47.0% | 0.28–0.72 |
| CPU | 0.005 | Forward | 51.1% | 0.21–0.82 |
| CPU | 0.005 | Forward + backward | 38.0% | 0.26–0.90 |
| CUDA | 0 | Forward | 41.7% | 0.30–1.43 |
| CUDA | 0 | Forward + backward | 34.9% | 0.52–0.84 |
| CUDA | 0.005 | Forward | 34.7% | 0.38–1.12 |
| CUDA | 0.005 | Forward + backward | 16.9% | 0.71–1.01 |

These are comparisons against the earlier tensor implementation, not CPU-versus-
GPU comparisons. CPU cases all improve in this matrix. CUDA forward cases have
outliers, including a 42.7% regression; the shared GPU's load changes during the
run. Control and repeat results are given below rather than dropping outliers.

A 16-case A/A control uses the same candidate file on both sides and 40 rounds.
Its median ratio is 1.000, with range 0.998–1.003; saved-storage
counts agree exactly in every training case. The worst full-matrix outlier
(B=1, D=32, requested G=31, zero end speed) repeats at 16.82 → 5.66 ms forward
and 39.18 → 34.31 ms training over 60 rounds. This reversal shows why the
full-matrix GPU percentages are observations under changing shared load, not
precise speedup guarantees. The original outlier remains in the table.

The NumPy stop and boundary-correctness fixes add a small cost: an 18-case,
30-round comparison against its implementation in `5cc6c1e` reports median overhead
of 3.8% for zero terminal speed and 2.0% for terminal speed 0.01. This uses 12
waypoints, 1/7/32 DOF, and identical requested grids of 50/200/800; it is a
separate CPU construction benchmark. The correctness changes are retained
with this measured tradeoff.

### CUDA memory tradeoffs

The following covers every batch/DOF pair at 106 actual grid points, zero
terminal speed, and forward + backward. MiB values use 2^20 bytes.

| Batch | DOF | Peak added allocator memory, old → new (MiB) | Peak live saved storage, old → new (MiB) |
| --- | --- | --- | --- |
| 1 | 1 | 1.29 → 0.74 | 0.16 → 0.15 |
| 1 | 7 | 3.21 → 1.50 | 0.80 → 0.65 |
| 1 | 32 | 57.55 → 15.15 | 3.50 → 2.76 |
| 16 | 1 | 3.48 → 2.95 | 2.52 → 2.36 |
| 16 | 7 | 15.44 → 17.20 | 12.87 → 10.44 |
| 16 | 32 | 76.68 → 57.07 | 55.97 → 44.14 |
| 64 | 1 | 11.63 → 10.42 | 10.08 → 9.42 |
| 64 | 7 | 59.80 → 52.11 | 51.46 → 41.77 |
| 64 | 32 | 261.80 → 227.60 | 223.89 → 176.56 |

The larger pair-selection chunks trade temporary memory for fewer launches.
For 16 paths and 7 DOF, forward-only allocator peak grows from 4.97 to 14.33 MiB;
training peak grows from 15.44 to 17.20 MiB. Live saved storage immediately before
backward is nearly unchanged (10.43 versus 10.44 MiB), while its peak during
construction falls from 12.87 to 10.44 MiB. Fewer saved calls do not imply less
retained storage or a smaller total allocation peak in every case.

Reproduce without the native extension or robot assets:

```bash
git show f2e2b17:python/holistic_motion/trajectory/toppra_torch.py > /tmp/toppra-old.py
git show ab848ad:python/holistic_motion/trajectory/toppra_torch.py > /tmp/toppra-after.py
python benchmarks/compare_toppra_torch.py --baseline /tmp/toppra-old.py --candidate /tmp/toppra-after.py --device cpu --rounds 10 --output /tmp/toppra-cpu.json
python benchmarks/compare_toppra_torch.py --baseline /tmp/toppra-old.py --candidate /tmp/toppra-after.py --device cuda --rounds 30 --output /tmp/toppra-cuda.json
python benchmarks/compare_toppra_torch.py --baseline /tmp/toppra-after.py --device cuda --waypoint-counts 128 --seed 1050 --batches 1 16 --dofs 7 --grid-sizes 255 --end-speeds 0 --rounds 16
HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python python examples/python/trajectory/toppra_differentiable.py --device cuda --optimize-steps 50
```

Local JSON repetitions, memory measurements, validation records, and the
high-precision check are saved under `build/toppra_tensor_refinement/` and are
excluded from commits. The benchmark records SHA-256 hashes of the source
bytes actually executed.

Physical outputs and gradients must remain representable. The backend still
synchronizes validation with the host, propagates sequentially along the grid,
and does not support CUDA Graph capture. CUDA execution does not establish
that GPU is faster than CPU for a given workload.
