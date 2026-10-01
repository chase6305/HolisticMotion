# TOPPRA solver and grid improvements

Reference: EmbodiChain `feat/differentiable-toppra`, commits `49d76866` and
`748006a1`. This change adapts zero-terminal-speed controllability and
segment-aligned grid construction to HolisticMotion's existing natural cubic
path. NumPy remains the only dependency; tensor gradients, signed limits,
batched tensor results, and Warp/CUDA execution are outside this change.

## Behavior

A zero terminal speed guarantees zero is in every controllable interval.
Only upper bounds need propagation, including when the initial speed is
nonzero. The forward solution is checked against every half-plane in a
vectorized pass. Nonzero terminal speeds retain both controllable bounds;
coefficient preparation needed only by the zero-terminal solve is skipped.

Default grids subdivide each spline segment. Near-integer subdivision counts
are stabilized against roundoff. Caller grid points within eight float64 epsilons
of a spline knot are replaced by that knot, while distinct spline knots are
retained. Geometry and interval-wide velocity/acceleration bounds are unchanged;
default grid points and therefore duration can change. Invalid `grid_size < 2`
is now rejected instead of being hidden by the waypoint count.

A concrete regression is the scalar path `[0, 0.1, 0.3, 1]`, velocity limit 1,
acceleration limit 2, and `grid_size=21`. The old grid can produce duplicate
arrival times and reject this valid path. Tests cover both generated and custom
grids, analytic straight-path reachable speeds, nonzero boundaries, short real
spline segments, and changes of time units.

## Performance

Baseline: HolisticMotion `6e56cc7`. Python 3.10.0, NumPy 2.2.6, Linux x86_64,
shared host. Each case uses 12 deterministic waypoints (seed 53), identical
caller grids, and 30 alternating baseline/candidate measurements after warmup.
Grid, speed, acceleration, timing, and 501 sampled states are compared before
timing. These are solver-construction timings, not sampling or GPU timings.

Across nine cases per terminal-speed group, median time reduction is **39.1%**
for zero terminal speed and **12.8%** for terminal speed 0.01. These observations
are hardware/load dependent and do not establish a speedup for all paths.

| DOF | Grid size | Terminal speed | Baseline ms | Candidate ms | Reduction |
| --- | --- | --- | --- | --- | --- |
| 1 | 50 | 0.0 | 2.447 | 1.532 | 37.4% |
| 1 | 50 | 0.01 | 2.410 | 2.140 | 11.2% |
| 1 | 200 | 0.0 | 8.052 | 4.795 | 40.4% |
| 1 | 200 | 0.01 | 7.983 | 7.046 | 11.7% |
| 1 | 800 | 0.0 | 30.975 | 17.778 | 42.6% |
| 1 | 800 | 0.01 | 31.593 | 28.252 | 10.6% |
| 7 | 50 | 0.0 | 2.869 | 1.783 | 37.9% |
| 7 | 50 | 0.01 | 3.170 | 2.764 | 12.8% |
| 7 | 200 | 0.0 | 9.422 | 5.734 | 39.1% |
| 7 | 200 | 0.01 | 9.556 | 8.183 | 14.4% |
| 7 | 800 | 0.0 | 36.483 | 22.885 | 37.3% |
| 7 | 800 | 0.01 | 36.190 | 31.587 | 12.7% |
| 32 | 50 | 0.0 | 6.867 | 4.100 | 40.3% |
| 32 | 50 | 0.01 | 6.762 | 4.942 | 26.9% |
| 32 | 200 | 0.0 | 23.097 | 13.678 | 40.8% |
| 32 | 200 | 0.01 | 23.176 | 16.473 | 28.9% |
| 32 | 800 | 0.0 | 85.291 | 52.805 | 38.1% |
| 32 | 800 | 0.01 | 85.448 | 62.444 | 26.9% |

Reproduce without the native extension or robot assets:

```bash
git show 6e56cc7:python/holistic_motion/trajectory/toppra.py > /tmp/toppra-old.py
python benchmarks/compare_toppra.py --baseline /tmp/toppra-old.py --rounds 30
```

The benchmark prints raw repetitions as JSON. The local run is saved to
`build/toppra_comparison.json` (generated, not committed).

## Validation

- NumPy 2.2.6: 74 tests passed across TOPPRA, numerical properties, and imports,
  using the `deep` Hypothesis profile (300 examples per generated test).
- NumPy 1.23.5: all 65 TOPPRA tests passed.
- Ruff lint/format checks and `git diff --check` passed.
- Strict English and Chinese documentation builds passed via `./scripts/docs.sh`.
- No external robot asset paths are required.
