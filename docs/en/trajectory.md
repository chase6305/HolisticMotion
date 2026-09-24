# Trajectory timing

<div class="language-switcher">English · <a href="../zh_CN/trajectory.html">简体中文</a></div>

HolisticMotion provides geometric path generation and time parametrization as
separate operations. `ToppraTrajectory` retimes an existing joint waypoint
path while preserving its geometry.

```python
from holistic_motion.trajectory import ToppraTrajectory

trajectory = ToppraTrajectory(
    [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
    max_velocity=[1.0, 0.8],
    max_acceleration=[2.0, 1.5],
)
times, position, velocity, acceleration = trajectory.sample_uniform(200)
```

Waypoint, limit, and timing-result arrays are immutable owned snapshots.
`sample()` and `sample_uniform()` return new arrays that applications may
modify freely without changing the trajectory object.

The implementation uses squared path velocity as its state, propagates both
bounds of controllable intervals backward, then selects the largest reachable
speed at each forward step. It has
no runtime dependency on the upstream TOPPRA package or an external LP/QP
solver.

`start_path_velocity` and `end_path_velocity` specify endpoint speeds of the
normalized chord-length parameter, both zero by default. Feasible results
preserve these values. Requests that are infeasible under the current grid's
constraints raise `ValueError`; no subsequent global time scaling changes the
requested endpoint speeds.

Run the example:

```bash
./scripts/run.sh python3 examples/python/trajectory/toppra_retiming.py
./scripts/run.sh python3 examples/python/trajectory/toppra_retiming.py --plot
```

For an interactive Viser view of the joint path, playback cursor, velocity and
acceleration traces, and live constraint utilization:

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/visualization/toppra_viser.py --autoplay --loop
```

To animate the TOPPRA result on the full Marvin URDF, including both arm paths
and all visual meshes:

```bash
./scripts/run.sh python3 \
  examples/python/visualization/toppra_robot_viser.py \
  --urdf /path/to/robot.urdf --autoplay --loop
```

Trajectory charts are separated into left- and right-arm tabs. Their compact
`J1`–`J7` legends use consistent colors and include a mapping table to the full
URDF joint names. Click a legend entry to isolate one curve.

Pass either `--asset-root` with a `--profile`, or an explicit `--urdf` path.
The compatibility spellings `--urdf-path` and `--urdf_path` are also accepted.
When launched directly from a built source checkout, the example automatically
enters `scripts/run.sh`.

The path interpolator is a joint-space natural cubic spline. Each interval uses
analytic extrema of the path derivative to bound velocity, and Bernstein
coefficients of the quadratic acceleration polynomial to bound acceleration.
These bounds cover the whole interval and both sides of each knot. They are
conservative and do not guarantee the globally shortest continuous trajectory;
increasing `grid_size` can reduce that conservatism. Custom grid endpoints within
`1e-12` of 0 and 1 are canonicalized to those exact values without modifying the
input; interior gridpoints must remain inside the path domain. `ToppraResult` also checks
that its speed, acceleration, and time arrays describe consistent motion.
Near a stop, its dynamics residual is compared against the scale of all terms
in the interval equation, so changing time units does not turn cancellation
roundoff into a spurious validation failure.

Native Bezier paths retain a reversing waypoint as a sharp linear join, even
with a positive blend tolerance. Double-S and trapezoidal timing stop there
before reversing direction. Quadratic blends use each corner's own control
point and normalized directions, so their tolerance scales with length units.
For fifth-degree Rn paths, a near reversal whose symmetric blend would be no
longer than the existing `1e-5` segment threshold also becomes a stopped linear
join. This preserves the waypoint instead of failing the whole trajectory.
Full waypoint debug strings are formatted only when debug logging is enabled.

The C++ `PathSegBezierCurve5th` segment uses the full tangent metric consistently
for its curve-length coefficients, including rotation for SE3 paths. The
Cartesian flag does not truncate tangent vectors to three entries. One- and
two-dimensional Rn segments therefore remain valid with that flag, and a straight
pure-rotation segment has the same path-length convention as a linear segment.

Fifth-degree Bezier tangent, curvature, and torsion queries evaluate Bernstein
basis functions on successive control-point differences. This avoids the
cancellation introduced by expanded power-basis coefficients, especially after
division by a short segment's squared or cubed length. Exactly uniform control
increments therefore retain zero second and third derivatives without applying
a derivative cutoff, preventing artificial acceleration/jerk speed restrictions
on straight segments. Input/control-point rounding remains subject to ordinary
floating-point precision.

For valid segments with very large lengths, quadratic curvature and fifth-degree
third-derivative queries divide by the length in stages if its square or cube
overflows. This preserves representable nonzero derivatives that would otherwise
be lost through division by infinity. Ordinary lengths retain the existing
normalization calculation; constructor validity checks still apply.

Direct C++ queries on native linear and Bezier segments follow the path query
contract: non-finite parameters throw `std::invalid_argument`, and finite queries
on invalid segments throw `std::logic_error`. Valid segments still clamp finite
out-of-range parameters. Segment validity requires a finite start and end parameter
and a finite non-negative length. Zero-length quadratic Beziers are invalid;
zero-length linear segments remain valid constant configurations. Fifth-degree
segments reject non-finite endpoint derivative norms and overflowing control points.
Check `IsValid()` after construction before using a segment.

The fifth-degree segment's length solver avoids subtracting nearly equal terms
in the quadratic formula. It computes one root with a sign-aware numerator and
recovers the other from the product of the roots, preserving accuracy near a
linear equation. The existing near-linear threshold and larger-root selection
remain unchanged; non-positive or non-finite lengths are still rejected.

For native Double-S and trapezoidal trajectories, `state(t)` returns position,
velocity, acceleration, and jerk together. It shares one time-spline lookup and
one geometric-segment selection; batch sampling uses the same combined query.
Scalar queries remain available and include the same time scaling.

The C++ `PSpline` uses right-continuous internal knots and clamps finite queries
outside its time range. Knot snapping uses the local timestamp and adjacent
interval lengths, so a long trailing segment does not erase earlier short
segments. Non-finite query times throw `std::invalid_argument`. `PushBack`
returns false without changing the spline when the accumulated timestamp would
overflow or fail to advance in floating-point arithmetic.

Double-S endpoint-speed adjustment uses the minimum transition distance under
the acceleration and jerk bounds, with at most 64 bisection steps when needed.
A small velocity difference alone does not establish feasibility on a short
path. Phase durations are checked before integration; rounding-sized negative
zero-duration phases become zero, while larger negative phases are rejected.
Time-law interpolation also rejects backwards timestamps beyond rounding
tolerance. Corrected endpoint speeds can change the timing of existing paths.

Time-spline conversion retains every positive-duration phase, including phases
shorter than 10 microseconds. Dropping a short acceleration or jerk phase would
change the final state and elapsed time. Large time-scale factors are applied
stepwise to derivatives to preserve representable small values; a duration
update that would overflow is rejected without changing the previous scale.

Native constraint reports form sample times from normalized fractions, avoiding
intermediate overflow for large finite durations. Construction-time limit
sampling also normalizes interval lengths before choosing a sample count, so
subnormal durations do not require a time step that rounds to zero. Samples on
the left of a knot use its local timestamp and preceding interval: an unrelated
long tail cannot create a false velocity or acceleration discontinuity. These
reports summarize discrete samples and do not certify continuous-time bounds.

Constraint reports reject sampled states containing non-finite positions or
derivatives with `std::runtime_error` (`RuntimeError` in Python), so a NaN cannot
be silently omitted from a peak calculation. If acceleration or jerk utilization
overflows during construction-time limit enforcement, its square or cube root
is computed by taking roots before dividing. A finite time-scale factor is not
rejected merely because the unrooted ratio overflows. Finite acceleration and
jerk ratios are reduced to their maxima before taking roots, avoiding repeated
root evaluations at each sample. This preserves the sampling grid, though
floating-point root rounding can change the last bits of the resulting scale.

For paths consisting entirely of linear segments, trajectory queries retain
the geometric segment that owns each time phase. Near a stopped corner, the
path position can round to the corner before the incoming phase has ended;
velocity, acceleration, and jerk still use the incoming segment's direction.
At the time knot, the spline's existing right-continuous rule selects the next
phase. Scalar, combined, and batch queries use this same selection after time
scaling. Blended paths continue to locate geometry by path position, since one
time phase can span several blend segments. Corrected derivative peaks can
reduce the global slowdown previously caused by a mismatched segment direction.

Trapezoidal profile generation also retains every positive acceleration,
constant-speed, and deceleration phase, including phases shorter than the
geometric tolerance. Ramp signs follow the actual endpoint-to-target speed
change. Cruise duration comes from the distance left after both ramps, including
cases where a scalar endpoint speed exceeds the internal speed cap. This keeps
phase integration consistent with reported endpoint positions and velocities;
the subsequent global limit enforcement still applies.

Converting trajectory phases to a time spline validates position, velocity,
acceleration, and jerk in every phase state, including the terminal state and
zero-duration transitions. A nonfinite value invalidates the entire conversion
and returns an empty spline, even after valid earlier phases. Finite zero-duration
transitions retain their existing handling.

At the path level, `PathBase::GetPathSegmentAtS` and the configuration,
tangent, curvature, and torsion queries reject NaN and infinite parameters with
`std::invalid_argument`. Finite out-of-range parameters remain clamped, and
internal boundaries select the following segment. Segment lookup by parameter
or index returns `nullptr` for an invalid path, even if it retains segments;
geometry queries on that path throw `std::logic_error`.

A nonfinite path position produced internally by a trajectory remains an
evaluation error (`std::runtime_error`). Constraint reports propagate that
error, while limit enforcement marks the trajectory invalid and fails.

For a distance too short to reach the requested scalar endpoint speed,
trapezoidal timing uses distance divided by average endpoint speed. This avoids
losing elapsed time when the velocity difference rounds to zero. Short
decelerations retain their requested endpoint speed and use the required
acceleration before global slowing. An unrepresentable end timestamp is
rejected before publishing phase states or changing the requested end speed.

When the peak speed stays below the speed cap, ramp durations use a rationalized
distance formula instead of subtracting nearly equal speeds. Nonzero endpoint
speeds therefore do not erase travel time on a short interval. A peak exactly
at the cap follows the capped calculation, retaining constant-speed travel.
Peak-speed calculation uses a scaled norm when squared intermediates overflow
or become subnormal.

Trapezoidal phases are published only after the complete profile passes finite
state and timestamp checks. If a positive phase cannot advance its timestamp
while changing position or materially changing velocity, it must either be
represented by a valid replacement or fail with an empty result. A phase whose
displacement is within 64 machine epsilons of
the path interval length and whose speed change meets the same relative bound
on its endpoint speeds may be omitted as rounding noise, retaining the current
state. This permits adjacent
segment speed caps to differ by a few ulps without using an absolute tolerance
that could erase real motion in small units. Double-S and trapezoidal step
integration skip the cubic position
term when jerk is zero, avoiding a NaN from zero times an overflowing time cube.

An unrepresentable initial or final trapezoidal ramp can be merged with its
adjacent cruise as a constant-acceleration phase. Its duration follows distance
divided by average endpoint speed, and acceleration uses the actual difference
between the stored timestamps. This carries the velocity change without a jump
at a duplicate timestamp. The replacement must meet the acceleration bound,
reproduce displacement within the existing relative roundoff budget, and change
the planned end time only within timestamp roundoff. Both boundary ramps can
share one replacement. Missing cruise time, nonfinite states, or a replacement
that materially changes timing or displacement still fail transactionally.

Double-S also validates all phase states before publishing a profile or adjusted
endpoint speeds. A phase whose timestamp cannot advance must leave position,
velocity, and acceleration unchanged; otherwise generation fails with an empty
result and preserves the requested speeds. Zero-duration transitions retain the
eight-state layout used by backtracking. At a triangular ramp boundary, an
unrepresentable constant-acceleration plateau may retain only its jerk switch
when its duration is within the propagated rounding bound from the ramp time
and endpoint-speed uncertainty divided by acceleration. Position changes are
bounded by 64 machine epsilons times the path interval length, velocity uses
the same relative bound on its endpoint speeds, and acceleration must remain
unchanged. The stored state then remains exactly unchanged
at the shared timestamp. Moving jerk phases and meaningful plateaus still fail
when their timestamps collapse. A valid profile requiring a reduced
start speed is still available for propagation to preceding segments, while an
empty failed profile stops backtracking immediately.

After backtracking, Double-S aligns the following profile using elapsed phase
times and a validated constant-speed bridge. A zero-length bridge requires no
division by speed, so stationary joins remain finite. Alignment rejects a
positive gap without forward speed, timestamp overflow, or a collapsed phase
that changes position, velocity, or acceleration. All new timestamps are
validated before replacing the originals. Tiny negative position gaps within
integration roundoff are treated as zero instead of creating backward travel.

Nonzero-jerk position integration falls back to mantissa/exponent scaling when
the time cube or its product with jerk overflows or underflows into the subnormal
range or zero. This
retains a representable cubic displacement even when an intermediate is outside
the floating-point range. Ordinary inputs keep the original arithmetic, and
zero jerk still skips the cubic term. A true displacement overflow remains
nonfinite and is rejected by profile validation. This fallback applies to the
cubic position term.

The quadratic contributions to position and velocity also use exponent scaling
when intermediate products leave the normal range. This avoids losing a tiny
acceleration or jerk when it is halved before multiplication by a large time.
Ordinary inputs keep the existing multiplication order. True overflow remains
nonfinite; the linear terms and the addition of integration terms retain their
existing formulas.

Curve-speed sampling validates the parameter interval and sampled tangent,
curvature, and torsion. Invalid intervals or nonfinite derivatives return zero
admissible speed, causing trajectory construction to fail. A fixed sample step
that rounds back to the same path parameter is also rejected, preventing a
stalled loop at large parameter values. Ordinary sampling retains its 0.01 step
(or the full length for shorter segments) and includes the exact endpoint.

Every finite nonzero derivative contributes to the sampled speed bound, even
below the geometric tolerance: a small derivative can still matter for a tighter
joint limit. Curvature and torsion bounds take square or cube roots before
division when the direct limit-to-derivative ratio overflows or underflows.
Ordinary ratios retain the existing arithmetic. These remain sampled bounds;
the subsequent global limit enforcement still applies.

Near-duplicate filtering applies to interior waypoints. The requested final
position is retained even when its last leg is shorter than the geometric
tolerance; positive linear legs keep their normalized path tangent. The Python
constructor also accepts distinct two-point motions below that tolerance;
exactly repeated points still do not define a moving trajectory. Blends
smaller than that tolerance are disabled before trimming either neighboring
line, so suppressing a blend does not leave a positional gap.

For time phases contained in one linear geometric segment, limit enforcement
evaluates the endpoints and any interior zero of scalar acceleration. These
are the possible extrema of the quadratic velocity, linear acceleration, and
constant jerk, avoiding a dense uniform time grid. A monotonicity check also
identifies these phases inside blended paths. Curved phases retain
time-proportional sampling with at least 65 checks per phase, so short curved
phases receive enough local resolution to expose peaks missed by a coarse
global grid. Curved-path checks remain sampled bounds rather than a proof of
continuous constraint satisfaction.

Double-S treats an endpoint speed above a neighboring segment cap by at most
128 machine epsilons (relative to that speed) as the same numerical cap. It
preserves that endpoint instead of backtracking through preceding segments;
the comparison always uses the original cap, so the allowance cannot accumulate
across endpoints. Larger differences still reduce the speed and backtrack,
and global joint-limit enforcement still applies to the composed trajectory.

A reduced Double-S entry speed is propagated to preceding phases even for the
final segment; otherwise the curve-to-line join would have a velocity jump.
Backtracking uses the segment's configured acceleration capacity, including
when the original profile was a pure cruise with zero observed acceleration.
Roundoff-sized triangular acceleration plateaus are removed before integration,
so rebasing them at a later timestamp cannot create state changes at zero time.

For a reproducible numerical audit with dense local sampling and phase-join
continuity checks, run:

```bash
PYTHONPATH=build/install python benchmarks/trajectory_audit.py \
  --seed 20260925 --cases 3000 --samples 2001 \
  --phase-samples 101 --check-continuity
```

The JSON output includes the seed, failing case index, and complete inputs.
These sampled checks are diagnostics, not a proof of continuous feasibility.

Nonzero C++ boundary speeds remain subject to profile feasibility. If the first
segment requires reducing the requested initial speed, construction fails without
attempting to backtrack beyond the beginning. A concave profile whose two endpoint
speeds exceed its scalar cap is also rejected when it lacks room to reach that
cap; the unsupported short-valley case must not publish an incorrect end state.
