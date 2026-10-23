# Core concepts

<div class="language-switcher">English · <a href="../zh_CN/concepts.html">简体中文</a></div>

## Explicit model ownership

HolisticMotion does not bundle robot assets. Pass an absolute or application-
resolved URDF path to `Robot`, `CollisionModel`, and retargeting solvers.

The default robot chain is the supported root-to-leaf path with the most actuated
joints. Ties retain the first visited leaf, and trailing fixed transforms remain
part of the tool pose. Paths containing mimic or unsupported joints are excluded.
Explicit base-to-tip construction uses incremental name indexes within each call
for long paths in larger models, while short paths retain direct lookup. C++ edits
to public model names and parent relations are observed on the next call; duplicate
names retain first-match resolution. A cyclic parent traversal terminates with
`nullptr` (`None` in Python) rather than hanging.

## Core and optional components

The core `holistic_motion` library contains robot models, kinematics,
manifolds, null-space path generation, and trajectories. Collision queries live
in the optional `holistic_motion_collision` companion library so ordinary
kinematics users do not acquire Pinocchio and Coal unnecessarily. Python mirrors
this boundary with separate core and collision extensions: ordinary package
imports load the core, and collision types or planning factories load collision
on first use. Core algorithms can share a process with Python Pinocchio while
the Conan collision extension remains unloaded.
Offline geometry preprocessing also loads on first access to `hm.geometry`;
ordinary imports do not initialize NumPy. Numerical array operations load it
when needed.

## Coordinates

- Target poses are finite 4×4 homogeneous transforms.
- Retargeting targets are expressed in the URDF world frame.
- Joint configurations follow Pinocchio model order in the generic retargeting
  result.
- Applications should perform any controller-specific joint reordering at
  their integration boundary.

For built-in floating-point scalars, the C++ `SO3` constructor with
`EulerAnglesType::rota_vec` retains rotations whose vector norm is representable
even when its square overflows or underflows. Autodiff scalars keep their
existing computation path.
Native floating-point SO3 tangent exponentials and Jacobians use unit-axis
formulas for large angles, avoiding overflowing powers and cancellation of
small transverse derivatives. Inverse Jacobians still require representable
entries and remain singular at full turns. Eigen autodiff scalar exponentials
and Jacobians are checked against finite differences.

## Joint names and coordinate mappings

A configuration `q` has length `nq`; velocities, displacement increments and
distance gradients have length `nv`. Native `Robot` stores one scalar per
independent joint, with `nq = nv = model_dof`. Pinocchio stores continuous joints
as `[cos(theta), sin(theta)]`, giving `nq=2, nv=1`. Configuration indices must
not be used as tangent-space gradient indices.

`robot.dof` and `robot.kinematics` describe the default selected chain.
`robot.model_dof`, `robot.nq` and `robot.nv` describe all independent scalar
joints, including other branches. Pass chain-sized positions to that solver;
use names to embed them into a full collision/retargeting configuration.

`Robot`'s scalar representation excludes floating, planar and mimic follower
coordinates. Supply those through the collision/retargeting model's reference
configuration. `joint_coordinates` exposes `name, idx_q, nq, idx_v, nv` snapshots.
`create_joint_position_map(names)` selects scalar joints in the requested order:

```python
names = robot.chain_joint_names("base", "left_tool")
mapping = collision_model.create_joint_position_map(names)
context = collision_model.neutral_configuration()
full_q = mapping.configuration_with_joint_positions(context, chain_q)
chain_q = mapping.joint_positions(full_q)
chain_gradient = mapping.joint_velocities(full_distance_gradient)
```

Mappings copy the reference and preserve unselected arms or a floating base.
Duplicate/unknown names and selected multi-DOF joints are rejected. Continuous
extraction returns principal angles without turn history. Units and joint-axis
conventions must agree across models. Recreate snapshots after topology changes.
Python Pinocchio models use the same semantics through
`holistic_motion.configuration.JointPositionMap.from_model(model, names)`;
retargeting solvers also expose `create_joint_position_map()`.

`Robot`, both native collision models and the Python retargeting/hand solvers
normalize imported revolute, continuous and prismatic URDF axes to unit
directions using stable scaling. Axis magnitude does not
scale joint angles or distances; negative directions retain their sign. Zero
or nonfinite scalar-joint axes are rejected. This corrects the previous native
URDF behavior that multiplied motion by the axis magnitude, and rejects zero
axes before a collision backend can produce a non-rigid placement. Caller
URDF files are preserved.
Imported unit axes also have exactly unit length under the native norm
calculation; this prevents rounding in the axis length from changing the phase
of an extremely large joint angle relative to Pinocchio.
Direct C++ `JointNode` construction retains its explicit axis scale, so manually built
chains must still use axes consistent with their collision/retargeting model.
`LoadURDF()` replaces the model only after construction and validation succeed;
failed parsing or validation preserves the previous model and solver.
Deferred `LoadVisuals()` publishes metadata from the same parsed URDF snapshot
as kinematics. Working-directory changes, source replacement or source removal
cannot select another visual model. It retains only visual metadata while
deferred and reads no mesh assets; publication does not parse the URDF again.
`GetURDFPath()` / `robot.urdf_path` retain the caller's original path.
Native model import rejects reversed position-limit intervals on every bounded
scalar joint, including branches outside the default kinematic chain. Equal
bounds remain valid locked positions; continuous joints have no position bounds.

Mappings transform coordinates without propagating mimic constraints or
checking hardware limits/collisions. Native collision models expose
`validate_configuration(q)` for full dimensions, finite values and normalized
continuous/quaternion blocks, without updating query workspaces. Named-joint
planning factories validate preserved manifold context during construction.

Continuous `joint_lower_limits` / `joint_upper_limits` return an angular
planning interval `[-pi, pi]`, rather than cosine/sine component bounds. Named
collision planning factories mark those joints as periodic. Use a caller-defined
scalar planning space when turn counts must be constrained.

## Initialization and runtime phases

Keep model construction and configuration outside real-time loops. `Robot`,
native collision models, continuous IK trackers, and TOPPRA trajectories
resolve topology and allocate persistent state in their constructors. For
retargeting, call `prepare(mode)` after construction or during a non-real-time
mode transition; `solve()` and `step()` then reuse mode indices and numerical
workspaces. Sampling planners and path optimizers retain joint topology in the
planner object, while each `plan()` or `optimize()` call owns only request-local
search state. Geometry sphere fitting is an offline initialization operation;
the generated sphere model is the runtime artifact.

Stateful retargeting solvers and collision models reuse mutable workspaces and are not
safe for concurrent calls on the same instance. Use one prepared instance per
runtime worker when parallel queries are required.

## Instance ownership and concurrency

| Class or operation | Contract for a shared instance |
|---|---|
| `Robot` tree queries and mapping construction | Read concurrently only while model fields and visual data remain unchanged; synchronize loading and edits |
| `JointPositionMap` | Immutable topology snapshot with independent outputs; shareable |
| Numerical, OPW, UR and FEP read-only solving | Separate caller-owned seeds/outputs and no concurrent model, TCP, frame or limit updates |
| `SRSKinematics` | Geometry and the immutable canonical model are prepared at construction/model updates; read-only queries may share an instance with separate seeds/outputs and unchanged configuration |
| `NullSpacePlanner` | Per-call path and workspace; the retained SRS solver may be shared while its configuration remains unchanged |
| `SRSContinuousTracker` / `FEPContinuousTracker` | Mutable history; construct a tracker per worker; the native solver may be shared for read-only queries |
| `PinocchioRetargetingSolver`, `PinkRetargetingSolver`, `CuroboRetargetingSolver` | Mutable Pinocchio Data and solve history; prepare independently per worker and give callbacks compatible state ownership |
| `DexHandRetargetingSolver` | Mutable hand-model workspace, optimizer and last solution; initialize per worker |
| `RetargetingModeManager` | Mutable mode selection; create one manager per solver worker, sharing only immutable `mode_specs` |
| Both native collision models | Queries update FK/collision workspaces; use one instance per worker |
| `SamplingPlanner` and `PathOptimizer` | Search state belongs to the call, but all callbacks must also support concurrent calls; native collision adapters and sphere-distance caches require separate worker instances |
| Built-in C++ trajectories and NumPy TOPPRA sampling | Concurrent read-only sampling after construction, with unchanged geometry, limits and result arrays; custom path segments define their own contract |
| PyTorch TOPPRA | Follow PyTorch device/stream and autograd rules; use independent graphs for parallel backward passes |
| `SimulationSession` | One state producer; overlapping/reentrant reset/step/close operations are rejected; adapters own device synchronization and buffer lifetimes |

The Python GIL is not a thread-safety guarantee: numerical libraries and device
operations may release it, and callbacks may reenter. Keep `prepare()`, mode
changes and configuration updates outside a worker's active query.
Built-in `solve_ik()`, the SRS/FEP specialized `solve()` and
`solve_configuration()` methods, SRS detailed/user-frame solve variants, and FEP
`forward_batch()` release the GIL during native computation. Inputs, including
branch configurations, are copied first; output conversion runs with the GIL.
This permits Python workers to share an unchanged solver with independent
outputs. It does not permit concurrent solver setters or shared tracker history.
Custom solvers, including subclasses of numerical/OPW/UR solvers, retain the GIL
unless they explicitly override capabilities to declare concurrent queries.
That opt-in also covers Python execution without the GIL.
Planning and path optimization offer the explicit Python keyword
`release_gil=True`; inputs and request configuration are copied before release,
and Python callbacks reacquire the GIL. Their default calling mode remains
compatible with native callbacks that require it.
Built-in numerical, SRS and FEP scalar FK queries compute only the terminal pose;
`GetAllFK()` still returns the complete frame sequence. Custom numerical subclasses
retain virtual `GetAllFK()` dispatch unless they opt into default chain semantics
through the protected `UsesDefaultForwardKinematics()` hook.
Construct collision adapters with independent models for each worker. Copying
a planner or a closure does not clone its captured model or shared distance cache.
Copying a C++ `Robot` also shares its `Joint`/`Link` objects and default solver.
Construct fresh models when workers edit the tree or solver configuration;
construct fresh mode managers from shared immutable specs for retargeting workers.

## Seven-axis kinematics

SRS prepares its canonical frame/offset model during construction and model
updates, without caching query targets or workspace. Changing TCP takes effect
on the next query; copying a solver shares only immutable preparation data.
Detailed reports compute singular values without unused singular vectors, and
joint-limit margins use the intersection of hardware and user limits. Closed-form
branch selection uses the same bounded periodic-coordinate logic as ordinary IK.
Direct C++ joint-axis scales are retained in canonical coordinates and elbow
seeds; imported URDF axes remain normalized. Offset-chain refinement preserves
TCP and the active limit intersection.
The canonical base gauge follows the upper-link side independently of the first
joint axis's sign. Canonical shoulder/elbow/wrist signs and arm angle describe
physical phases, so adding full coordinate periods preserves their values.
SRS and FEP candidate comparisons use each joint's coordinate period
`2*pi/axis.norm()`. Duplicate detection compares physical angular phases; branch
ranking keeps joint-coordinate distances. FEP passes the same periods to CPU
and CUDA scoring and bounds redundancy sampling by physical half turns.

Inspect `solver.capabilities` in Python or `GetCapabilities()` through the C++
`KinematicsBase` interface before selecting optional operations. The snapshot
reports the solver family, default IK algorithm, DOF, Jacobian, redundancy,
closed-form IK, batch FK, compiled CUDA and concurrent-query support. It does
not initialize CUDA or guarantee reachability for a particular target. Custom
C++ solvers inherit conservative defaults and can override the query.
`KinematicsBase` is also a Python alias for the historical `AnalyticKinematics`
base class; the historical name remains available.

SRS and FEP inherit numerical `inverse()` / `GetNearestIK()` entry points.
Their specialized `solve()` methods are explicit alternatives. A seven-R
chain may have SRS type without passing the geometry prerequisites for
closed-form IK; `supports_closed_form_ik` reports that distinction using the
default geometry tolerance. Passing a custom tolerance to `analyze_geometry()`
does not change this capability or the configured solver. FEP currently uses
numerical correction and does not advertise closed-form IK.

`solve_ik(target, seed, selection=IKSelection.NEAREST)` provides a common
owned report for all built-in solvers. Its `status` distinguishes `SUCCESS`,
`INVALID_INPUT` and `NO_SOLUTION`, and `algorithm` records the common entry
point's numerical or analytic algorithm. The seed is unchanged; failures have
empty `solutions` and `distances`. Use `IKSelection.ALL` to request the
candidates available from the supplied seed; this does not guarantee all IK
solutions. Invalid Python types/shapes still raise during argument conversion.
The C++ equivalent is `KinematicsBase::SolveIK()` returning `IKSolveReport`.
The existing `inverse()`, `solve_all()` and specialized solve APIs remain
available with their existing behavior.

```python
import holistic_motion as hm

report = robot.kinematics.solve_ik(target_pose, seed)
if report.success:
    joints = report.solutions[0]
elif report.status == hm.IKSolveStatus.INVALID_INPUT:
    raise ValueError("invalid IK target or seed")
```

`SRSKinematics` is selected for compatible seven-revolute spherical
shoulder/wrist chains. Its null-space projection rejects non-finite inputs and
uses a relative singular-value cutoff so near-singular configurations do not
amplify numerical noise. Projection uses the retained right singular vectors
directly, without forming a pseudoinverse. `NullSpacePlanner` requires its start
to satisfy both hardware and user limits. The magnitude of a nonzero preferred
direction does not change its step, including very small or large finite inputs.
A seed that cannot change the current floating-point coordinates, or an IK
correction back to the current configuration, fails and clears the output path.
Ideal SRS models retain the direct closed-form result;
small URDF link offsets use that result as a branch-preserving seed for a
strict local correction. Equivalent revolute angles are selected directly
from the declared joint-limit interval rather than from a fixed wrap count.

For streaming targets, use the stateful Python tracker instead of selecting
each frame independently:

```python
from holistic_motion.kinematics import SRSContinuousTracker

tracker = SRSContinuousTracker(solver, initial_joints=q0)
result = tracker.solve(target_pose, dt=0.01)
q_command = result.joints
```

The tracker expands periodic angles near the predicted state, applies branch
hysteresis, filters candidates by pose and dynamic limits, and reports
singularity, branch-change, velocity, acceleration, and residual diagnostics.
The native `SRSKinematics` keeps no streaming-target history and can still be
used directly. Geometry measurements are computed at construction and after
successful model updates; `analyze_geometry(tolerance)` only classifies that
read-only snapshot. Concurrent queries with different tolerances do not write
shared state. Configuration updates still require external synchronization.
Use `solve_detailed(target, seed, method)` when the caller needs a non-throwing
status plus each solution's configuration, minimum Jacobian singular value,
per-joint limit margins, near-singularity flag, and limit-hit flag. The existing
`solve()` API remains the compact exception-based interface.
TCP and user-frame changes are explicit runtime configuration. `forward()` and
`solve()` keep using the robot base frame; `forward_user()`, `inverse_user()`,
`solve_user()`, and `solve_detailed_user()` apply the configured user transform
using the SDK-compatible `user_T_base * base_T_tcp` convention. Use the
`tcp`/`user_frame` properties and `clear_tcp()`/`clear_user_frame()` to inspect
or reset that state.

`FEPKinematics` supports offset seven-axis chains and batched FK. CPU is used
for small batches without initializing CUDA; `AUTO` selects CUDA only when a runtime device is available
and the batch is large enough to amortize transfers. Returned FEP IK solutions
are independently checked to 10 µm and 10 µrad after optional high-precision
refinement. Explicit `CUDA` requests fail instead of silently falling back.
The high-precision pass inherits the current TCP, so its target and final check
use the same tool frame as the initial solve.
Refinement stops at the same 10 µm and 10 µrad acceptance limits, avoiding
unnecessary iterations and false failures near singularities.
CPU batch FK prepares joint axes and the fixed terminal/TCP transform once per
call, then accumulates each pose without allocating per-row joint or pose lists.
Model and TCP changes take effect on the next call. Scalar FK uses axis-angle
rotations for oblique revolute axes, and C++ continuous joint nodes contribute
to both FK and the geometric Jacobian. Empty batches return shape `(0, 4, 4)`.
CPU and CUDA batch FK reject non-finite output poses even when the input joints
and TCP are finite. C++ failures clear all poses, including any completed rows.
Streaming offset-arm targets can use `FEPContinuousTracker` with the same
options and diagnostics as the SRS tracker. Both trackers solve the current
branch on every frame, refresh all branches periodically, and immediately
enumerate all candidates if the current branch fails or approaches a
singularity. Malformed candidates, including nested complex values and cyclic
object containers, are filtered before branch metadata queries.
An unusable seeded result, including one rejected by pose or dynamics checks,
also triggers a refresh between periodic refreshes. Each frame enumerates all
branches at most once; a failed refresh leaves tracker history intact.
`candidate_refresh_interval` controls that tradeoff.
Both trackers retain small pose errors with stable position norms and `atan2`
rotation angles. Candidates with non-finite dynamics, FK poses or scores are
discarded before ranking; an unrepresentable prediction leaves history intact.
Initial joints, dynamic limits, weights and result snapshots must contain real
values. Complex inputs raise `TypeError` before conversion.
Scalar tracker weights and thresholds are stored as owned floats; modifying a
caller-provided zero-dimensional array cannot change validated options.
Non-scalar arrays are rejected for these scalar fields.

Numerical IK reuses its thin-SVD, Jacobian, and step buffers within each solve.
An already converged seed skips these buffers. The workspace is local to the
call and introduces no shared mutable solver cache. C++ tolerance, step-size,
and damping setters require finite positive values; invalid values leave the
previous setting intact. `GetIK()` clears solution and distance outputs before
validation, so a failed call does not retain distances from a previous solve.
If a C++ seed is an element of the previous IK output, the numerical, UR and
OPW solvers copy it before clearing that output. Independent seeds avoid this copy.
C++ IK targets use the same rigid-frame validation as TCP and user frames:
finite coefficients and a quaternion squared norm within `1e-9` of one.
Invalid targets fail with empty solution outputs and preserve a valid independent seed.
Native Python FK/IK, SRS/FEP queries and numerical frame setters reject complex
arrays before Eigen conversion, including object arrays containing complex
scalars. Raw target matrices are checked before quaternion construction, so
invalid rotation blocks or homogeneous last rows are rejected. Real lists and
object arrays remain supported; rejected configuration updates preserve state.
SRS/FEP configuration flags and redundancy values also require real scalars;
rejected assignments preserve the configuration.
UR/OPW parameter assignments and constructor limits follow the same real-input
rule. Base and tool transforms must be rigid; rejected assignments retain the
previous parameter value.
Numerical `GetNearestIK()` and Python `inverse()` choose the periodic
representation of the converged branch minimizing weighted joint movement
from the caller's finite seed. A zero-seed convergence retry
still uses the original seed to choose multi-turn representations and distances.
Extreme-scale Jacobians and errors use a normalized decomposition and bounded
mantissa/exponent arithmetic when direct singular values, projections, or squared
regularization terms overflow or lose subnormal precision.
Translation norms retain tiny nonzero errors when their component squares
underflow. Rotation errors use `atan2` to preserve angles whose trace rounds to
three; FK also retains tiny revolute motions when their norm squares vanish.
It uses a scaled norm when squaring a large rotation vector overflows, retaining
finite rotation magnitudes and rejecting angles or translations that themselves
cannot be represented.
Scalar FK, CPU/CUDA batch FK and periodic-coordinate conversion use the same
host-prepared axis scale. This avoids changing a large angle's phase by rounding
the norm of `q * axis` or recomputing the scale on the device. Six-DoF legacy
elbow-up filtering also uses scalar axis-angle motion; malformed prefix geometry
or candidate coordinates clear its solution output instead of being indexed.

C++ solver construction validates joint fields and coordinate layout before
allocating model-sized state. Numerical models reserve their last node for a
fixed tool transform (`UNKNOWN` is also accepted for compatibility); OPW and UR
require at least six coordinate slots, with only fixed/unknown nodes after them.
Invalid models throw `std::invalid_argument`. `NumericalKinematics::SetDOF()`
accepts only the existing model-derived dimension; changing it requires a new
solver. A numerical model with one fixed node has zero DOF: its Jacobian is
`6 x 0`, and IK returns one empty joint vector only when the target already
matches its pose within tolerance.

`SetJointNode()` rejects size changes, unsupported joint types, invalid fields,
or actuated nodes beyond the coordinate slots. Both hardware and user limit
updates must preserve a nonempty intersection. Rejected updates leave the
previous model or limits intact. C++ TCP and user frames require finite
coefficients and a unit quaternion. FK also checks arithmetic overflow from
finite joint inputs: failure clears the pose list and propagates as `ValueError`
through Python `forward()`, `forward_all()`, and `jacobian()`. Full model
validation happens at construction/update time; FK retains lightweight runtime
dimension and arithmetic checks.

`IkRtn::GetLimitsIK()` expands revolute/continuous coordinates inside inclusive
hardware/user limits. It computes the feasible turn interval directly, so work
no longer grows with the number of turns in an out-of-limit seed. Extreme seeds
use their sine/cosine phase when ordinary remainder reduction loses accuracy;
returned wraps must preserve that phase to `1e-10`. Nonperiodic coordinates are
only checked against their interval. An already in-limit configuration with no
other periodic representation skips turn enumeration and its workspace.
For nonunit rotational axes supplied through C++ `JointNode`, the coordinate
period is `2*pi/axis.norm()`; URDF-imported axes are unit directions.
Wrapping checks the physical rotation angle used by FK, including oblique axes.

Expansion is limited to 65536 configurations by default, counted before any
later deduplication. The C++ overload `GetLimitsIK(nodes, max_solutions)` accepts
an explicit budget from 1 through `INT_MAX`. Exceeding the budget or an
unrepresentable turn range returns `false` with an empty result. No partial
solution set is reported. Numerical `solve_all()` inherits the default budget;
large multi-turn ranges can therefore fail even if individual joints have valid
representations. Analytic nearest-branch filtering uses its existing separate
`WrapToLimitsNear()` path. Numerical `GetNearestIK()` and Python `inverse()`
also select one representation per joint directly, so they can succeed when
enumerating all turn combinations would exceed the all-solutions budget.
Nearest wrapping checks nonperiodic coordinates without changing them and
rejects NaN limits or an unrepresentable selected periodic phase.

`IkRtn` result postprocessing uses `ik_joints` as its source of truth and
reconciles the cached `ik_number`. Deduplication keeps lexicographically first
representatives: two solutions merge only when every coordinate differs by less
than the finite nonnegative tolerance; zero tolerance merges exact duplicates.
This deterministic greedy rule does not treat approximate equality as transitive.
Distance sorting preserves the input order of ties and consumes caller-supplied
finite distances. Deduplication, sorting and joint filtering clear `joint_states`;
malformed input clears solution outputs, and failed sorting also clears distances.
Each mutable result belongs to one caller or worker.

## Scope

Null-space path generation and pose retargeting are solver-adjacent algorithms.
The optional collision-backed sampling planner provides a focused set of
joint-space RRT algorithms implemented by HolisticMotion. Controllers,
calibration, implicit model downloads, and mandatory visualization dependencies
remain outside the core library.
