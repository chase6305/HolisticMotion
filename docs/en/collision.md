# Collision checking

The default build provides `CollisionModel`, backed by Conan-managed
Pinocchio and Coal. It loads collision geometry from a caller-provided URDF
and checks self-collision at one configuration. Python loads its optional
extension on first access to a collision type or collision planning factory;
ordinary `import holistic_motion` loads the core alone.
`runtime_info().collision_loaded` reports the actual loaded state.

Kinematics and collision geometry are constructed from the same owned URDF text
snapshot. The constructor reads the source once, so replacing the file during
construction cannot combine two model revisions. Mesh files are still resolved
through the caller's package roots.

Queries reject non-finite world geometry placements with `OverflowError`, even
when every input joint value is finite. Distance queries also reject results
that Coal cannot represent, including non-finite nearest points despite a finite
signed distance, instead of returning an unrelated pair or reporting an indexing
error. A failed query leaves the instance available for the next
valid configuration; clearance settings are restored when a margin query exits.
`evaluate()` and `collisions()` share one FK/geometry update between collision
and distance calculations.

World-fixed box obstacles can be added at runtime with `add_box_obstacle` and
removed with `remove_obstacle`. Include each obstacle name in a collision group
to select the robot links that should be checked against it.

```bash
./scripts/build.sh
python3 examples/python/collision/basic_query.py \
  --urdf /absolute/path/to/robot.urdf \
  -q 0 0 0 0 0 0
```

The number of values after `-q` must equal the reported Pinocchio `nq`.
The default is the model's neutral configuration, including `[1, 0]` for
continuous joints and an identity quaternion for floating joints. Explicit
configurations must contain normalized continuous/quaternion blocks.

For a sampled scalar joint-space segment, provide joint names in endpoint order:

```bash
python3 examples/python/collision/path_scan.py --urdf /path/to/robot.urdf \
  --joint-names shoulder_slide wrist_turn \
  --q-start=0.2,3.0 --q-end=0.4,3.3 --samples 101
```

Named mode linearly interpolates the supplied scalar positions and embeds them
in a neutral full configuration, or in the full `nq` reference supplied with
`--context=...`. Angles can be unwrapped to select the intended turn direction;
the example does not choose a shortest angular path automatically. Without
`--joint-names`, endpoints retain the full scalar-configuration convention;
non-Euclidean models require named mode so cosine/sine and quaternion components
are not interpolated as scalar joints. Sampling does not certify collision
freedom between samples.

If the URDF uses `package://` mesh URLs, pass one or more package roots:

```bash
python3 examples/python/collision/basic_query.py \
  --urdf /path/to/robot.urdf \
  --package-dir /path/to/ros/workspace/src \
  -q 0 0 0 0 0 0
```

Adjacent-link pairs are excluded by default. Use `--include-adjacent` when
they should participate in the query. Run `--help` for all options.

Python configurations, geometry, and distance thresholds must be real. Complex
inputs raise `TypeError` before queries or geometry updates; rejected sphere
assignments retain their previous values. Real integer arrays and strided views
remain supported.

## Collision-pair groups

For a dual-arm robot, define each arm as a link group and enable only the
left-arm/right-arm pair. Every collision geometry attached to a listed link
is treated as part of that arm:

```bash
python3 examples/python/collision/basic_query.py --urdf /path/to/dual_arm.urdf \
  --group left_arm=left_shoulder,left_upper_arm,left_forearm,left_hand \
  --group right_arm=right_shoulder,right_upper_arm,right_forearm,right_hand \
  --check-groups left_arm:right_arm \
  -q 0 0 0 0 0 0 0 0 0 0 0 0
```

This checks all left-arm geometry against all right-arm geometry, but does not
check links within either arm. Add `--check-groups left_arm:left_arm` or
`right_arm:right_arm` to enable self-collision inside an arm.
Same-link and adjacent-link pairs are automatically excluded from within-group
self-collision checks.

The equivalent API is:

```python
model.set_collision_groups(
    {"left_arm": left_links, "right_arm": right_links},
    [("left_arm", "right_arm")],
)
for pair in model.collision_pairs:
    print(pair.first_link, pair.second_link)
```

`set_collision_groups()` replaces the active pair set. Call
`reset_collision_pairs()` to restore all pairs, with adjacent pairs excluded
by default. Individual pairs can still be disabled with
`remove_collision_pair(first_geometry, second_geometry)`.

## Collision-sphere backend

`SphereCollisionModel` is a lightweight alternative for high-frequency state
and trajectory queries. The caller supplies positive-radius spheres in link
coordinates; Pinocchio transforms them during FK and the backend evaluates the
active sphere pairs. It supports semantic groups, safety margins, signed
minimum distance, world-space sphere output, and `[batch, nq]` distance queries.

Sphere geometry is fixed when the model is constructed. The Python `spheres`
property returns independent snapshots; editing them does not change the model.
Construct a new model to use modified geometry. Boolean queries check immutable
radius bounds once per query to select the ordinary squared-distance loop;
extreme radii or margins retain the extended-precision fallback.
Each query updates joint kinematics and the distinct frames owning spheres;
unrelated sensor and tool frames are not refreshed. Spheres sharing a frame
reuse its placement, including when the active collision groups change.

```python
spheres = [
    hm.CollisionSphere("left_upper_0", "left_upper_arm", [0, 0, 0.12], 0.09),
    hm.CollisionSphere("right_upper_0", "right_upper_arm", [0, 0, 0.12], 0.09),
]
sphere_model = hm.SphereCollisionModel(urdf_path, spheres)
sphere_model.set_collision_groups(
    {"left_arm": left_links, "right_arm": right_links},
    [("left_arm", "right_arm")],
)
distance = sphere_model.minimum_distance(q).distance
distance_and_gradient = sphere_model.minimum_distance_with_gradient(q)
trajectory_distances = sphere_model.batch_minimum_distances(q_path)

planner = hm.SamplingPlanner.from_sphere_collision_joints(
    sphere_model, joint_names, sphere_model.neutral_configuration(),
    security_margin=0.01,
)
```

The combined query returns the minimum signed distance and its analytic
`nv`-dimensional tangent-space gradient from one kinematics and Jacobian update,
including for manifold configurations where `nq != nv`. Link-local sphere
offsets include both frame translation and angular point-velocity terms.
The local offset is rotated directly, preserving its Jacobian contribution
after large world translations. Every nonzero center separation has a unit
normal; exactly coincident centers use the deterministic +X normal and a zero
distance gradient. Spheres rigidly attached to the same joint have an exact zero
distance gradient, including spheres on different links joined by fixed joints.
Static world frames skip their identically zero point Jacobian. Distances and
collision predicates fall back to extended
precision when squared lengths overflow or underflow. World coordinates or
signed distances that remain unrepresentable raise `OverflowError`; overflowing
gradient arithmetic also raises `OverflowError` rather than returning NaN or Inf.
An unrepresentably large positive clearance does not hide a finite minimum and
is excluded by finite distance filters. A minimum query still raises when its
final distance is unrepresentable, including for a batch row.
Batch queries reuse consecutive exactly equal configurations within that call;
distinct small movements are evaluated separately, and no cache survives the call.
`q` and each `q_path` row are complete normalized `nq` configurations. Named
planning factories instead accept scalar positions in `joint_names` order and
preserve the supplied full reference configuration.
`PathOptimizer.from_sphere_collision_joints` pulls back this `nv` gradient to
the selected scalar coordinates when positive soft `clearance` is requested,
including for continuous joints and frozen floating-base contexts. The legacy
`from_sphere_collision_model` factory accepts only vector-space configurations;
use the named factory for non-Euclidean models.
`pair_revision` increases whenever collision groups are reset, replaced, or
cleared, allowing short-lived distance caches to detect reconfiguration.
If an existing path optimizer observes an empty pair set, both its hard check
and soft clearance term become inactive until pairs are restored.
Without soft clearance the adapter uses early-exit collision checks; minimum-
distance scans and their one-shot cache are enabled only when clearance costs
need them.

Sphere geometry is an approximation and is never generated implicitly. Use a
conservatively padded sphere set for rejection queries, and retain Coal as the
exact final validator unless the sphere model has a separately verified
coverage guarantee. An instance owns mutable Pinocchio query buffers and must
not be queried concurrently without external synchronization.

The design is informed by cuRobo's collision-sphere representation and fused
batch-query architecture, but is independently implemented around Pinocchio
and Eigen; the collision backend does not import Torch, Warp, or cuRobo.

### Offline sphere fitting

The NumPy fitter selects large medial candidates from interior and surface
samples. Its mesh adapter uses optional `trimesh` and SciPy dependencies for
voxelization and filling. `inscribed` mode keeps sampled interior radii; the optional
`sampled_coverage` mode expands the selected spheres to cover every supplied
sample. The latter remains a discrete approximation, not a mathematical proof
of continuous mesh coverage.

`SphereFitOptions.chunk_size` (default 512) bounds both dimensions of the
temporary distance blocks. Radius fitting, sampled expansion, and coverage
metrics process blocks without creating an all-samples distance matrix.
Long runs of contained candidates are pruned in bounded blocks while preserving
the stable greedy order and checks against newly selected spheres.
Coordinate differences are computed directly, preserving small separations
when the model is far from the origin, within floating-point input precision.
Exceptional coordinate ranges use stable norms to retain finite radii and gaps
when intermediate squares overflow or underflow. Gap evaluation also subtracts
the radius before restoring an overflowing distance's scale. Unrepresentable
final gaps are rejected; sampled coverage keeps its absolute `1e-12` tolerance.
This preprocessing uses NumPy and does not require the native collision component.
Coordinates, radii, fit metrics, and floating-point options must be real; complex values
raise `TypeError` before conversion, including values with zero imaginary part.

```bash
python -m pip install '.[examples]' 'scipy>=1.10'
./scripts/run.sh python3 \
  examples/python/visualization/sphere_fit_viser.py \
  --mesh /absolute/path/to/link.stl \
  --link left_forearm \
  --max-spheres 24 \
  --sampled-coverage \
  --padding 0.003 \
  --output left_forearm.spheres.json
```

Use `--fit-only` on build machines without a viewer. The JSON file is
versioned and stores link-local spheres; load it with `load_sphere_model()` and
convert it with `make_collision_spheres()` before constructing a
`SphereCollisionModel`. Always inspect `sampled_coverage`, mean gap, maximum
gap, and the Viser overlay before using a generated model for planning.

For a complete robot, the batch tool reads every URDF `<collision>` element,
including mesh scale and collision origin. Mesh, box, cylinder, and sphere
geometry are supported; repeat `--package-dir` for `package://` resources and
repeat `--link` to fit only selected links. This loads geometry only for the
requested links, so unrelated meshes need not be present. URDF link names are
still checked for duplicates.
Repeated mesh URIs are resolved and read once per call. Repeated box, cylinder,
and sphere declarations with identical attributes also share their initial mesh
construction. Each instance receives independent scale/origin transforms, and
cached geometry is released at its last use. Unique geometries and links with
one collision element do not require extra copies.

```bash
./scripts/run.sh python3 examples/python/collision/fit_urdf_spheres.py \
  --urdf /absolute/path/to/robot.urdf \
  --package-dir /path/to/ros/src \
  --sampled-coverage --padding 0.003 \
  --output robot.spheres.json
```

The lightweight editor exposes link selection, sphere count, voxel pitch,
minimum radius, padding, coverage mode, refitting, metrics, and saving in a
browser UI:

```bash
./scripts/run.sh python3 \
  examples/python/visualization/sphere_model_editor.py \
  --urdf /absolute/path/to/robot.urdf \
  --output robot.spheres.json
```

If the output JSON already exists, the editor loads and displays its spheres.
Refitting one link preserves every untouched link, and **Fit all links** is
available for deliberate full regeneration. Saving uses an atomic replacement
so a validation or write failure cannot leave a partially written model.
Temporary files are also removed if writing, flushing or syncing fails or is
interrupted, provided the filesystem permits cleanup; the existing model is
preserved until the final replacement succeeds.

## HumanoidAssets gizmo demo

The dual-arm Viser demo accepts an explicit Marvin URDF or asset-root path.
Drag either end-effector gizmo to solve IK and run left-arm/right-arm geometry
checks in real time. Each complete-arm group
includes the arm chain plus every collision-bearing descendant of the EE,
including the hand base, fingers, tool, or fixture links:

```bash
python -m pip install '.[examples]'
./scripts/build.sh
python3 examples/python/visualization/dual_arm_collision_gizmo.py \
  --urdf /path/to/robot_with_ee.urdf
```

The Collision panel reports clear/collision state and minimum distance. Red
markers and a segment show Coal's nearest points. Useful variants are:

The **Collision mode** menu switches directly between complete-arm checks
(including fixtures), arm-chain-only checks, complete-arm checks plus internal
self-collision, and disabled. **Active collision pair** exposes the expanded
geometry pairs; use **Disable selected pair** and **Restore mode pairs** to
edit them interactively.

Use the **Safety margin (mm)** control and enable **Reject unsafe IK** to keep
the last accepted arm pose when a candidate falls below the threshold. The
Marvin demo filters its known `wrist_pitch_j5`/`wrist_roll_j7` mechanism pairs;
additional semantic exclusions can be supplied with
`--disable-link-pair LINK1:LINK2`.

```bash
# Also detect self-collision inside each arm.
python3 examples/python/visualization/dual_arm_collision_gizmo.py --arm-self-collision

# Validate IK, asset loading, group expansion, and one distance query.
python3 examples/python/visualization/dual_arm_collision_gizmo.py --validate-only

# Benchmark the combined collision + minimum-distance query.
python3 examples/python/visualization/dual_arm_collision_gizmo.py \
  --validate-only --benchmark-samples 200
```

On the development machine, the complete Marvin arms (including fixtures)
expanded to 100 cross-arm pairs and measured about 40.23 ms average (about
25 Hz). With filtered arm self-collision enabled, 164 pairs measured about
87.98 ms average (about 11 Hz). Select **Between arm chains only** when the
faster 49-pair arm-only query is sufficient. Results depend on CPU, meshes, configuration, and Coal's
broad-phase state; benchmark on the deployment machine.

### Real-time query levels

The Gizmo uses two query levels instead of computing exact distance every
frame:

- `is_within_distance(q, margin)` uses Coal's collision request security
  margin and early stopping for the per-frame safety decision.
- `evaluate(q)` computes every collision plus exact minimum distance and
  nearest points at the configurable **Exact distance rate (Hz)**.

For the 100-pair complete-arm Marvin setup, the 20 mm threshold query measured
about 0.0076 ms average and 0.0077 ms P95 on the development machine, while
the exact query remained about 40 ms. The default exact-distance rate is 4 Hz;
the safety decision still runs on every accepted Gizmo update.

Other command-line examples:

- `collision_demo.py`: one-configuration query.
- `collision_pair_management_demo.py`: inspect, remove, and reset pairs.
- `dual_arm_collision_demo.py`: configure two semantic arm groups without GUI.
- `collision_path_scan_demo.py`: sample a joint-space segment for collisions.
