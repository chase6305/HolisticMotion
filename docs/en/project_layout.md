# Project layout

The repository root is the project boundary. It deliberately does not add a
second `holistic_motion/` wrapper around `include`, `src`, `bindings`, and
`python`: the C++ install namespace already lives at
`include/holistic_motion/`, while the Python import namespace lives at
`python/holistic_motion/`.

```text
include/holistic_motion/   public C++ API, grouped by robotics domain
src/                       C++ implementation mirroring the public modules
bindings/python/           compiled Python binding translation units
python/holistic_motion/    pure-Python trajectory, retargeting, and visualization APIs
examples/python/           examples grouped by domain
tests/cpp/                 C++ unit and integration tests
tests/python/              Python unit and integration tests
cmake/modules/             target source lists and build modules
docs/en, docs/zh_CN/       bilingual user and developer documentation
scripts/                   supported developer entry points
```

Legacy launchers remain under `python/examples/` while examples migrate to the
domain-oriented tree. New reusable visualization or collision policy code
belongs in `python/holistic_motion/`, not in an example script.

Algorithm integrations are organized by domain rather than upstream project
name: TOPPRA timing belongs in `python/holistic_motion/trajectory/`, while
Pink-style IK belongs in `python/holistic_motion/kit/retargeting/`. Upstream
repositories are neither nested in this tree nor registered as submodules.

`bindings/python/module.cpp` initializes the core extension;
`collision_module.cpp` initializes the optional collision extension on first use. Domain registration
lives in `RobotBindings.cpp`, `KinematicsBindings.cpp`, `SevenAxisBindings.cpp`,
`TrajectoryBindings.cpp`, and the planning/collision units. Optional collision
adapters and collision types are linked into `_holistic_motion_collision`,
separately from the core `_holistic_motion` module. The core registers deferred
collision factories, which retain the public planner API without linking Coal
or Pinocchio into ordinary imports. Named coordinate
conversion is shared through `robot/JointPositionMap.h` and the pure-Python
`configuration.py`; the latter also adapts Python Pinocchio models without loading
the native extension.

Trajectory registration delegates Cartesian operations to
`CartesianTrajectoryBindings.cpp` and joint templates to four translation units
covering dimensions 1–8, 9–16, 17–24 and 25–32. Their private headers hold the
shared array conversion and joint wrapper implementation; changing the module
entry point does not instantiate every dimension in one compiler process.

Native trajectory templates handle path geometry and segment assembly. The
largest implementations (`PathBezierCurve`, `PathSegment`, `TrajectoryBase` and
`TrajectoryReport`) keep definitions in private `*Impl.h` files. Their `.cpp`
files instantiate dimensions 1–8, 9–16, 17–24 and 25–32 separately, with SE(3)
in the first batch. `TrajectoryInstantiationBatches.h` defines the type lists;
report files instantiate only the report member. This bounds compiler memory
without duplicating algorithm definitions or changing the installed headers.
Static consumers can pull the archive members for the dimensions they use.
`src/trajectory/DoubleSProfile.cpp` and `TrapeziumProfile.cpp` implement scalar
clocks once, independently of the Lie-group dimension. The existing protected
profile methods remain wrappers for subclasses.

`src/kinematics/JointMotion.h` defines the host axis scale used by scalar FK,
batch FK and periodic-coordinate conversion. CUDA receives that prepared scale.
`JointPeriod.h` shares nearest admissible periodic-coordinate selection between
ordinary IK and SRS analytic branches. SRS canonical geometry and axis scales
are immutable preparation data in `SRSKinematicsInternal.h`; request workspaces
remain local to each call.
`RobotConfigFilter.cpp` keeps the legacy robot-configuration geometry policy
separate from joint-limit conversion in `utility.cpp`. `IKResults.cpp` owns
result validation, deterministic deduplication, distance ordering and joint filters.

The public TOPPRA and retargeting modules keep their existing entry points.
Private sibling modules separate spline construction, numerical constraints,
speed-profile certification and repair,
task/collision objectives, and bounded QP work. Add algorithm internals there
without creating new public imports or coupling optional robotics runtimes to
the core toolkit. Installation metadata and import-boundary checks belong to
`runtime.py`.
