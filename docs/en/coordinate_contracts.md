# Coordinate contracts across motion modules

<div class="language-switcher">English · <a href="../zh_CN/coordinate_contracts.html">简体中文</a></div>

Choose joint names and a full reference configuration before combining IK,
planning and timing. Vector length alone does not identify a configuration:
two models with equal dimensions can use different joint orders or encodings.

## Which vector does each API consume?

| API | Position coordinates and order | Pose frame |
|---|---|---|
| `robot.kinematics`, `robot.dof` | Scalar positions of the automatically selected default chain | Chain base |
| `robot.create_kinematics(base, tip)` | Scalar positions in `robot.chain_joint_names(base, tip)` order | Explicit chain base |
| `RobotModel`, `robot.model_dof`, `robot.nq/nv` | All independent scalar joints; continuous joints use one angle | URDF model root |
| Native collision model | Full `nq` configuration; use `joint_coordinates` for block indices | URDF model root |
| Pinocchio/Pink/multi-seed retargeting | Full Python Pinocchio `nq` configuration, in its model order | URDF model root |
| `SamplingPlanner`, `PathOptimizer` | Scalar positions in the `JointSpace` order | Defined by the validator |
| `CuroboPlanner` | Complete active scalar joint set in `planner.joint_names` order; configured locked joints are omitted | Configured cuRobo base |
| Native joint trajectories and TOPPRA | Columns retain the supplied waypoint order | No pose-frame conversion |

Native `Robot` selects the longest supported root-to-leaf chain by default.
For branched robots, select `base` and `tip` explicitly. `JointSpace.joint_names`
labels coordinates; it does not reorder input vectors. In contrast, the optional
`CuroboPlanner(joint_names=...)` accepts a permutation of its complete active set
and converts backend input/output order.

Scalar revolute/continuous positions use radians; prismatic positions use meters.
Derivative limits use these same joint units per second and per second squared.
Position arrays from scalar trajectory samplers are not full manifold
configurations, and their derivatives are not full-model tangent vectors.

## Configuration indices and tangent indices

| Joint representation | Position block (`q`, size `nq`) | Tangent block (`v`, size `nv`) |
|---|---|---|
| Scalar revolute or prismatic | One angle or displacement | One velocity |
| Native scalar continuous | One angle | One angular velocity |
| Pinocchio continuous | `[cos(angle), sin(angle)]` | One angular velocity |
| Pinocchio free flyer | Translation and quaternion | Six tangent coordinates |

Use `idx_q/nq` to pack configurations and `idx_v/nv` to restrict velocities,
displacements or tangent-space gradients. A continuous joint can make all later
position and velocity indices differ. `JointPositionMap.joint_velocities()`
restricts an **already tangent-space** gradient; it does not differentiate a
cost expressed in raw `nq` coordinates or rotate a Cartesian Jacobian.

## Prepare a named mapping once

```python
names = robot.chain_joint_names("torso", "left_tool")
chain = robot.create_kinematics("torso", "left_tool")
if chain is None:
    raise ValueError("unsupported chain")

# Application-supplied full context, including the other arm and base pose.
collision.validate_configuration(reference)
mapping = collision.create_joint_position_map(names)
chain_q = mapping.joint_positions(reference)
base_T_tool = chain.forward(chain_q)
full_q = mapping.configuration_with_joint_positions(reference, chain_q)
```

The mapping owns a topology snapshot and preserves the supplied name order.
Packing returns an independent full vector and retains every unselected block.
Extraction of continuous pairs returns principal angles, without winding history.
Mappings do not propagate mimic constraints, enforce limits or check collisions.
Only selected continuous pairs are checked during extraction; a preserved base
quaternion still needs validation by its owning model. Rebuild maps after model
topology changes.

Python Pinocchio models use the same mapping contract without importing the
native extension:

```python
from holistic_motion.configuration import JointPositionMap

mapping = JointPositionMap.from_model(solver.model, names)
full_q = mapping.configuration_with_joint_positions(reference, chain_q)
```

Use the corresponding model's reference and mapping at each integration boundary.
The Python mapping does not make independently packaged native collision and
Python Pinocchio runtimes compatible; retain the process boundary described in
{doc}`collision`. Chain FK returns `base_T_tool`; obtaining `world_T_tool`
requires multiplying by the current `world_T_base` from the full model.

## Preserve topology through planning and timing

1. Create named collision factories with the same names and full reference.
   Their contexts are captured at construction; recreate them to change the
   other arm's configuration or floating base.
2. Check the planning and optimization results before using their paths.
3. Call `result.joint_space.unwrap_path(result.path)` on the final successful
   result. This uses the request's limits and periodic topology, even if the
   planner was subsequently reconfigured.
4. Time the lifted scalar path. Pack sampled positions through the prepared
   mapping to recover full configurations, and validate the resulting motion.

Across `+pi/-pi`, unwrapping retains the shortest planned arc. Lifted angles may
leave the representative `[-pi, pi]` interval; this is expected for periodic
joints. Bounded joints keep their original coordinates. If hardware needs
multiple revolutions or absolute winding limits, use an application-defined
bounded scalar space instead of inferring winding from principal angles.

TOPPRA uses a natural cubic spline through supplied waypoints. A multi-waypoint
spline or native trajectory blending can change the checked geometry. The
`named_plan_and_retime.py` example times each polyline edge as a separate
two-point TOPPRA path, stopping at corners. It retains the checked geometry at
the cost of extra stops; acceleration can jump at joins and jerk is not bounded.
Discrete planning and output checks remain resolution-dependent.

## Run the complete named-joint example

Supply names, assets, start/goal and derivative limits for your robot:

```bash
./scripts/run.sh python3 examples/python/planning/named_plan_and_retime.py \
  --urdf /absolute/path/robot.urdf \
  --joint-names left_shoulder left_elbow \
  --start 0.1 -0.2 --goal 0.3 -0.4 \
  --max-velocity 0.8 0.8 --max-acceleration 1.5 1.5 \
  --output /tmp/named-motion.npz
```

Replace every name and limit with caller-owned model data. `--reference` accepts
the complete collision-model `nq` configuration; omitting it uses neutral context.
The same names order applies to start, goal, limits and exported scalar columns.
The NPZ also stores full configurations and the reference. Only selected joints
move; select both arms together to search coordinated dual-arm motion.

The JSON summary separates planning, optimization, timing and verification costs,
and reports active collision pairs. A model with no pairs performs no collision
checks. Failed planning/optimization does not export a trajectory. The example
uses one model sequentially; parallel workers need independent collision models.

For algorithm and backend selection, see {doc}`architecture`; for repeatable
end-to-end measurements, see {doc}`development`.
