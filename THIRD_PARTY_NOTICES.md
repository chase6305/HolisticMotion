# Third-party notices

This file records both linked third-party dependencies and upstream projects
that informed independent HolisticMotion implementations. It does not replace
the license files distributed by each dependency.

## Linked dependencies

### Pinocchio

- Project: Pinocchio rigid-body dynamics library
- Upstream: <https://github.com/stack-of-tasks/pinocchio>
- Version declared by HolisticMotion: 3.8.0
- License: BSD 2-Clause

Pinocchio is resolved by Conan and linked by the optional collision component,
which is enabled by default. It is not vendored in this repository.

### Coal

- Project: Coal collision detection library
- Upstream: <https://github.com/humanoid-path-planner/coal>
- Version declared by HolisticMotion: 3.0.2
- License: BSD License

Coal is resolved by Conan and linked by the optional collision component,
which is enabled by default. It is not vendored in this repository. Conan is
configured with Coal's OctoMap integration disabled.

### dex-retargeting (optional Python hand backend)

- Project: dex-retargeting
- Upstream: <https://github.com/dexsuite/dex-retargeting>
- Version declared by HolisticMotion: 0.5.0
- License: MIT

The `hand-retargeting` extra imports the upstream VectorOptimizer, RobotWrapper,
and mimic-joint adaptor. Integration was validated against the caller-provided
checkout at commit `3f56141`. Source and robot assets are not vendored or downloaded
implicitly. The HolisticMotion adapter corrects the temporal objective scalar,
enforces mimic-induced position limits, and reports native optimizer failures.
This optional backend requires PyTorch and the upstream runtime dependencies,
including NLopt; those packages retain their own licenses.

### NVIDIA cuRobo v2 (optional Python planning backend)

- Project: NVIDIA cuRobo
- Upstream: <https://github.com/NVlabs/curobo>
- Qualified source: `299d7a6c271476c7e9fd9ecd1a37a462f14754d1`
- License: Apache License 2.0
- Copyright: NVIDIA Corporation

`holistic_motion.kit.curobo_planner` imports the upstream public motion planner,
scene and state APIs only when explicitly constructed. Callers install cuRobo
and its CUDA/PyTorch/Warp dependencies separately and provide local robot
configurations and assets. No upstream source, generated GPU artifacts or
robot assets are vendored or downloaded by the adapter. Transitive dependencies
and external assets retain their respective licenses.

## Algorithm and design references

The local algorithms described below do not import their upstream projects.
The separate optional integrations above have their own runtime dependencies.

## TOPPRA

- Project: Time-Optimal Path Parameterization via Reachability Analysis
- Upstream: <https://github.com/hungpham2511/toppra>
- License: MIT
- Copyright: 2017 Hung Pham

The path timing implementation uses the published TOPPRA state formulation and
backward/forward reachability approach. The implementation and public API in
HolisticMotion are maintained locally.

## Pink

- Project: Python inverse kinematics based on Pinocchio
- Upstream: <https://github.com/stephane-caron/pink>
- License: Apache License 2.0

The retargeting toolkit follows Pink's weighted task-space differential-IK
formulation. HolisticMotion provides its own solver integration without a
runtime dependency on Pink or qpsolvers.

## cuRobo

- Project: CUDA Robot Motion
- Upstream: <https://github.com/NVlabs/curobo>
- License: Apache License 2.0
- Copyright: NVIDIA Corporation

cuRobo informed the collision-sphere representation, batched-query design,
deterministic multi-seed retargeting, and the separation between seed
generation, feasibility-aware optimization, and best-solution tracking.
HolisticMotion's sphere fitting, collision model, path optimizer, cuRobo-style
retargeting, and native collision-query APIs are maintained locally and do not depend on cuRobo,
PyTorch, or Warp. The separate optional dex-retargeting hand backend uses PyTorch.

## Scope

Conan packages and optional Python packages retain their own copyright and
license terms. Consult the package's installed license files before
redistributing a binary bundle. References above describe provenance and
technical influence; they do not imply endorsement by the upstream projects.
