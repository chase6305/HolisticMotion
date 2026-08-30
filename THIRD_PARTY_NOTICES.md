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

## Algorithm and design references

The projects below are not imported, linked, or vendored at runtime.

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
HolisticMotion's sphere fitting, collision model, path optimizer, retargeting,
and query APIs are maintained locally and do not depend on cuRobo, PyTorch, or
Warp.

## Scope

Conan packages and optional Python packages retain their own copyright and
license terms. Consult the package's installed license files before
redistributing a binary bundle. References above describe provenance and
technical influence; they do not imply endorsement by the upstream projects.
