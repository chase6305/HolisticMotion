# Third-party notices

HolisticMotion contains independent, project-native implementations informed
by the following projects. Their source trees are not vendored and are not
runtime dependencies.

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
