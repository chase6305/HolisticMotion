# HolisticMotion documentation

<div class="language-switcher">English · <a href="../zh_CN/index.html">简体中文</a></div>

![HolisticMotion modular capabilities covering robot models, kinematics,
retargeting, trajectories, planning, and collision](../assets/holistic-motion-overview.png)

HolisticMotion provides reusable C++17 robotics primitives and Python bindings
for model loading, kinematics, trajectories, collision queries, and pose
retargeting. Robot assets remain caller-provided and are never downloaded
implicitly.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
concepts
collision
planning
trajectory
retargeting
```

```{toctree}
:maxdepth: 2
:caption: Reference

cpp_api
python_api
development
project_layout
```

## First steps

1. Follow {doc}`installation` to build the library with Conan.
2. Read {doc}`concepts` for ownership and optional-feature boundaries.
3. Use {doc}`retargeting` for single-arm, dual-arm, and whole-body IK.
4. Consult {doc}`cpp_api` and {doc}`python_api` for generated references.
