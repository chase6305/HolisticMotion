# Installation

<div class="language-switcher">English · <a href="../zh_CN/installation.html">简体中文</a></div>

## Requirements

- A C++17 compiler and CMake 3.22 or newer.
- Conan 2.28 or newer.
- Python 3.9 or newer when bindings are enabled.
- A caller-provided URDF for model-dependent operations.

## Standard build

```bash
./scripts/build.sh
```

This one-command build enables Python and collision support. Add `--cuda` for
the optional CUDA backend, or use `--no-collision` for a lighter build.

Run the installed bindings rather than importing stale files from the source
tree:

```bash
PYTHONPATH=$PWD/build/install pytest tests/python
```

## Optional features

| Conan option | Purpose | Default |
|---|---|---|
| `with_python` | Build pybind11 extension | `True` |
| `with_tests` | Build C++ smoke tests | `False` |
| `with_cuda` | Build CUDA FEP batch backend | `False` |
| `with_collision` | Build Pinocchio/Coal collision library | `True` |

Minimal CPU-only installation without collision support:

```bash
conan install . --output-folder=build --build=missing \
  -o '&:with_cuda=False' -o '&:with_collision=False'
```

Pinocchio and Coal are direct Conan requirements in the default configuration.
`CMakeDeps` supplies their targets; CMake does not search for unmanaged system
copies.

## C++ target boundaries

Consumers link only the components they use. Core kinematics, trajectory, and
planning APIs use `HolisticMotion::holistic_motion`; collision queries use the
separate `HolisticMotion::collision` target, which propagates Pinocchio and Coal:

```cmake
find_package(HolisticMotion REQUIRED CONFIG)
target_link_libraries(my_motion_app PRIVATE HolisticMotion::holistic_motion)
target_link_libraries(my_collision_app PRIVATE HolisticMotion::collision)
```

These names are identical for a CMake install tree and Conan `CMakeDeps`.
When CUDA is enabled, only the core target propagates `CUDA::cudart`; the
collision component remains independent of CUDA and the core library.
