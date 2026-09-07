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

## Compiler and Conan profiles

Newly generated Conan toolchains record the selected compiler family and version.
CMake checks them against the actual C++ compiler before configuring dependencies.
A mismatch stops configuration with instructions to select a matching compiler.
This checks compiler identity and version, not the full ABI of prebuilt dependencies.
Standalone CMake builds and older toolchains without this metadata skip the check;
regenerate the Conan toolchain to enable it in an existing project.

Use `CONAN_PROFILE_HOST` and `CONAN_PROFILE_BUILD` to select profile names or paths
for `scripts/build.sh`. Both default to `default`. For example, on a system with
GCC 12 installed at the following paths, save this as `/tmp/hmotion-gcc12.profile`:

```ini
include(default)

[settings]
compiler=gcc
compiler.version=12
compiler.libcxx=libstdc++11
compiler.cppstd=gnu17

[conf]
tools.build:compiler_executables={"c": "/usr/bin/gcc-12", "cpp": "/usr/bin/g++-12"}
```

Build in a fresh directory after changing compilers:

```bash
CONAN_PROFILE_HOST=/tmp/hmotion-gcc12.profile \
CONAN_PROFILE_BUILD=/tmp/hmotion-gcc12.profile \
BUILD_DIR="$PWD/build-gcc12" ./scripts/build.sh --tests --no-collision
```

The profile settings describe the intended compiler; `tools.build:compiler_executables`
selects its executables. See the official [Conan CMakeToolchain documentation](https://docs.conan.io/2/reference/tools/cmake/cmaketoolchain.html).
This example leaves the global default profile unchanged. For cross compilation,
select separate host and build profiles appropriate to the target and build machine.

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
