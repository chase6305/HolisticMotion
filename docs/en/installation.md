# Installation

<div class="language-switcher">English · <a href="../zh_CN/installation.html">简体中文</a></div>

## Requirements

- A C++17 compiler and CMake 3.22 or newer.
- Conan 2.28 or newer.
- Python 3.9 or newer when bindings are enabled.
- NumPy 1.23 or newer in the interpreter used to run the Python package.
- A caller-provided URDF for model-dependent operations.

## Standard build

```bash
./scripts/build.sh
```

This one-command build enables Python and collision support. Add `--cuda` for
the optional CUDA backend, or use `--no-collision` for a lighter build.

On Linux, the build helper records Conan dependency directories in the local
libraries' runtime search paths. `source scripts/activate.sh` and
`./scripts/run.sh python3 ...` expose the install while preserving the caller's
loader paths for separate native-collision and Python-retargeting processes.
Older installs and other platforms use the Conan runtime environment. Exported
Conan packages retain their standard configuration; the local search-path
setting is enabled by `scripts/build.sh`.

Python-enabled Conan packages expose their installation root through the Conan
run environment's `PYTHONPATH`. The consumer test selects the recorded Python
minor version and verifies the installed bindings, model loading, named joint
mapping, and trajectory evaluation. Install NumPy in that interpreter before
running `conan create`; Conan manages the native dependencies, not Python runtime
packages.

Conan's global runtime environment can shadow libraries used by Python
Pinocchio. For a Linux cache-local Python package with per-library search paths:

```bash
conan create . --no-remote -o '&:with_python=True' -o '&:shared=True' \
  -c tools.build:jobs=2 \
  -c 'tools.cmake.cmaketoolchain:extra_variables={"CMAKE_INSTALL_RPATH_USE_LINK_PATH": True}'
```

This mode records absolute Conan cache paths; rebuild after moving that cache.
Use the default configuration for relocatable Conan packages.

For a manual cache-local CMake installation, pass
`-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON` at configuration, as the build helper
does. A default relocatable install instead requires the Conan runtime
environment when linking and running consumers. Discovering dependency packages
with CMake does not itself configure the operating system's dynamic loader.
Per-library search paths do not establish compatibility between independent
Pinocchio runtimes; the supported process combinations below still apply.

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
| `python_version` | Native Python major.minor in the package identity | `auto` |
| `python_abi` | Interpreter implementation and layout in the native identity | `auto` |
| `with_tests` | Build C++ smoke tests | `False` |
| `with_cuda` | Build CUDA FEP batch backend | `False` |
| `cuda_architectures` | Explicit GPU targets in the CUDA binary identity | `auto` |
| `cuda_version` | CUDA toolkit major.minor in the binary identity | `auto` |
| `with_collision` | Build Pinocchio/Coal collision library | `True` |

Minimal CPU-only installation without collision support:

```bash
conan install . --output-folder=build --build=missing \
  -o '&:with_cuda=False' -o '&:with_collision=False'
```

Pinocchio and Coal are direct Conan requirements in the default configuration.
`CMakeDeps` supplies their targets; CMake does not search for unmanaged system
copies.

`cuda_architectures=auto` records `HOLISTICMOTION_CUDA_ARCHITECTURES` from
`tools.cmake.cmaketoolchain:extra_variables`, or defaults to `75;80;86;89;120`.
An explicit option also configures CMake. Distinct target sets, including
`-real`/`-virtual` code selections, have distinct binary identities; order and
duplicates are normalized. A conflicting explicit option and CMake value is
rejected. CUDA-disabled packages omit this option from their identity.

```bash
conan install . --output-folder=build-cuda --build=missing \
  -o '&:with_cuda=True' -o '&:cuda_architectures=120'
```

Conan CUDA packages require explicit numeric target lists. Compiler/device
keywords such as `all` or `native` remain available in standalone CMake builds;
they cannot identify a reproducible architecture set before Conan resolves a
binary. Select the CUDA compiler and toolkit through the build environment.

`cuda_version=auto` probes the selected `nvcc` (`CMAKE_CUDA_COMPILER` in CMake
extra variables, `tools.build:compiler_executables["cuda"]`, `CUDACXX`, or
`nvcc` on PATH, in that order), then pins that compiler for configuration.
Different toolkit major.minor releases receive distinct package IDs. To consume
a prebuilt package without a compiler available during Conan graph resolution,
select its version explicitly, for example `-o '&:cuda_version=12.8'`.
CMake verifies the package identity against the actual toolkit before building;
installed CMake and Conan consumers require that same major.minor toolkit.
Patch releases are accepted. This conservative supported combination prevents
silently substituting a different CUDA runtime. CUDA-disabled packages omit
both CUDA options from their identity.

`python_version=auto` records the interpreter running Conan, or the interpreter
selected through `tools.cmake.cmaketoolchain:extra_variables` as
`Python3_EXECUTABLE`. Different Python minors receive different native package
IDs; `with_python=False` removes this option from the C++ package identity.
CMake checks that the selected interpreter matches the recorded minor.
`scripts/build.sh` forwards the minor of `PYTHON_EXECUTABLE` automatically.
For an explicit interpreter:

```bash
conan create . --no-remote -o '&:python_version=3.12' \
  -c 'tools.cmake.cmaketoolchain:extra_variables={"Python3_EXECUTABLE": "/path/to/python3.12"}'
```

Use separate build/install directories for each Python minor, including native
pip builds using a prepared Conan toolchain. New build metadata records
`runtime_info().native_python_version`; package import rejects a different minor
before loading the extension. Patch releases of the same minor are accepted.
The portable toolkit wheel carries no native Python ABI requirement.

`python_abi=auto` also probes the selected interpreter. Standard CPython uses
`cpython`; debug and free-threaded layouts add `d` / `t`, and PyPy includes its
extension ABI tag. These variants have separate package IDs even at the same
Python minor. When only a different `python_version` is specified without an
interpreter, the default target is standard CPython; select `Python3_EXECUTABLE`
or an explicit `python_abi` for another layout. CMake verifies the actual
interpreter, and `runtime_info().native_python_abi` lets imports reject a foreign
ABI before loading the extension. C++-only packages omit both Python options.
`scripts/build.sh` forwards the selected interpreter's version and ABI together.

ABI separation does not change the instance concurrency contracts. The extension
does not declare GIL-free operation; free-threaded CPython applies its
[default extension GIL behavior](https://docs.python.org/3.13/howto/free-threading-extensions.html).

Linux core builds have been exercised on CPython 3.9, 3.10, 3.12 and 3.13,
free-threaded CPython 3.13, and PyPy 3.10. The optional native ABI CI job covers
PyPy and both 3.13 layouts without collision or CUDA. These checks do not extend
the separately validated Python Pinocchio wheel combinations below.

## Python native runtime combinations

Install only the Python toolkit (retargeting, TOPPRA, geometry preprocessing and
simulation adapters):

```bash
python -m pip install . \
  -Ccmake.define.HOLISTICMOTION_PYTHON_TOOLKIT_ONLY=ON \
  -Cwheel.platlib=false
```

This builds a `py3-none-any` wheel without finding a C++ compiler, Eigen,
pybind11's CMake package or Conan dependencies. Replace `.` with
`'.[retargeting]'` or another optional Python dependency group as needed.
The wheel selects toolkit mode automatically and exposes no native `Robot` or
collision types. Native builds use the Conan/CMake workflow above; pip builds
of the native package also require an explicitly prepared Conan toolchain.

| Mode | Supported components in one process |
|---|---|
| Default native build | C++ core with either Python Pinocchio retargeting or lazily loaded Conan collision |
| Native build with `--no-collision` | C++ core together with Python Pinocchio retargeting |
| `HOLISTICMOTION_PURE_PYTHON=1` | Python Pinocchio retargeting, NumPy/PyTorch trajectories and pure Python tools |
| Pure Python toolkit wheel | Same toolkit capabilities, selected automatically through installed metadata |

When updating an older monolithic Python extension, rebuild and reinstall the
native package together with its Python files and generated metadata. A
collision-enabled install now contains both core and collision extension files;
the public `holistic_motion` import names remain available. Rebuild C++ consumers
against the updated public headers as well.
The extensions and their linked core/collision libraries verify native build
identities derived from native sources, compiler settings and configuration.
The public collision loader queries the candidate binary before Python type
registration, rejecting mismatched or unverifiable collision extensions with a
rebuild diagnostic. After replacing native binaries, start a fresh Python process.

Run independently packaged Python Pinocchio and the Conan collision runtime in
separate processes. Matching Pinocchio version numbers do not establish ABI
compatibility for Coal, Boost or compiler settings. Ordinary `import holistic_motion`
loads only the native core. Accessing a collision type such as `hm.CollisionModel`,
or invoking a collision planning factory, loads the separate collision extension.
The public loaders check for Python Pinocchio and independently loaded `coal`,
`hppfcl`, or `eigenpy` before loading collision; retargeting checks for native
collision before loading Python Pinocchio. These loaders serialize concurrent
first imports. `from holistic_motion import *` requests all public types and
therefore loads collision when it was compiled. Direct imports from third-party code
do not pass through these checks and must respect the same process boundary.
Start the Python Pinocchio worker outside an active global Conan runtime
environment: a child process inherits loader search paths. Process separation
does not reset `LD_LIBRARY_PATH`. The local build helper's per-library runtime
paths let its native application and an independent Python worker keep their
own dependency lookup.

The Linux runtime combinations verified by the optional checks are:

| Runtime | Pinocchio | Boost | URDFDOM / TinyXML2 |
|---|---|---|---|
| Conan collision | 3.8.0 | 1.88.0 | Conan-managed dependencies |
| Python minimum wheel cohort | 3.8.0 | 1.89.0 | 4.0.1 / 10.0.0 |
| Python tested 4.x wheel cohort | 4.1.0 | 1.90.0 | 6.0.0 / 11.0.0 |

The same Pinocchio version can therefore use different Boost libraries. For the
3.8.0 Python wheel, explicitly install its matching parser dependencies:

```bash
python -m pip install --only-binary=:all: 'pin==3.8.0' 'coal==3.0.2' \
  'cmeel-urdfdom==4.0.1' 'cmeel-tinyxml2==10.0.0'
```

Installing `pin==3.8.0` alone can select URDFDOM 6 and fail to import because
`liburdfdom_sensor.so.4.0` is missing. The minimum-version CI job pins the tested
cohort on Python 3.10; the current-version job follows `pin>=3.8,<5` on Python
3.11 with its resolved dependencies. The toolkit-only wheel independently
supports Python 3.9 and newer. Python 3.9 is not part of the validated retargeting
wheel cohort: a Linux installation probe found a Pinocchio 3.8 wheel but could
not resolve its required `eigenpy>=3.12` binary dependency. `--only-binary` keeps
that installation from silently switching to an unvalidated native source build.

```bash
./scripts/run.sh python3 examples/python/collision/basic_query.py --help
./scripts/run-python-toolkit.sh python3 examples/python/retargeting/pink_dual_arm.py --help
```

Set the pure Python environment variable before importing the package. Changing
it later does not unload shared libraries. `holistic_motion.runtime_info()`
reports loaded components, build capabilities and Pinocchio versions without
loading Pinocchio, Torch or the collision extension. `collision_compiled` reports
build support; `collision_loaded` reports whether this process has actually loaded
the collision extension. Unknown build capabilities are `None`.
CMake installs generated build metadata as `_build_config.py`.

## C++ target boundaries

Consumers link only the components they use. Core kinematics, trajectory, and
planning APIs use `HolisticMotion::holistic_motion`; collision queries use the
separate `HolisticMotion::collision` target, which depends on Pinocchio and Coal:

```cmake
find_package(HolisticMotion REQUIRED CONFIG)
target_link_libraries(my_motion_app PRIVATE HolisticMotion::holistic_motion)
target_link_libraries(my_collision_app PRIVATE HolisticMotion::collision)
```

These names are identical for a CMake install tree and Conan `CMakeDeps`.
When CUDA is enabled, only the core target propagates `CUDA::cudart`; the
collision component remains independent of CUDA and the core library.

The installed CMake SDK supports explicit components. Omitting components keeps
the historical behavior of loading all built components. Selecting `core` skips
Pinocchio/Coal discovery; selecting `collision` skips the core fmt/spdlog dependencies:

```cmake
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core)
target_link_libraries(my_motion_app PRIVATE HolisticMotion::holistic_motion)

# Both components: COMPONENTS core collision
# Collision only: COMPONENTS collision
# Optional discovery: COMPONENTS core OPTIONAL_COMPONENTS collision
```

Either component can be optional; inspect `HolisticMotion_core_FOUND` or
`HolisticMotion_collision_FOUND` after discovery. Unavailable optional core
dependencies, including CUDA, do not prevent collision-only use. Missing
required components and unknown names produce clear diagnostics. Conan's full
dependency graph remains controlled by package options; use
`with_collision=False` for a minimal Conan installation.
