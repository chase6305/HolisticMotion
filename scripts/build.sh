#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPOSITORY_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
CONAN_PROFILE_HOST="${CONAN_PROFILE_HOST:-default}"
CONAN_PROFILE_BUILD="${CONAN_PROFILE_BUILD:-default}"
RUN_TESTS=false
CONAN_WITH_TESTS=False
CONAN_SHARED=False
CONAN_WITH_CUDA=False
CONAN_WITH_COLLISION=True

usage() {
    cat <<EOF
Usage: $0 [options]

Builds and installs HolisticMotion with Python and collision support.

Options:
  --tests          Build and run the C++ and Python tests
  --shared         Build the core library as a shared library
  --no-cuda        Disable the CUDA batch backends (default)
  --no-collision   Disable the Pinocchio/Coal collision component
  --cuda           Enable the optional CUDA batch backends
  --collision      Enable collision (kept for backward compatibility; default)
  -h, --help       Show this help

Environment:
  BUILD_DIR, BUILD_TYPE, PYTHON_EXECUTABLE
  JOBS (positive integer override; automatic CPU/RAM estimate by default)
  CONAN_PROFILE_HOST, CONAN_PROFILE_BUILD (names or paths; default: default)
EOF
}

while (($#)); do
    case "$1" in
        --tests)
            RUN_TESTS=true
            CONAN_WITH_TESTS=True
            ;;
        --shared)
            CONAN_SHARED=True
            ;;
        --cuda)
            CONAN_WITH_CUDA=True
            ;;
        --no-cuda)
            CONAN_WITH_CUDA=False
            ;;
        --collision)
            CONAN_WITH_COLLISION=True
            ;;
        --no-collision)
            CONAN_WITH_COLLISION=False
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ! command -v conan >/dev/null 2>&1; then
    echo "error: Conan 2 is required (python3 -m pip install --user conan)" >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "error: CMake is required" >&2
    exit 1
fi

if ! command -v "${PYTHON_EXECUTABLE}" >/dev/null 2>&1; then
    echo "error: Python executable not found: ${PYTHON_EXECUTABLE}" >&2
    exit 1
fi

# Resolve caller-relative paths before moving to the repository. CMake resolves
# an interpreter path against its build directory, and Conan activation scripts
# may change PATH later in this process.
PYTHON_EXECUTABLE="$("${PYTHON_EXECUTABLE}" -I -S -c 'import sys; print(sys.executable)')"
BUILD_DIR="$("${PYTHON_EXECUTABLE}" -I -S -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "${BUILD_DIR}")"

if [[ -n "${JOBS}" && ( ! "${JOBS}" =~ ^[0-9]+$ || "${JOBS}" =~ ^0+$ ) ]]; then
    echo "error: JOBS must be a positive integer" >&2
    exit 2
fi

cd "${REPOSITORY_DIR}"

if ! conan profile show -pr:h "${CONAN_PROFILE_HOST}" \
        -pr:b "${CONAN_PROFILE_BUILD}" >/dev/null 2>&1; then
    if [[ "${CONAN_PROFILE_HOST}" == default && "${CONAN_PROFILE_BUILD}" == default ]]; then
        echo "[1/4] Detecting the Conan default profile"
        conan profile detect
    else
        echo "error: cannot load Conan profiles: host=${CONAN_PROFILE_HOST}, build=${CONAN_PROFILE_BUILD}" >&2
        exit 1
    fi
else
    echo "[1/4] Using Conan profiles: host=${CONAN_PROFILE_HOST}, build=${CONAN_PROFILE_BUILD}"
fi

echo "[2/4] Installing dependencies"
HOLISTICMOTION_PYTHON_MINOR="$("${PYTHON_EXECUTABLE}" -I -S -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
HOLISTICMOTION_PYTHON_ABI="$("${PYTHON_EXECUTABLE}" -I -S "${REPOSITORY_DIR}/python/holistic_motion/_python_abi.py" --abi)"
echo "      Python=${HOLISTICMOTION_PYTHON_MINOR} (${HOLISTICMOTION_PYTHON_ABI}), CUDA=${CONAN_WITH_CUDA}, collision=${CONAN_WITH_COLLISION}, tests=${CONAN_WITH_TESTS}"
conan install . \
    -pr:h "${CONAN_PROFILE_HOST}" -pr:b "${CONAN_PROFILE_BUILD}" \
    --output-folder="${BUILD_DIR}" \
    --build=missing \
    -s build_type="${BUILD_TYPE}" \
    -o "&:shared=${CONAN_SHARED}" \
    -o "&:with_python=True" \
    -o "&:python_version=${HOLISTICMOTION_PYTHON_MINOR}" \
    -o "&:python_abi=${HOLISTICMOTION_PYTHON_ABI}" \
    -o "&:with_tests=${CONAN_WITH_TESTS}" \
    -o "&:with_cuda=${CONAN_WITH_CUDA}" \
    -o "&:with_collision=${CONAN_WITH_COLLISION}"

TOOLCHAIN="${BUILD_DIR}/build/${BUILD_TYPE}/generators/conan_toolchain.cmake"
if [[ ! -f "${TOOLCHAIN}" ]]; then
    echo "error: Conan toolchain was not generated at ${TOOLCHAIN}" >&2
    exit 1
fi

if [[ -z "${JOBS}" ]]; then
    JOBS="$("${PYTHON_EXECUTABLE}" -I -S "${REPOSITORY_DIR}/scripts/build_parallelism.py")"
fi
echo "[3/4] Configuring and building"
echo "      Parallel jobs=${JOBS} (override with JOBS)"
cmake -S . -B "${BUILD_DIR}/cmake" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DHOLISTICMOTION_PYTHON_VERSION="${HOLISTICMOTION_PYTHON_MINOR}" \
    -DHOLISTICMOTION_PYTHON_ABI="${HOLISTICMOTION_PYTHON_ABI}" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DPython3_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -DHOLISTICMOTION_BUILD_PYTHON=ON \
    -DHOLISTICMOTION_BUILD_TESTS="${RUN_TESTS}" \
    -DHOLISTICMOTION_ENABLE_CUDA="${CONAN_WITH_CUDA}" \
    -DHOLISTICMOTION_ENABLE_COLLISION="${CONAN_WITH_COLLISION}"
cmake --build "${BUILD_DIR}/cmake" --parallel "${JOBS}"

echo "[4/4] Installing into ${BUILD_DIR}/install"
cmake --install "${BUILD_DIR}/cmake" --prefix "${BUILD_DIR}/install"

CONAN_RUN_ENV="${BUILD_DIR}/build/${BUILD_TYPE}/generators/conanrun.sh"
if [[ -f "${CONAN_RUN_ENV}" &&
      ! -f "${BUILD_DIR}/install/.holistic_motion_local_runtime" ]]; then
    # Make Conan-managed shared libraries (notably Pinocchio and Coal)
    # available to the installed Python extension and test executables.
    # shellcheck disable=SC1090
    source "${CONAN_RUN_ENV}"
fi

HOLISTICMOTION_PURE_PYTHON=0 \
PYTHONPATH="${BUILD_DIR}/install${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_EXECUTABLE}" -c \
    "import holistic_motion as hm; print('Python import smoke test: OK'); print(hm.runtime_info()); print(f'FEP CUDA compiled/runtime: {hm.FEPKinematics.cuda_compiled}/{hm.FEPKinematics.cuda_available}'); assert hm.FEPKinematics.cuda_compiled is ${CONAN_WITH_CUDA}; assert hm.runtime_info().collision_compiled is ${CONAN_WITH_COLLISION}; assert hasattr(hm, 'CollisionModel') is ${CONAN_WITH_COLLISION}"

if [[ "${RUN_TESTS}" == true ]]; then
    ctest --test-dir "${BUILD_DIR}/cmake" --output-on-failure --no-tests=error
    echo "Running Python tests against the installed package"
    if ! "${PYTHON_EXECUTABLE}" -c "import pytest" >/dev/null 2>&1; then
        echo "error: --tests requires pytest" >&2
        exit 1
    fi
    HOLISTICMOTION_PYTEST_PLUGINS=()
    if "${PYTHON_EXECUTABLE}" -c \
        'import importlib.util; raise SystemExit(importlib.util.find_spec("hypothesis") is None)'; then
        HOLISTICMOTION_PYTEST_PLUGINS=(-p hypothesis.extra.pytestplugin)
    fi
    # Optional plugins from ROS or other environments can import incompatible
    # native runtimes before collection. Load only the project's property plugin.
    HOLISTICMOTION_PURE_PYTHON=0 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH="${BUILD_DIR}/install${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_EXECUTABLE}" -m pytest "${HOLISTICMOTION_PYTEST_PLUGINS[@]}" tests/python -q
fi

echo
HOLISTICMOTION_BUILD_HINT=""
if [[ "${BUILD_DIR}" != "${REPOSITORY_DIR}/build" || "${BUILD_TYPE}" != Release ]]; then
    printf -v HOLISTICMOTION_BUILD_HINT 'BUILD_DIR=%q BUILD_TYPE=%q ' "${BUILD_DIR}" "${BUILD_TYPE}"
fi
printf -v HOLISTICMOTION_PYTHON_HINT '%q' "$(command -v "${PYTHON_EXECUTABLE}")"
print_run_example() {
    printf '  %s./scripts/%s %s %s\n' "${HOLISTICMOTION_BUILD_HINT}" "$1" \
        "${HOLISTICMOTION_PYTHON_HINT}" "$2"
}
echo "Build completed. Activate the local install with:"
printf '  %ssource scripts/activate.sh\n' "${HOLISTICMOTION_BUILD_HINT}"
echo "Or run a command in the build environment with:"
print_run_example run.sh examples/python/planning/classic_plan_and_retime.py
print_run_example run.sh examples/python/trajectory/toppra_retiming.py
echo "Run a pure-Python/Pink toolkit without native Pinocchio ABI mixing with:"
print_run_example run-python-toolkit.sh 'examples/python/retargeting/pink_dual_arm.py --help'
print_run_example run-python-toolkit.sh 'examples/python/retargeting/classic_torso_first.py --help'
echo "Run the viewer with:"
print_run_example run.sh 'examples/python/visualization/rrt_robot_viser.py --urdf /path/to/robot.urdf'
echo "Generate trajectory diagnostics with:"
print_run_example run.sh 'python/examples/trajectory/plot_joint_trajectory.py --output build/trajectory_profiles.png'
if [[ "${CONAN_WITH_CUDA}" == True ]]; then
    echo "Run the FEP CUDA benchmark with:"
    print_run_example run.sh 'python/examples/fep_cuda_benchmark.py --urdf /path/to/robot.urdf'
fi
