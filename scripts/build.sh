#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPOSITORY_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
RUN_TESTS=false
CONAN_WITH_TESTS=False
CONAN_SHARED=False
CONAN_WITH_CUDA=True
CONAN_WITH_COLLISION=True

usage() {
    cat <<EOF
Usage: $0 [options]

Builds and installs HolisticMotion with Python, CUDA, and collision support.

Options:
  --tests          Build and run the C++ and Python tests
  --shared         Build the core library as a shared library
  --no-cuda        Disable the CUDA batch backends
  --no-collision   Disable the Pinocchio/Coal collision component
  --cuda           Enable CUDA (kept for backward compatibility; default)
  --collision      Enable collision (kept for backward compatibility; default)
  -h, --help       Show this help

Environment:
  BUILD_DIR, BUILD_TYPE, JOBS, PYTHON_EXECUTABLE
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

cd "${REPOSITORY_DIR}"

if ! conan profile show -pr default >/dev/null 2>&1; then
    echo "[1/4] Detecting the Conan default profile"
    conan profile detect
else
    echo "[1/4] Using the existing Conan default profile"
fi

echo "[2/4] Installing dependencies"
echo "      CUDA=${CONAN_WITH_CUDA}, collision=${CONAN_WITH_COLLISION}, tests=${CONAN_WITH_TESTS}"
conan install . \
    --output-folder="${BUILD_DIR}" \
    --build=missing \
    -s build_type="${BUILD_TYPE}" \
    -o "&:shared=${CONAN_SHARED}" \
    -o "&:with_python=True" \
    -o "&:with_tests=${CONAN_WITH_TESTS}" \
    -o "&:with_cuda=${CONAN_WITH_CUDA}" \
    -o "&:with_collision=${CONAN_WITH_COLLISION}"

TOOLCHAIN="${BUILD_DIR}/build/${BUILD_TYPE}/generators/conan_toolchain.cmake"
if [[ ! -f "${TOOLCHAIN}" ]]; then
    echo "error: Conan toolchain was not generated at ${TOOLCHAIN}" >&2
    exit 1
fi

echo "[3/4] Configuring and building"
cmake -S . -B "${BUILD_DIR}/cmake" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DHOLISTICMOTION_BUILD_PYTHON=ON \
    -DHOLISTICMOTION_BUILD_TESTS="${RUN_TESTS}" \
    -DHOLISTICMOTION_ENABLE_CUDA="${CONAN_WITH_CUDA}" \
    -DHOLISTICMOTION_ENABLE_COLLISION="${CONAN_WITH_COLLISION}"
cmake --build "${BUILD_DIR}/cmake" --parallel "${JOBS}"

echo "[4/4] Installing into ${BUILD_DIR}/install"
cmake --install "${BUILD_DIR}/cmake" --prefix "${BUILD_DIR}/install"

CONAN_RUN_ENV="${BUILD_DIR}/build/${BUILD_TYPE}/generators/conanrun.sh"
if [[ -f "${CONAN_RUN_ENV}" ]]; then
    # Make Conan-managed shared libraries (notably Pinocchio and Coal)
    # available to the installed Python extension and test executables.
    # shellcheck disable=SC1090
    source "${CONAN_RUN_ENV}"
fi

PYTHONPATH="${BUILD_DIR}/install${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_EXECUTABLE}" -c \
    "import holistic_motion as hm; print('Python import smoke test: OK'); print(f'FEP CUDA compiled/runtime: {hm.FEPKinematics.cuda_compiled}/{hm.FEPKinematics.cuda_available}'); assert not ${CONAN_WITH_CUDA} or hm.FEPKinematics.cuda_compiled"

if [[ "${RUN_TESTS}" == true ]]; then
    ctest --test-dir "${BUILD_DIR}/cmake" --output-on-failure
    echo "Running Python tests against the installed package"
    if ! "${PYTHON_EXECUTABLE}" -c "import pytest" >/dev/null 2>&1; then
        echo "error: --tests requires pytest" >&2
        exit 1
    fi
    PYTHONPATH="${BUILD_DIR}/install${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_EXECUTABLE}" -m pytest tests/python -q
fi

echo
echo "Build completed. Activate the local install with:"
echo "  source scripts/activate.sh"
echo "Or run a command in the build environment with:"
echo "  ./scripts/run.sh python3 examples/python/trajectory/toppra_retiming.py"
echo "Run a pure-Python/Pink toolkit without native Pinocchio ABI mixing with:"
echo "  ./scripts/run-python-toolkit.sh python3 examples/python/retargeting/pink_dual_arm.py --help"
echo "Run the viewer with:"
echo "  ./scripts/run.sh python3 python/examples/viser_dual_arm_line.py --autoplay"
echo "Generate trajectory diagnostics with:"
echo "  ./scripts/run.sh python3 python/examples/trajectory/plot_joint_trajectory.py --output build/trajectory_profiles.png"
if [[ "${CONAN_WITH_CUDA}" == True ]]; then
    echo "Run the FEP CUDA benchmark with:"
    echo "  ./scripts/run.sh python3 python/examples/fep_cuda_benchmark.py --urdf /path/to/robot.urdf"
fi
