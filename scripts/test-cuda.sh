#!/usr/bin/env bash

# Actual-device validation; CPU-only CUDA compilation is covered separately in CI.
set -euo pipefail

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_BUILD_DIR="${CUDA_BUILD_DIR:-${BUILD_DIR:-${REPOSITORY_DIR}/build}/cmake}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
WITH_TORCH=false
while (($#)); do
    case "$1" in
        --torch) WITH_TORCH=true; shift ;;
        --build-dir|--python)
            if (($# < 2)); then
                echo "error: $1 requires a value" >&2
                exit 2
            fi
            if [[ "$1" == --build-dir ]]; then
                CUDA_BUILD_DIR="$2"
            else
                PYTHON_EXECUTABLE="$2"
            fi
            shift 2
            ;;
        --help)
            echo "usage: $0 [--torch] [--build-dir CMake-dir] [--python executable] [pytest arguments ...]"
            echo "Environment: CUDA_BUILD_DIR, PYTHON_EXECUTABLE"
            exit 0
            ;;
        --) shift; break ;;
        *) break ;;
    esac
done
if [[ ! -x "${CUDA_BUILD_DIR}/fep_batch_smoke_test" ]]; then
    echo "error: build with --cuda --tests, or select its CMake directory with --build-dir" >&2
    exit 2
fi
CUDA_BUILD_DIR="$(cd -- "${CUDA_BUILD_DIR}" && pwd)"
PYTHON_EXECUTABLE="$("${PYTHON_EXECUTABLE}" -I -S -c 'import sys; print(sys.executable)')"

export HOLISTICMOTION_PURE_PYTHON=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="${REPOSITORY_DIR}/python:${CUDA_BUILD_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPOSITORY_DIR}"

"${PYTHON_EXECUTABLE}" -c \
    'import holistic_motion as hm; assert hm.FEPKinematics.cuda_compiled, "CUDA was not compiled"; assert hm.FEPKinematics.cuda_available, "CUDA device unavailable"; print(hm.runtime_info())'
"${CUDA_BUILD_DIR}/fep_batch_smoke_test"
if [[ ! -x "${CUDA_BUILD_DIR}/fep_selection_smoke_test" ]]; then
    echo "error: rebuild CUDA tests to include the candidate-scoring check" >&2
    exit 2
fi
"${CUDA_BUILD_DIR}/fep_selection_smoke_test"
if [[ ! -x "${CUDA_BUILD_DIR}/concurrent_queries_smoke_test" ]]; then
    echo "error: rebuild CUDA tests to include concurrent query validation" >&2
    exit 2
fi
"${CUDA_BUILD_DIR}/concurrent_queries_smoke_test"
if [[ -x "${CUDA_BUILD_DIR}/cuda_workspace_smoke_test" ]]; then
    "${CUDA_BUILD_DIR}/cuda_workspace_smoke_test"
fi

tests=(tests/python/test_fep_batch.py)
if [[ "${WITH_TORCH}" == true ]]; then
    "${PYTHON_EXECUTABLE}" -c \
        'import torch; assert torch.cuda.is_available(), "Torch CUDA device unavailable"; print("Torch:", torch.__version__, "device:", torch.cuda.get_device_name()); print(torch.ones(1, device="cuda").sum().item())'
    tests+=(tests/python/test_toppra_torch*.py)
fi
exec "${PYTHON_EXECUTABLE}" -m pytest "${tests[@]}" "$@"
