#!/usr/bin/env bash

# Source this file after scripts/build.sh to use the local install tree:
#   source scripts/activate.sh

HOLISTICMOTION_REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HOLISTICMOTION_BUILD_DIR="${BUILD_DIR:-${HOLISTICMOTION_REPOSITORY_DIR}/build}"
HOLISTICMOTION_BUILD_TYPE="${BUILD_TYPE:-Release}"
HOLISTICMOTION_CONAN_RUN="${HOLISTICMOTION_BUILD_DIR}/build/${HOLISTICMOTION_BUILD_TYPE}/generators/conanrun.sh"

if [[ ! -f "${HOLISTICMOTION_CONAN_RUN}" ]]; then
    echo "error: build environment not found; run ./scripts/build.sh first" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${HOLISTICMOTION_CONAN_RUN}"
export PYTHONPATH="${HOLISTICMOTION_BUILD_DIR}/install${PYTHONPATH:+:${PYTHONPATH}}"
export CMAKE_PREFIX_PATH="${HOLISTICMOTION_BUILD_DIR}/install${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
unset HOLISTICMOTION_REPOSITORY_DIR HOLISTICMOTION_BUILD_DIR
unset HOLISTICMOTION_BUILD_TYPE HOLISTICMOTION_CONAN_RUN
