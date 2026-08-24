#!/usr/bin/env bash

set -euo pipefail

if (($# == 0)); then
    echo "usage: $0 python-command [args ...]" >&2
    exit 2
fi

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPOSITORY_DIR}/build}"
export HOLISTICMOTION_PURE_PYTHON=1
export PYTHONPATH="${BUILD_DIR}/install${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
