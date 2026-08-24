#!/usr/bin/env bash

set -euo pipefail

if (($# == 0)); then
    echo "usage: $0 command [args ...]" >&2
    exit 2
fi

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPOSITORY_DIR}/scripts/activate.sh"
exec "$@"
