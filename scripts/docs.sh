#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

for command_name in cmake doxygen sphinx-build; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing documentation tool: ${command_name}" >&2
        exit 1
    fi
done

cmake -E rm -rf "${repository_root}/docs/_build"
cmake -E make_directory "${repository_root}/docs/_build"
cmake -E chdir "${repository_root}/docs" make html
cmake -E copy "${repository_root}/docs/index.html" \
    "${repository_root}/docs/_build/html/index.html"
