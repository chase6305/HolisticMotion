"""Development-tree import helper shared by repository examples."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def import_holistic_motion():
    repository = Path(__file__).resolve().parents[2]
    candidates = []
    explicit = os.environ.get("HOLISTICMOTION_PYTHON_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    # Load the source-tree package first, then let its fallback import resolve
    # the freshly built top-level extension. An older local install must not
    # shadow a just-compiled development build.
    candidates.extend([
        repository / "python",
        repository / "build" / "cmake",
        repository / "build" / "install",
    ])
    # Insert in reverse so the documented order remains the actual import
    # priority. An explicit build is never shadowed by repository/build.
    for candidate in reversed(candidates):
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
    try:
        return importlib.import_module("holistic_motion")
    except ModuleNotFoundError as error:
        if error.name not in {
            "holistic_motion",
            "holistic_motion._holistic_motion",
            "_holistic_motion",
        }:
            raise
        raise SystemExit(
            "HolisticMotion Python bindings are not built. Run:\n"
            "  ./scripts/build.sh\n"
            "Then rerun this example."
        ) from error
