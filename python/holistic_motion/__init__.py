"""Python interface for HolisticMotion.

Set ``HOLISTICMOTION_PURE_PYTHON=1`` for toolkits that intentionally do not
load the compiled extension, such as a pip-provided Pinocchio runtime that must
not share Conan's C++ Pinocchio libraries in the same process.
"""

import os as _os

from . import geometry

if _os.environ.get("HOLISTICMOTION_PURE_PYTHON") != "1":
    try:
        from ._holistic_motion import *
    except ModuleNotFoundError as error:
        if error.name != "holistic_motion._holistic_motion":
            raise
        # Support direct use from a CMake build tree, where the extension is a
        # top-level target rather than installed inside this package.
        from _holistic_motion import *
