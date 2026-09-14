"""Python interface for HolisticMotion.

Set ``HOLISTICMOTION_PURE_PYTHON=1`` for toolkits that intentionally do not
load the compiled extension, such as a pip-provided Pinocchio runtime that must
not share Conan's C++ Pinocchio libraries in the same process.
"""

import os as _os

from . import geometry as geometry

if _os.environ.get("HOLISTICMOTION_PURE_PYTHON") != "1":
    try:
        from ._holistic_motion import *
    except ModuleNotFoundError as error:
        if error.name != "holistic_motion._holistic_motion":
            raise
        # Support direct use from a CMake build tree, where the extension is a
        # top-level target rather than installed inside this package.
        from _holistic_motion import *

from .logging import (
    _install_native_bridge,
    auto_configure_debug_logging,
)
from .logging import (
    add_logger_handler as add_logger_handler,
)
from .logging import (
    get_logger as get_logger,
)
from .logging import (
    reset_logging as reset_logging,
)
from .logging import (
    setup_logging as setup_logging,
)

_install_native_bridge()
auto_configure_debug_logging()
