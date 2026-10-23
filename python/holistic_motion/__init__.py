"""Python interface for HolisticMotion.

Core algorithms load independently of the optional collision extension. Accessing
a collision type or invoking a collision planning factory loads that extension.
Set ``HOLISTICMOTION_PURE_PYTHON=1`` to use only the Python toolkit.
"""

import os as _os

from .runtime import _toolkit_only_build
from .runtime import runtime_info as runtime_info

_COLLISION_EXPORTS = (
    "CollisionSphere",
    "SpherePairInfo",
    "SphereDistanceResult",
    "SphereDistanceGradientResult",
    "CollisionPairInfo",
    "CollisionReport",
    "CollisionResult",
    "CollisionModel",
    "SphereCollisionModel",
)


def _load_collision():
    import importlib

    from .runtime import (
        _collision_import,
        _collision_module,
        _verify_collision_extension,
    )

    with _collision_import():
        # A multi-phase/Python module can be in sys.modules before its body
        # finishes. Read the cache under the same lock as initialization.
        collision = _collision_module()
        if collision is None:
            _verify_collision_extension()
            try:
                collision = importlib.import_module(
                    "._holistic_motion_collision", __name__
                )
            except ModuleNotFoundError as error:
                if error.name != "holistic_motion._holistic_motion_collision":
                    raise
                collision = importlib.import_module("_holistic_motion_collision")
        for name in _COLLISION_EXPORTS:
            globals()[name] = getattr(collision, name)
        return collision


def __getattr__(name):
    if name == "geometry":
        from importlib import import_module

        module = import_module(".geometry", __name__)
        globals()[name] = module
        return module
    if name in _COLLISION_EXPORTS:
        info = runtime_info()
        if info.native_loaded and info.collision_compiled:
            return getattr(_load_collision(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = set(globals()) | {"geometry"}
    info = runtime_info()
    if info.native_loaded and info.collision_compiled:
        names.update(_COLLISION_EXPORTS)
    return sorted(names)


if _os.environ.get("HOLISTICMOTION_PURE_PYTHON") != "1" and not _toolkit_only_build():
    from .runtime import _check_native_import

    _check_native_import()
    try:
        from ._holistic_motion import *  # noqa: F403
    except ModuleNotFoundError as error:
        if error.name != "holistic_motion._holistic_motion":
            raise
        # Support direct use from a CMake build tree, where the extension is a
        # top-level target rather than installed inside this package.
        from _holistic_motion import *  # noqa: F403

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

__all__ = [name for name in globals() if not name.startswith("_")]
if "geometry" not in __all__:
    __all__.append("geometry")
if runtime_info().native_loaded and runtime_info().collision_compiled:
    __all__ += [name for name in _COLLISION_EXPORTS if name not in __all__]
