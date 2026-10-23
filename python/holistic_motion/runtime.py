"""Inspect runtime boundaries without importing optional native dependencies.

Native core algorithms can share a process with Python Pinocchio. The optional
Conan collision extension loads on first use and requires a separate process
from an independently packaged Pinocchio/Coal/Boost runtime.
"""

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Optional

from ._python_abi import python_abi

__all__ = ["RuntimeInfo", "runtime_info"]

_import_lock = RLock()
_collision_import_attempted = False


def _build_config():
    for name in ("holistic_motion._build_config", "_holistic_motion_build_config"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    return None


def _native_module():
    return sys.modules.get("holistic_motion._holistic_motion") or sys.modules.get(
        "_holistic_motion"
    )


def _collision_module():
    return sys.modules.get(
        "holistic_motion._holistic_motion_collision"
    ) or sys.modules.get("_holistic_motion_collision")


def _verify_collision_extension():
    """Check the candidate binary before its PyInit registers shared C++ types."""
    expected = getattr(_native_module(), "_native_build_id", None)
    if expected is None:
        # Older coherent core/collision pairs predate the identity protocol.
        # New collision extensions also check their core during initialization.
        return
    message = (
        "HolisticMotion's core and collision extensions have different or "
        "unverifiable native build identities. Rebuild and install all native "
        "components together."
    )
    if not isinstance(expected, str) or not expected:
        raise ImportError(message)
    from importlib.machinery import EXTENSION_SUFFIXES
    from importlib.util import find_spec

    for name in (
        "holistic_motion._holistic_motion_collision",
        "_holistic_motion_collision",
    ):
        spec = find_spec(name)
        if spec is None:
            continue
        # Synthetic Python loaders used by embedders do not exchange native
        # C++ layouts. Actual extension files must implement the handshake.
        if spec.origin is None or not any(
            spec.origin.endswith(suffix) for suffix in EXTENSION_SUFFIXES
        ):
            return
        import ctypes

        try:
            library = ctypes.CDLL(spec.origin)
            identity = library.HolisticMotionCollisionBuildIdentity
            identity.argtypes = []
            identity.restype = ctypes.c_char_p
            observed = identity()
        except (OSError, AttributeError) as error:
            raise ImportError(message) from error
        if observed != expected.encode("utf-8"):
            raise ImportError(message)
        return


def _toolkit_only_build():
    return bool(getattr(_build_config(), "TOOLKIT_ONLY", False))


@dataclass(frozen=True)
class RuntimeInfo:
    """Loaded components and build capabilities, with no dependency imports."""

    pure_python: bool
    native_loaded: bool
    collision_compiled: Optional[bool]
    cuda_compiled: Optional[bool]
    native_pinocchio_version: str
    python_pinocchio_loaded: bool
    python_pinocchio_version: str
    native_python_version: str = ""
    native_python_abi: str = ""
    collision_loaded: bool = False


def runtime_info():
    """Report the current process; unknown build capabilities are ``None``."""
    native = _native_module()
    config = _build_config()
    pin = sys.modules.get("pinocchio")
    return RuntimeInfo(
        pure_python=(
            os.environ.get("HOLISTICMOTION_PURE_PYTHON") == "1"
            or bool(getattr(config, "TOOLKIT_ONLY", False))
        ),
        native_loaded=native is not None,
        collision_compiled=(
            bool(
                getattr(
                    native, "_collision_compiled", hasattr(native, "CollisionModel")
                )
            )
            if native is not None
            else getattr(config, "COLLISION_ENABLED", None)
        ),
        cuda_compiled=(
            bool(native.FEPKinematics.cuda_compiled)
            if native is not None and hasattr(native, "FEPKinematics")
            else getattr(config, "CUDA_ENABLED", None)
        ),
        native_pinocchio_version=getattr(config, "PINOCCHIO_VERSION", ""),
        python_pinocchio_loaded=pin is not None,
        python_pinocchio_version=str(getattr(pin, "__version__", "")),
        native_python_version=str(getattr(config, "PYTHON_VERSION", "")),
        native_python_abi=str(getattr(config, "PYTHON_ABI", "")),
        collision_loaded=(
            _collision_module() is not None
            or (native is not None and hasattr(native, "CollisionModel"))
        ),
    )


def _check_native_import():
    """Check before dlopen, including the Python-Pinocchio-first import order."""
    info = runtime_info()
    if info.native_python_version and info.native_python_version.split(".")[:2] != [
        str(sys.version_info.major),
        str(sys.version_info.minor),
    ]:
        raise ImportError(
            f"HolisticMotion's native extension was built for Python "
            f"{info.native_python_version}, but this interpreter is Python "
            f"{sys.version_info.major}.{sys.version_info.minor}. Select a matching "
            "Conan python_version and Python3_EXECUTABLE, or install the pure Python toolkit."
        )
    if info.native_python_abi and info.native_python_abi != python_abi():
        raise ImportError(
            f"HolisticMotion's native extension was built for Python ABI "
            f"{info.native_python_abi}, but this interpreter uses {python_abi()}. "
            "Select a matching Conan python_abi and Python3_EXECUTABLE, or "
            "install the pure Python toolkit."
        )
    if bool(getattr(_build_config(), "COLLISION_SEPARATE", False)):
        return
    independent_modules = [
        name
        for name in ("pinocchio", "coal", "hppfcl", "eigenpy")
        if name in sys.modules
    ]
    if independent_modules and info.collision_compiled is not False:
        python_runtime = (
            "Python Pinocchio"
            if info.python_pinocchio_loaded
            else f"Python collision dependency {independent_modules[0]!r}"
        )
        raise ImportError(
            f"{python_runtime} is already loaded. Loading the HolisticMotion "
            "Conan collision runtime in this process is unsupported. Set "
            "HOLISTICMOTION_PURE_PYTHON=1 before importing holistic_motion for "
            "retargeting, or use a build with --no-collision to combine Python "
            "Pinocchio with native core algorithms."
        )


def _check_python_pinocchio_import():
    """Check before optional retargeting loads Python Pinocchio."""
    info = runtime_info()
    if info.collision_loaded or _collision_import_attempted:
        raise ImportError(
            "HolisticMotion's Conan collision runtime is already loaded. "
            "Python Pinocchio retargeting requires a separate process with "
            "HOLISTICMOTION_PURE_PYTHON=1 set before importing holistic_motion, "
            "or a native core build with --no-collision. Use "
            "scripts/run-python-toolkit.sh for the separate-process toolkit."
        )


def _check_collision_import():
    """Check before dlopen of the separately packaged collision extension."""
    modules = [
        name
        for name in ("pinocchio", "coal", "hppfcl", "eigenpy")
        if name in sys.modules
    ]
    if modules:
        raise ImportError(
            f"Python collision dependency {modules[0]!r} is already loaded. "
            "Loading the HolisticMotion Conan collision runtime in this process "
            "is unsupported; keep using the native core and Python Pinocchio, "
            "or run native collision queries in a separate process."
        )


@contextmanager
def _collision_import():
    """Serialize the public native-collision and Python-robotics loaders."""
    global _collision_import_attempted
    with _import_lock:
        _check_collision_import()
        # A failed extension initialization may retain native libraries. Do not
        # subsequently load an independent Pinocchio runtime in this process.
        attempted_before = _collision_import_attempted
        _collision_import_attempted = True
        try:
            yield
        except ModuleNotFoundError as error:
            if error.name in (
                "holistic_motion._holistic_motion_collision",
                "_holistic_motion_collision",
            ):
                # Neither installed nor build-tree extension exists. No native
                # collision dependency has been loaded by this attempt.
                _collision_import_attempted = attempted_before
            raise


@contextmanager
def _python_pinocchio_import():
    with _import_lock:
        _check_python_pinocchio_import()
        yield
