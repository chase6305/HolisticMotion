"""Optional tests obey the same native-runtime boundary as applications."""

import importlib

import pytest
from holistic_motion import runtime_info


def skip_native_collision_runtime():
    info = runtime_info()
    if info.collision_loaded:
        pytest.skip(
            "Python Pinocchio requires a process without the Conan collision runtime; "
            "run the separate retargeting test job"
        )


def python_pinocchio_or_skip(module="pinocchio"):
    skip_native_collision_runtime()
    return optional_module_or_skip(module)


def optional_module_or_skip(module):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        pytest.skip(f"optional dependency {module} is unavailable: {error}")


def tracemalloc_or_skip():
    try:
        return importlib.import_module("tracemalloc")
    except ImportError:
        pytest.skip("this interpreter does not provide the tracemalloc backend")
