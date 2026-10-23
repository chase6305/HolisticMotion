"""Exercise the actual split extensions in fresh interpreter processes."""

import os
import subprocess
import sys

import pytest
from holistic_motion import runtime_info


def run_fresh(command):
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def collision_build():
    info = runtime_info()
    if not info.native_loaded or not info.collision_compiled:
        pytest.skip("requires a native collision build")


def test_core_import_does_not_load_collision_libraries(collision_build):
    run_fresh("""
import sys
from pathlib import Path
import holistic_motion as hm
assert hm.runtime_info().collision_compiled
assert not hm.runtime_info().collision_loaded
assert 'CollisionModel' in dir(hm)
assert 'CollisionModel' in hm.__all__
assert 'SphereDistanceGradientResult' in dir(hm)
assert 'SphereDistanceGradientResult' in hm.__all__
assert not hm.runtime_info().collision_loaded
assert not any('pinocchio' in key or key in ('coal', 'hppfcl', 'eigenpy') for key in sys.modules)
maps = Path('/proc/self/maps')
if maps.exists():
    paths = [line.split()[-1] for line in maps.read_text().splitlines()]
    assert not any(Path(path).name.startswith(('libpinocchio', 'libcoal', 'libhpp-fcl', 'eigenpy')) for path in paths)
# Noncollision planning and its Python callback remain available.
planner = hm.SamplingPlanner([-1.0], [1.0], lambda q: True)
assert planner.plan([0.0], [0.3]).success
assert not hm.runtime_info().collision_loaded
""")


@pytest.mark.parametrize("direct", [False, True])
def test_collision_build_mismatch_preserves_uninitialized_core_factories(
    collision_build, direct
):
    run_fresh(
        """
import importlib
import importlib.util
import sys
import holistic_motion as hm
from holistic_motion import runtime
core = runtime._native_module()
expected = core._native_build_id
core._native_build_id = 'stale:Release'
direct = """
        + repr(direct)
        + """
try:
    if direct:
        names = ('holistic_motion._holistic_motion_collision', '_holistic_motion_collision')
        name = next(name for name in names if importlib.util.find_spec(name) is not None)
        importlib.import_module(name)
    else:
        hm.CollisionModel
except ImportError as error:
    details = str(error) + str(error.__cause__)
    assert 'native build identities' in details, details
else:
    raise AssertionError('mismatched native build was accepted')
assert runtime._collision_module() is None
assert 'CollisionModel' not in vars(hm)
assert callable(core.SamplingPlanner.from_collision_joints)
core._native_build_id = expected
assert hm.CollisionModel.__name__ == 'CollisionModel'
assert runtime._collision_module()._native_build_id == expected
assert hm.SamplingPlanner([-1.0], [1.0]).plan([-.8], [.8], release_gil=True).success
"""
    )


@pytest.mark.parametrize(
    "environment_key",
    ["HOLISTICMOTION_TEST_LEGACY_COLLISION", "HOLISTICMOTION_TEST_MISMATCHED_COLLISION"],
)
def test_incompatible_collision_binary_is_rejected_before_cpp_registration(
    collision_build, environment_key
):
    if os.environ.get(environment_key) is None:
        pytest.skip(f"set {environment_key} to supply an incompatible binary")
    run_fresh(
        """
import importlib.abc
import importlib.util
import os
import sys
import holistic_motion as hm
from holistic_motion import runtime
assert runtime._native_module()._native_build_id
environment_key = """
        + repr(environment_key)
        + """
class IncompatibleCollision(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'holistic_motion._holistic_motion_collision':
            return importlib.util.spec_from_file_location(
                fullname, os.environ[environment_key])
sys.meta_path.insert(0, IncompatibleCollision())
if environment_key.endswith('MISMATCHED_COLLISION'):
    import ctypes
    library = ctypes.CDLL(os.environ[environment_key])
    identity = library.HolisticMotionCollisionBuildIdentity
    identity.restype = ctypes.c_char_p
    assert identity() != runtime._native_module()._native_build_id.encode()
try:
    hm.CollisionModel
except ImportError as error:
    assert 'native build identities' in str(error), str(error)
else:
    raise AssertionError('incompatible collision binary entered C++ type registration')
assert runtime._collision_module() is None
assert hm.SamplingPlanner([-1.0], [1.0]).plan([-.8], [.8]).success
"""
    )


@pytest.mark.parametrize(
    "syntax", ["attribute", "from", "star", "planner", "optimizer", "threads"]
)
def test_first_collision_use_loads_the_optional_extension(collision_build, syntax):
    run_fresh(
        """
import sys
from concurrent.futures import ThreadPoolExecutor
import holistic_motion as hm
assert not hm.runtime_info().collision_loaded
factory = hm.SamplingPlanner.from_collision_joints
kind = """
        + repr(syntax)
        + """
if kind == 'attribute':
    model_type = hm.CollisionModel
elif kind == 'from':
    from holistic_motion import CollisionModel as model_type
elif kind == 'star':
    from holistic_motion import *
    model_type = CollisionModel
elif kind in ('planner', 'optimizer'):
    deferred = factory if kind == 'planner' else hm.PathOptimizer.from_collision_joints
    try:
        deferred(unknown_keyword=True)
    except TypeError:
        pass
    else:
        raise AssertionError('invalid factory call accepted')
    model_type = hm.CollisionModel
else:
    def get_type(_):
        return hm.CollisionModel
    with ThreadPoolExecutor(8) as workers:
        types = list(workers.map(get_type, range(32)))
    assert all(value is types[0] for value in types)
    model_type = types[0]
assert hm.runtime_info().collision_loaded
assert model_type is hm.CollisionModel
module = sys.modules.get('holistic_motion._holistic_motion_collision') or sys.modules['_holistic_motion_collision']
assert module.SamplingPlanner is hm.SamplingPlanner
assert module.PathOptimizer is hm.PathOptimizer
# Native result classes must remain public after splitting the extension.
# Inspect the binding's exports rather than just iterating the loader's list,
# which would miss a name accidentally omitted from that list.
for name, value in vars(module).items():
    if not name.startswith('_') and isinstance(value, type):
        assert name in hm.__all__, name
        assert getattr(hm, name) is value, name
for name in hm._COLLISION_EXPORTS:
    assert getattr(hm, name) is getattr(module, name)
# A factory reference captured before loading must still forward to typed overloads.
try:
    factory(unknown_keyword=True)
except TypeError:
    pass
else:
    raise AssertionError('invalid deferred call accepted')
assert hm.SamplingPlanner([-1.0], [1.0], lambda q: True).plan([0.0], [0.2]).success
"""
    )


def test_pure_mode_does_not_expose_collision_types():
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import holistic_motion as hm
assert not hm.runtime_info().collision_loaded
assert not hm.runtime_info().native_loaded
assert 'CollisionModel' not in dir(hm)
assert 'CollisionModel' not in hm.__all__
assert not hasattr(hm, 'CollisionModel')
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_collision_cache_waits_for_module_initialization(collision_build, tmp_path):
    delayed = tmp_path / "delayed_collision.py"
    delayed.write_text(
        "import holistic_motion as hm\n"
        "hm._test_collision_started.set()\n"
        "assert hm._test_collision_release.wait(5)\n"
        "for name in hm._COLLISION_EXPORTS:\n"
        "    globals()[name] = type(name, (), {})\n"
    )
    run_fresh(
        """
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
import holistic_motion as hm
hm._test_collision_started = Event()
hm._test_collision_release = Event()
class DelayedModule:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('holistic_motion._holistic_motion_collision', '_holistic_motion_collision'):
            return importlib.util.spec_from_file_location(fullname, """
        + repr(str(delayed))
        + """)
sys.meta_path.insert(0, DelayedModule())
def get_type():
    return hm.CollisionModel
with ThreadPoolExecutor(2) as workers:
    first = workers.submit(get_type)
    try:
        assert hm._test_collision_started.wait(5)
        # Python's importer has already published the partially initialized
        # module. Another public access must wait for the initialization lock.
        assert 'holistic_motion._holistic_motion_collision' in sys.modules
        second = workers.submit(get_type)
        try:
            second.result(timeout=0.2)
        except TimeoutError:
            pass
        else:
            raise AssertionError('a partial collision module escaped the loader')
    finally:
        hm._test_collision_release.set()
    assert first.result(timeout=5) is second.result(timeout=5) is hm.CollisionModel
"""
    )
