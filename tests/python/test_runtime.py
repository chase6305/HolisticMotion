"""Runtime checks run before either independent Pinocchio library is loaded."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from holistic_motion import runtime


@pytest.fixture
def isolated_runtime(monkeypatch):
    monkeypatch.delitem(sys.modules, "holistic_motion._holistic_motion", raising=False)
    monkeypatch.delitem(sys.modules, "_holistic_motion", raising=False)
    monkeypatch.delitem(
        sys.modules, "holistic_motion._holistic_motion_collision", raising=False
    )
    monkeypatch.delitem(sys.modules, "_holistic_motion_collision", raising=False)
    monkeypatch.setattr(runtime, "_collision_import_attempted", False)
    monkeypatch.delitem(sys.modules, "pinocchio", raising=False)
    for name in ("coal", "hppfcl", "eigenpy"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(
        runtime,
        "_build_config",
        lambda: SimpleNamespace(
            COLLISION_ENABLED=True, CUDA_ENABLED=False, PINOCCHIO_VERSION="3.8.0"
        ),
    )


def test_runtime_inspection_does_not_import_optional_libraries(isolated_runtime):
    info = runtime.runtime_info()
    assert not info.native_loaded
    assert not info.python_pinocchio_loaded
    assert info.collision_compiled
    assert not info.cuda_compiled
    assert info.native_pinocchio_version == "3.8.0"
    assert "pinocchio" not in sys.modules
    assert "holistic_motion._holistic_motion" not in sys.modules


def test_native_import_rejects_a_different_python_minor(isolated_runtime, monkeypatch):
    version = f"{sys.version_info.major}.{sys.version_info.minor + 1}.0"
    monkeypatch.setattr(
        runtime,
        "_build_config",
        lambda: SimpleNamespace(PYTHON_VERSION=version, COLLISION_ENABLED=False),
    )
    assert runtime.runtime_info().native_python_version == version
    with pytest.raises(ImportError, match="built for Python.*Conan python_version"):
        runtime._check_native_import()
    assert "holistic_motion._holistic_motion" not in sys.modules


def test_native_import_accepts_patch_releases_of_the_same_python_minor(
    isolated_runtime, monkeypatch
):
    monkeypatch.setattr(
        runtime,
        "_build_config",
        lambda: SimpleNamespace(
            PYTHON_VERSION=f"{sys.version_info.major}.{sys.version_info.minor}.999",
            COLLISION_ENABLED=False,
        ),
    )
    runtime._check_native_import()


def test_native_import_rejects_a_different_interpreter_abi(
    isolated_runtime, monkeypatch
):
    monkeypatch.setattr(
        runtime,
        "_build_config",
        lambda: SimpleNamespace(PYTHON_ABI="mismatched-abi", COLLISION_ENABLED=False),
    )
    assert runtime.runtime_info().native_python_abi == "mismatched-abi"
    with pytest.raises(ImportError, match="built for Python ABI.*Conan python_abi"):
        runtime._check_native_import()
    assert "holistic_motion._holistic_motion" not in sys.modules


@pytest.mark.parametrize(
    "debug,free,trace,expected",
    [
        (0, 0, 0, "cpython"),
        (1, 0, 0, "cpython-d"),
        (0, 1, 0, "cpython-t"),
        (1, 1, 0, "cpython-dt"),
        (0, 0, 1, "cpython-tracerefs"),
        ("0", "0", "0", "cpython"),
    ],
)
def test_python_abi_distinguishes_cpython_layout_flags(
    monkeypatch, debug, free, trace, expected
):
    from holistic_motion import _python_abi

    monkeypatch.setattr(
        _python_abi,
        "sys",
        SimpleNamespace(implementation=SimpleNamespace(name="cpython")),
    )
    flags = {"Py_DEBUG": debug, "Py_GIL_DISABLED": free, "Py_TRACE_REFS": trace}
    monkeypatch.setattr(_python_abi.sysconfig, "get_config_var", flags.get)
    assert _python_abi.python_abi() == expected


def test_package_checks_python_version_before_loading_extension(tmp_path):
    source = Path(__file__).resolve().parents[2] / "python" / "holistic_motion"
    package = tmp_path / "holistic_motion"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__"))
    version = f"{sys.version_info.major}.{sys.version_info.minor + 1}.0"
    (package / "_build_config.py").write_text(
        f'PYTHON_VERSION = "{version}"\nTOOLKIT_ONLY = False\nCOLLISION_ENABLED = False\n'
    )
    (package / "_holistic_motion.py").write_text(
        'raise AssertionError("mismatched extension was loaded")\n'
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    environment["HOLISTICMOTION_PURE_PYTHON"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", "import holistic_motion"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "built for Python" in result.stderr
    assert "Conan python_version" in result.stderr
    assert "mismatched extension was loaded" not in result.stderr


def test_native_import_refuses_python_pinocchio_even_with_matching_version(
    isolated_runtime, monkeypatch
):
    monkeypatch.setitem(sys.modules, "pinocchio", SimpleNamespace(__version__="3.8.0"))
    with pytest.raises(ImportError, match="already loaded.*unsupported"):
        runtime._check_native_import()
    assert "holistic_motion._holistic_motion" not in sys.modules


def test_python_pinocchio_import_refuses_native_collision(
    isolated_runtime, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "holistic_motion._holistic_motion",
        SimpleNamespace(CollisionModel=object()),
    )
    from holistic_motion.kit.retargeting.dex_hand import _load_dex
    from holistic_motion.kit.retargeting.pinocchio_solver import _load_pinocchio

    for loader in (_load_pinocchio, _load_dex):
        with pytest.raises(ImportError, match="separate process"):
            loader()
        assert "pinocchio" not in sys.modules


@pytest.mark.parametrize("module", ["coal", "hppfcl", "eigenpy"])
@pytest.mark.parametrize("collision", [False, True])
def test_native_import_checks_independent_collision_dependencies(
    isolated_runtime, monkeypatch, module, collision
):
    monkeypatch.setattr(
        runtime, "_build_config", lambda: SimpleNamespace(COLLISION_ENABLED=collision)
    )
    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    if collision:
        with pytest.raises(ImportError, match=f"{module}.*already loaded.*unsupported"):
            runtime._check_native_import()
    else:
        runtime._check_native_import()
    assert "pinocchio" not in sys.modules
    assert "holistic_motion._holistic_motion" not in sys.modules


def test_core_only_runtime_allows_python_pinocchio(isolated_runtime, monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_build_config",
        lambda: SimpleNamespace(
            COLLISION_ENABLED=False, CUDA_ENABLED=False, PINOCCHIO_VERSION=""
        ),
    )
    pin = SimpleNamespace(__version__="4.1.0")
    monkeypatch.setitem(sys.modules, "pinocchio", pin)
    runtime._check_native_import()
    monkeypatch.setitem(
        sys.modules, "holistic_motion._holistic_motion", SimpleNamespace()
    )
    runtime._check_python_pinocchio_import()
    assert runtime.runtime_info().python_pinocchio_version == "4.1.0"


@pytest.mark.parametrize("module", ["pinocchio", "coal", "hppfcl", "eigenpy"])
def test_separate_collision_allows_core_but_checks_collision_before_import(
    isolated_runtime, monkeypatch, module
):
    monkeypatch.setattr(
        runtime,
        "_build_config",
        lambda: SimpleNamespace(COLLISION_ENABLED=True, COLLISION_SEPARATE=True),
    )
    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    runtime._check_native_import()
    runtime._check_python_pinocchio_import()
    with pytest.raises(ImportError, match="already loaded.*unsupported"):
        with runtime._collision_import():
            pytest.fail("collision initialization must not start")
    assert not runtime._collision_import_attempted


@pytest.mark.parametrize(
    "missing",
    ["holistic_motion._holistic_motion_collision", "_holistic_motion_collision"],
)
def test_missing_collision_extension_does_not_block_python_pinocchio(
    isolated_runtime, missing
):
    with pytest.raises(ModuleNotFoundError):
        with runtime._collision_import():
            raise ModuleNotFoundError(name=missing)
    assert not runtime._collision_import_attempted
    runtime._check_python_pinocchio_import()


@pytest.mark.parametrize(
    "error", [ImportError("dlopen failed"), ModuleNotFoundError(name="coal")]
)
def test_failed_collision_initialization_retains_runtime_boundary(
    isolated_runtime, error
):
    with pytest.raises(type(error)):
        with runtime._collision_import():
            raise error
    assert runtime._collision_import_attempted
    with pytest.raises(ImportError, match="separate process"):
        runtime._check_python_pinocchio_import()


def test_missing_collision_retry_does_not_clear_an_earlier_native_attempt(
    isolated_runtime,
):
    with runtime._collision_import():
        pass
    with pytest.raises(ModuleNotFoundError):
        with runtime._collision_import():
            raise ModuleNotFoundError(name="_holistic_motion_collision")
    assert runtime._collision_import_attempted


@pytest.mark.parametrize("first", ["collision", "pinocchio"])
def test_concurrent_optional_loaders_serialize_before_import(isolated_runtime, first):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered, release, waiting = Event(), Event(), Event()
    first_loader = (
        runtime._collision_import
        if first == "collision"
        else runtime._python_pinocchio_import
    )
    second_loader = (
        runtime._python_pinocchio_import
        if first == "collision"
        else runtime._collision_import
    )

    def load_first():
        with first_loader():
            entered.set()
            assert release.wait(5)
            if first == "pinocchio":
                sys.modules["pinocchio"] = SimpleNamespace()

    def load_second():
        waiting.set()
        with pytest.raises(ImportError):
            with second_loader():
                pytest.fail("incompatible import was allowed")

    try:
        with ThreadPoolExecutor(2) as workers:
            started = workers.submit(load_first)
            assert entered.wait(5)
            blocked = workers.submit(load_second)
            assert waiting.wait(5)
            assert not blocked.done()
            release.set()
            started.result(timeout=5)
            blocked.result(timeout=5)
    finally:
        release.set()
        if first == "pinocchio":
            sys.modules.pop("pinocchio", None)


def test_missing_build_metadata_is_reported_without_loading_extension(
    isolated_runtime, monkeypatch
):
    monkeypatch.setattr(runtime, "_build_config", lambda: None)
    info = runtime.runtime_info()
    assert info.collision_compiled is None
    assert info.cuda_compiled is None
    assert info.native_pinocchio_version == ""


def test_toolkit_build_selects_pure_mode_without_environment_variable(
    isolated_runtime, monkeypatch
):
    monkeypatch.delenv("HOLISTICMOTION_PURE_PYTHON", raising=False)
    monkeypatch.setattr(
        runtime,
        "_build_config",
        lambda: SimpleNamespace(
            TOOLKIT_ONLY=True,
            COLLISION_ENABLED=False,
            CUDA_ENABLED=False,
            PINOCCHIO_VERSION="",
        ),
    )
    assert runtime.runtime_info().pure_python
    assert runtime._toolkit_only_build()


def test_pure_python_inspection_in_fresh_process():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "1"
    environment["PYTHONPATH"] = str(repository / "python")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import holistic_motion as hm
info = hm.runtime_info()
assert info.pure_python and not info.native_loaded
assert 'pinocchio' not in sys.modules
assert 'torch' not in sys.modules
assert not hasattr(hm, 'Robot')
""",
        ],
        env=environment,
        check=True,
    )


@pytest.mark.parametrize("order", ["pin_first", "native_first"])
def test_native_collision_import_boundary_in_fresh_process(order):
    info = runtime.runtime_info()
    if not info.native_loaded or info.collision_compiled is not True:
        pytest.skip("requires a native collision build")
    if order == "pin_first" and importlib.util.find_spec("pinocchio") is None:
        pytest.skip("Python Pinocchio is unavailable")
    command = """
import sys
if sys.argv[1] == 'pin_first':
    import pinocchio
    import holistic_motion as hm
    assert hm.runtime_info().native_loaded
    assert not hm.runtime_info().collision_loaded
    try:
        hm.CollisionModel
    except ImportError as error:
        assert 'already loaded' in str(error) and 'unsupported' in str(error)
    else:
        raise AssertionError('incompatible collision import was allowed')
    assert not hm.runtime_info().collision_loaded
    assert '_holistic_motion_collision' not in sys.modules
    assert 'holistic_motion._holistic_motion_collision' not in sys.modules
else:
    import holistic_motion as hm
    from holistic_motion.kit.retargeting.pinocchio_solver import _load_pinocchio
    from holistic_motion.kit.retargeting.dex_hand import _load_dex
    assert hm.runtime_info().collision_compiled
    assert not hm.runtime_info().collision_loaded
    hm.CollisionModel
    assert hm.runtime_info().collision_loaded
    for load in (_load_pinocchio, _load_dex):
        try:
            load()
        except ImportError as error:
            assert 'separate process' in str(error)
        else:
            raise AssertionError('incompatible optional runtime was allowed')
        assert 'pinocchio' not in sys.modules
        assert 'dex_retargeting' not in sys.modules
    assert hm.runtime_info().native_loaded
"""
    subprocess.run(
        [sys.executable, "-c", command, order],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )


def test_loaded_coal_obeys_the_native_collision_boundary():
    info = runtime.runtime_info()
    if not info.native_loaded:
        pytest.skip("requires a native build")
    if importlib.util.find_spec("coal") is None:
        pytest.skip("Python Coal is unavailable")
    command = """
import sys
import coal
assert 'pinocchio' not in sys.modules
import holistic_motion as hm
assert hm.runtime_info().native_loaded
assert not hm.runtime_info().collision_loaded
if sys.argv[1] == '1':
    try:
        hm.CollisionModel
    except ImportError as error:
        assert 'coal' in str(error) and 'unsupported' in str(error)
    else:
        raise AssertionError('independent Coal and native collision were combined')
    assert not hm.runtime_info().collision_loaded
else:
    assert hm.runtime_info().native_loaded
    assert hm.runtime_info().collision_compiled is False
"""
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "0"
    subprocess.run(
        [sys.executable, "-c", command, "1" if info.collision_compiled else "0"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("order", ["pin_first", "core_first"])
def test_core_and_python_pinocchio_solve_in_both_import_orders(tmp_path, order):
    info = runtime.runtime_info()
    if not info.native_loaded:
        pytest.skip("requires a native build")
    if importlib.util.find_spec("pinocchio") is None:
        pytest.skip("Python Pinocchio is unavailable")
    urdf = tmp_path / "runtime.urdf"
    urdf.write_text("""<robot name="runtime">
      <link name="base"/><link name="arm"/><link name="tool"/>
      <joint name="turn" type="continuous"><parent link="base"/><child link="arm"/>
        <axis xyz="0 0 1"/><limit effort="1" velocity="1"/></joint>
      <joint name="slide" type="prismatic"><parent link="arm"/><child link="tool"/>
        <origin xyz="0.1 0 0"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
    </robot>""")
    command = """
import sys
import numpy as np
if sys.argv[2] == 'pin_first':
    import pinocchio as pin
    import holistic_motion as hm
else:
    import holistic_motion as hm
    import pinocchio as pin
from holistic_motion.configuration import JointPositionMap
from holistic_motion.kit.retargeting import PinocchioRetargetingSolver
assert hm.runtime_info().native_loaded and not hm.runtime_info().collision_loaded
robot = hm.Robot(sys.argv[1])
chain = robot.create_kinematics('base', 'tool')
model = pin.buildModelFromUrdf(sys.argv[1])
data = model.createData()
mapping = JointPositionMap.from_model(model, robot.chain_joint_names('base', 'tool'))
context = pin.neutral(model)
for joints in np.random.default_rng(82).uniform(-0.8, 0.8, (16, 2)):
    full = mapping.configuration_with_joint_positions(context, joints)
    pin.forwardKinematics(model, data, full)
    pin.updateFramePlacements(model, data)
    np.testing.assert_allclose(chain.forward(joints), data.oMf[model.getFrameId('tool')].homogeneous, atol=1e-12)
solver = PinocchioRetargetingSolver(sys.argv[1], {'left_hand': 'tool'}, {'left_arm': ['turn', 'slide']})
solver.prepare('left_arm')
result = solver.solve({'left_hand': chain.forward([0.2, 0.3])})
assert result.success
np.testing.assert_allclose(solver.create_joint_position_map(['turn', 'slide']).joint_positions(result.configuration), [0.2, 0.3], atol=1e-3)
"""
    subprocess.run(
        [sys.executable, "-c", command, str(urdf), order],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
