"""Verify collision binaries before their Python initialization exchanges C++ types."""

import ctypes
import importlib.machinery
import importlib.util
from types import SimpleNamespace

import pytest
from holistic_motion import runtime


@pytest.mark.parametrize("layout", ["installed", "build_tree"])
@pytest.mark.parametrize("observed", [b"matching:Release", b"stale:Release", None])
def test_collision_binary_identity_is_checked_before_initialization(
    monkeypatch, layout, observed
):
    expected = "matching:Release"
    monkeypatch.setattr(
        runtime, "_native_module", lambda: SimpleNamespace(_native_build_id=expected)
    )
    name = (
        "holistic_motion._holistic_motion_collision"
        if layout == "installed"
        else "_holistic_motion_collision"
    )
    origin = "/synthetic/collision" + importlib.machinery.EXTENSION_SUFFIXES[0]
    spec = importlib.machinery.ModuleSpec(name, loader=None, origin=origin)
    looked_up, opened = [], []

    def find_spec(candidate):
        looked_up.append(candidate)
        return spec if candidate == name else None

    class Identity:
        def __call__(self):
            return observed

    identity = Identity()

    def open_library(path):
        opened.append(path)
        return SimpleNamespace(HolisticMotionCollisionBuildIdentity=identity)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(ctypes, "CDLL", open_library)
    if observed == expected.encode():
        runtime._verify_collision_extension()
    else:
        with pytest.raises(ImportError, match="native build identities"):
            runtime._verify_collision_extension()
    assert looked_up[-1] == name
    assert opened == [origin]
    assert identity.argtypes == []
    assert identity.restype is ctypes.c_char_p


@pytest.mark.parametrize(
    "failure", [OSError("wrong library"), AttributeError("legacy")]
)
def test_unverifiable_collision_binary_has_a_rebuild_diagnostic(monkeypatch, failure):
    monkeypatch.setattr(
        runtime,
        "_native_module",
        lambda: SimpleNamespace(_native_build_id="matching:Release"),
    )
    spec = importlib.machinery.ModuleSpec(
        "holistic_motion._holistic_motion_collision",
        loader=None,
        origin="/synthetic/collision" + importlib.machinery.EXTENSION_SUFFIXES[0],
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)

    def open_library(path):
        raise failure

    monkeypatch.setattr(ctypes, "CDLL", open_library)
    with pytest.raises(ImportError, match="Rebuild and install") as caught:
        runtime._verify_collision_extension()
    assert caught.value.__cause__ is failure


def test_missing_collision_binary_is_left_to_the_optional_importer(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_native_module",
        lambda: SimpleNamespace(_native_build_id="matching:Release"),
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    runtime._verify_collision_extension()


@pytest.mark.parametrize("expected", [None, "", 42])
def test_legacy_and_invalid_core_identities(monkeypatch, expected):
    monkeypatch.setattr(
        runtime, "_native_module", lambda: SimpleNamespace(_native_build_id=expected)
    )
    if expected is None:
        runtime._verify_collision_extension()
    else:
        with pytest.raises(ImportError, match="native build identities"):
            runtime._verify_collision_extension()
