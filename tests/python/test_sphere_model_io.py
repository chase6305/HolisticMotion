"""Sphere-model input ownership and atomic writes under I/O failures."""

from collections.abc import Mapping

import pytest
from holistic_motion.geometry import sphere_fit


@pytest.mark.parametrize("mutation", ["clear", "replace"])
def test_save_captures_link_entries_before_sphere_iteration(mutation, tmp_path):
    first = sphere_fit.SphereSpec((0.0, 0.0, 0.0), 0.2)
    second = sphere_fit.SphereSpec((1.0, 0.0, 0.0), 0.3)
    links = {}
    calls = []

    def spheres():
        calls.append(True)
        if mutation == "clear":
            links.clear()
        else:
            links["second"] = []
        yield first

    links.update(first=spheres(), second=[second])
    path = tmp_path / "spheres.json"
    sphere_fit.save_sphere_model(path, links)
    loaded, _ = sphere_fit.load_sphere_model(path)
    assert calls == [True]
    if mutation == "clear":
        assert links == {}
    else:
        assert links["second"] == []
    assert loaded == {"first": (first,), "second": (second,)}


def test_save_consumes_custom_mapping_values_before_next_lookup(tmp_path):
    first = sphere_fit.SphereSpec((0.0, 0.0, 0.0), 0.2)
    second = sphere_fit.SphereSpec((1.0, 0.0, 0.0), 0.3)

    class ReusingMapping(Mapping):
        def __init__(self):
            self.spheres = []

        def __len__(self):
            return 2

        def __iter__(self):
            return iter(("first", "second"))

        def __getitem__(self, key):
            if key not in ("first", "second"):
                raise KeyError(key)
            self.spheres[:] = [first if key == "first" else second]
            return self.spheres

    path = tmp_path / "spheres.json"
    sphere_fit.save_sphere_model(path, ReusingMapping())
    loaded, _ = sphere_fit.load_sphere_model(path)
    assert loaded == {"first": (first,), "second": (second,)}


@pytest.mark.parametrize("stage", ["write", "flush", "fsync", "replace"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_interrupted_save_preserves_model_and_removes_temporary(
    stage, error_type, tmp_path, monkeypatch
):
    target = tmp_path / "spheres.json"
    original = {"body": [sphere_fit.SphereSpec((0.0, 0.0, 0.0), 0.2)]}
    replacement = {"body": [sphere_fit.SphereSpec((1.0, 0.0, 0.0), 0.3)]}
    sphere_fit.save_sphere_model(target, original)
    before = target.read_bytes()
    error = error_type(f"{stage} failed")
    create = sphere_fit.tempfile.NamedTemporaryFile

    class FailingFile:
        def __init__(self, file):
            self.file = file

        def __getattr__(self, name):
            return getattr(self.file, name)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def write(self, text):
            if stage == "write":
                self.file.write(text[:20])
                raise error
            return self.file.write(text)

        def flush(self):
            if stage == "flush":
                raise error
            return self.file.flush()

    def fail(*args):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(
            sphere_fit.tempfile,
            "NamedTemporaryFile",
            lambda **kwargs: FailingFile(create(**kwargs)),
        )
        if stage in ("fsync", "replace"):
            patch.setattr(sphere_fit.os, stage, fail)
        with pytest.raises(BaseException) as caught:
            sphere_fit.save_sphere_model(target, replacement)
    assert caught.value is error
    assert target.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["spheres.json"]
    sphere_fit.save_sphere_model(target, replacement)
    loaded, _ = sphere_fit.load_sphere_model(target)
    assert loaded["body"] == tuple(replacement["body"])
