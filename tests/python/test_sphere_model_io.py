"""Atomic sphere-model writes under actual temporary-file I/O failures."""

import pytest
from holistic_motion.geometry import sphere_fit


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
