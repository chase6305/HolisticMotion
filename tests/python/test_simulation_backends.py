"""Composition tests use synthetic adapters, not a claimed physics engine."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from holistic_motion.kit.simulation import BackendRegistry, SimulationSession


class Physics:
    def __init__(self, events):
        self.events = events
        self.value = 0
        self.fail = False

    def reset(self):
        self.events.append("reset")
        self.value = 0
        return self.value

    def step(self, action, dt):
        self.events.append("step")
        self.value += action
        if self.fail:
            raise RuntimeError("physics failed after mutation")
        return self.value

    def close(self):
        self.events.append("close physics")


class Camera:
    def __init__(self, physics, events):
        self.physics = physics
        self.events = events
        self.payload = object()
        self.frames = []
        self.fail = False

    def capture(self, state):
        self.events.append("capture")
        self.frames.append(state)
        assert state.payload == self.physics.value
        if self.fail:
            raise RuntimeError("capture failed")
        return self.payload

    def close(self):
        self.events.append("close camera")


class Viewer:
    def __init__(self, events):
        self.events = events
        self.frames = []
        self.fail = False

    def publish(self, state):
        self.events.append("publish")
        self.frames.append(state)
        if self.fail:
            raise ConnectionError("browser disconnected")

    def close(self):
        self.events.append("close viewer")


def registry(events):
    result = BackendRegistry()
    result.register("physics", "test", lambda: Physics(events), scene_format="test-v1")
    result.register(
        "observation",
        "camera",
        lambda physics: Camera(physics, events),
        scene_format="test-v1",
    )
    result.register(
        "viewer", "web", lambda physics: Viewer(events), scene_format="test-v1"
    )
    return result


@pytest.mark.parametrize("viewer", ["null", "web"])
def test_capture_is_independent_of_display_with_native_payload_identity(viewer):
    events = []
    with registry(events).create(
        physics="test",
        observation="camera",
        viewer=viewer,
        capture_every_n_steps=2,
        publish_every_n_steps=3,
    ) as session:
        first = session.reset()
        assert (first.episode_id, first.step_id, first.sim_time) == (0, 0, 0)
        results = [session.step(1, 0.1) for _ in range(6)]
        captured = [r for r in results if r.observation is not None]
        assert [r.state.step_id for r in captured] == [2, 4, 6]
        assert [r.state.payload for r in results] == list(range(1, 7))
        assert results[-1].state.sim_time == pytest.approx(0.6)
        assert all(r.observation.state is r.state for r in captured)
        assert all(
            r.observation.payload is captured[0].observation.payload for r in captured
        )
        assert events.count("publish") == (2 if viewer == "web" else 0)
        # At a coincident cadence, capture precedes visualization.
        assert events[-3:] == (
            ["step", "capture", "publish"]
            if viewer == "web"
            else ["step", "step", "capture"]
        )
        second = session.reset()
        assert (second.episode_id, second.step_id, second.sim_time) == (1, 0, 0)
        assert session.step(1, 0.1).observation is None
    assert events[-2:] == ["close camera", "close physics"]


def test_disabled_camera_is_never_created():
    events = []
    backends = registry(events)

    def fail(physics):
        pytest.fail("disabled camera factory was called")

    backends.register("observation", "unused", fail, scene_format="test-v1")
    with backends.create(physics="test") as session:
        session.reset()
        assert session.step(1, 0.1).observation is None


@pytest.mark.parametrize("kind", ["observation", "viewer"])
def test_incompatibility_rejected_before_any_factory_runs(kind):
    events = []
    backends = registry(events)
    backends.register(
        kind,
        "wrong",
        lambda **kwargs: pytest.fail("factory called"),
        scene_format="another-engine-v1",
    )
    with pytest.raises(ValueError, match="incompatible"):
        backends.create(physics="test", **{kind: "wrong"})
    assert events == []


@pytest.mark.parametrize(
    "selection", [{"physics": "missing"}, {"observation": "missing"}, {"viewer": None}]
)
def test_unknown_backend_rejected_before_creation(selection):
    events = []
    args = {"physics": "test", **selection}
    with pytest.raises(ValueError, match="unknown"):
        registry(events).create(**args)
    assert events == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_cadence_rejected_before_creation(value):
    events = []
    with pytest.raises(ValueError, match="positive integer"):
        registry(events).create(physics="test", capture_every_n_steps=value)
    assert events == []


def test_viewer_disconnect_does_not_stop_capture_or_physics():
    events = []
    physics = Physics(events)
    camera, viewer = Camera(physics, events), Viewer(events)
    viewer.fail = True
    with SimulationSession(physics, observation=camera, viewer=viewer) as session:
        session.reset()
        results = [session.step(1, 0.1) for _ in range(4)]
        assert all(isinstance(r.viewer_error, ConnectionError) for r in results)
        assert events.count("publish") == 1
        assert events.count("capture") == events.count("step") == 4
    assert events[-3:] == ["close viewer", "close camera", "close physics"]


@pytest.mark.parametrize("failure", ["physics", "camera"])
def test_failed_state_update_requires_reset_and_does_not_repeat_action(failure):
    events = []
    physics = Physics(events)
    camera = Camera(physics, events)
    with SimulationSession(physics, observation=camera) as session:
        session.reset()
        adapter = physics if failure == "physics" else camera
        adapter.fail = True
        with pytest.raises(RuntimeError, match="failed"):
            session.step(1, 0.1)
        with pytest.raises(RuntimeError, match="reset is required"):
            session.step(1, 0.1)
        assert physics.value == 1
        adapter.fail = False
        session.reset()
        result = session.step(2, 0.1)
        assert result.state.episode_id == 1
        assert result.state.step_id == 1
        assert result.observation.state.payload == 2


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf"), True])
def test_invalid_dt_does_not_mutate_physics(dt):
    events = []
    with SimulationSession(Physics(events)) as session:
        session.reset()
        with pytest.raises((ValueError, TypeError)):
            session.step(1, dt)
        assert events == ["reset"]
        assert session.step(1, 0.1).state.step_id == 1


def test_failed_factory_closes_previously_created_adapters():
    events = []
    backends = registry(events)

    def fail(physics):
        raise ImportError("optional viewer is unavailable")

    backends.register("viewer", "missing_dep", fail, scene_format=None)
    with pytest.raises(ImportError, match="optional viewer"):
        backends.create(physics="test", observation="camera", viewer="missing_dep")
    assert events == ["close camera", "close physics"]


def test_close_attempts_every_adapter_and_is_idempotent():
    events = []
    physics = Physics(events)

    class BadViewer(Viewer):
        def close(self):
            super().close()
            raise RuntimeError("close failed")

    session = SimulationSession(
        physics, observation=Camera(physics, events), viewer=BadViewer(events)
    )
    with pytest.raises(RuntimeError, match="failed to close"):
        session.close()
    session.close()
    assert events == ["close viewer", "close camera", "close physics"]
    assert len(session.close_errors) == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.reset()


def test_session_requires_explicit_reset():
    session = SimulationSession(Physics([]))
    with session, pytest.raises(RuntimeError, match="reset is required"):
        session.step(1, 0.1)


def test_plain_import_does_not_load_optional_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from holistic_motion.kit.simulation import BackendRegistry; import sys; assert all(n not in sys.modules for n in ('newton', 'warp', 'viser', 'rerun', 'pyglet', 'torch')); BackendRegistry()",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "kind,name,scene_format",
    [
        ("bad", "x", "v1"),
        ("physics", "", "v1"),
        ("physics", "x", None),
        ("viewer", "x", ""),
    ],
)
def test_invalid_registration(kind, name, scene_format):
    with pytest.raises(ValueError):
        BackendRegistry().register(kind, name, lambda: None, scene_format=scene_format)


def test_duplicate_registration_does_not_replace_factory():
    events = []
    backends = registry(events)
    with pytest.raises(ValueError, match="already registered"):
        backends.register("physics", "test", lambda: None, scene_format="wrong")
    with backends.create(physics="test") as session:
        session.reset()
        assert session.step(3, 0.1).state.payload == 3


def test_invalid_factory_output_releases_engine():
    events = []
    backends = registry(events)
    backends.register(
        "observation", "invalid", lambda physics: None, scene_format="test-v1"
    )
    with pytest.raises(TypeError, match="observation adapter"):
        backends.create(physics="test", observation="invalid")
    assert events == ["close physics"]


@pytest.mark.parametrize(
    "options",
    [{"observation": {}}, {"unknown": {}}, {"physics": {"physics": object()}}],
)
def test_invalid_options_fail_before_factory_creation(options):
    events = []
    with pytest.raises(ValueError):
        registry(events).create(physics="test", options=options)
    assert events == []


def test_context_cleanup_preserves_original_error():
    class FailingClose(Physics):
        def close(self):
            super().close()
            raise RuntimeError("cleanup failure")

    events = []
    session = SimulationSession(FailingClose(events))
    with pytest.raises(ValueError, match="original failure"), session:
        raise ValueError("original failure")
    assert events == ["close physics"]
    assert str(session.close_errors[0]) == "cleanup failure"


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_close_interrupt_still_releases_all_resources(interrupt_type):
    events = []
    failure = interrupt_type("interrupted close")

    class InterruptedViewer(Viewer):
        def close(self):
            super().close()
            raise failure

    physics = Physics(events)
    session = SimulationSession(
        physics, observation=Camera(physics, events), viewer=InterruptedViewer(events)
    )
    with pytest.raises(interrupt_type) as caught:
        session.close()
    assert caught.value is failure
    assert events == ["close viewer", "close camera", "close physics"]
    assert session.close_errors == (failure,)
    session.close()
    assert len(events) == 3


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_context_error_survives_interrupted_cleanup(interrupt_type):
    events = []
    cleanup_error = interrupt_type("cleanup interrupted")
    original = ValueError("original failure")

    class InterruptedViewer(Viewer):
        def close(self):
            super().close()
            raise cleanup_error

    session = SimulationSession(Physics(events), viewer=InterruptedViewer(events))
    with pytest.raises(BaseException) as caught, session:
        raise original
    assert caught.value is original
    assert events == ["close viewer", "close physics"]
    assert session.close_errors == (cleanup_error,)


@pytest.mark.parametrize("creation_type", [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [RuntimeError, KeyboardInterrupt])
def test_factory_failure_preserved_despite_cleanup_interrupt(
    creation_type, cleanup_type
):
    events = []
    backends = registry(events)
    original = creation_type("factory failed")

    class FailingCamera(Camera):
        def close(self):
            super().close()
            raise cleanup_type("camera cleanup failed")

    def fail(physics):
        raise original

    backends.register(
        "observation",
        "failing_camera",
        lambda physics: FailingCamera(physics, events),
        scene_format="test-v1",
    )
    backends.register("viewer", "failing_viewer", fail, scene_format=None)
    with pytest.raises(BaseException) as caught:
        backends.create(
            physics="test", observation="failing_camera", viewer="failing_viewer"
        )
    assert caught.value is original
    assert events == ["close camera", "close physics"]


def test_tiny_or_overflowing_timestep_cannot_advance_physics():
    events = []
    with SimulationSession(Physics(events)) as session:
        session.reset()
        session.step(1, 1e308)
        with pytest.raises(ValueError, match="too small"):
            session.step(1, 1.0)
        with pytest.raises(ValueError, match="finite"):
            session.step(1, 1e308)
        assert events.count("step") == 1


@pytest.mark.parametrize("phase", ["reset", "step", "capture", "publish"])
@pytest.mark.parametrize("nested", ["reset", "step", "close"])
def test_adapter_callback_cannot_reenter_session(monkeypatch, phase, nested):
    events = []
    physics = Physics(events)
    camera, viewer = Camera(physics, events), Viewer(events)
    with SimulationSession(physics, observation=camera, viewer=viewer) as session:
        session.reset()
        adapter = {
            "reset": physics,
            "step": physics,
            "capture": camera,
            "publish": viewer,
        }[phase]
        original = getattr(adapter, phase)
        attempted = []

        def callback(*args):
            # Avoid infinite recursion on the unfixed implementation.
            if not attempted:
                attempted.append(True)
                with pytest.raises(RuntimeError, match="already running"):
                    if nested == "step":
                        session.step(99, 0.1)
                    else:
                        getattr(session, nested)()
            return original(*args)

        monkeypatch.setattr(adapter, phase, callback)
        if phase == "reset":
            frame = session.reset()
            assert (frame.episode_id, frame.step_id, frame.payload) == (1, 0, 0)
        else:
            result = session.step(1, 0.1)
            assert result.viewer_error is None
            assert (
                result.state.episode_id,
                result.state.step_id,
                result.state.payload,
            ) == (0, 1, 1)
            assert result.observation.state is result.state
        assert attempted == [True]
        assert not any(event.startswith("close") for event in events)
        assert events.count("step") == (0 if phase == "reset" else 1)


@pytest.mark.parametrize("concurrent", ["reset", "step", "close"])
def test_concurrent_operation_rejected_without_waiting_or_closing_resources(
    monkeypatch, concurrent
):
    events = []
    physics = Physics(events)
    entered, release = Event(), Event()
    original = physics.step

    def blocked(*args):
        entered.set()
        assert release.wait(timeout=5), "test did not release adapter"
        return original(*args)

    monkeypatch.setattr(physics, "step", blocked)
    with (
        SimulationSession(physics) as session,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        session.reset()
        future = pool.submit(session.step, 1, 0.1)
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(RuntimeError, match="already running"):
                if concurrent == "step":
                    session.step(99, 0.1)
                else:
                    getattr(session, concurrent)()
            assert events == ["reset"]
        finally:
            release.set()
        result = future.result(timeout=5)
        assert (
            result.state.episode_id,
            result.state.step_id,
            result.state.payload,
        ) == (0, 1, 1)
        # The guard is released after success and permits normal subsequent work.
        assert session.step(2, 0.1).state.payload == 3


@pytest.mark.parametrize("phase", ["reset", "step", "capture"])
def test_interrupt_releases_guard_but_requires_reset(monkeypatch, phase):
    events = []
    physics = Physics(events)
    camera = Camera(physics, events)
    with SimulationSession(physics, observation=camera) as session:
        session.reset()
        adapter = physics if phase != "capture" else camera
        original = getattr(adapter, phase)

        def interrupt(*args):
            original(*args)
            raise KeyboardInterrupt("interrupted adapter")

        with monkeypatch.context() as patch:
            patch.setattr(adapter, phase, interrupt)
            with pytest.raises(KeyboardInterrupt, match="interrupted adapter"):
                if phase == "reset":
                    session.reset()
                else:
                    session.step(1, 0.1)
        with pytest.raises(RuntimeError, match="reset is required"):
            session.step(1, 0.1)
        recovered = session.reset()
        result = session.step(2, 0.1)
        assert recovered.episode_id == (2 if phase == "reset" else 1)
        assert result.state.step_id == 1
        assert result.observation.state.payload == 2


def test_uncaught_viewer_reentry_isolated_without_changing_state(monkeypatch):
    events = []
    physics = Physics(events)
    viewer = Viewer(events)
    with SimulationSession(physics, viewer=viewer) as session:
        session.reset()
        monkeypatch.setattr(viewer, "publish", lambda state: session.reset())
        result = session.step(1, 0.1)
        assert isinstance(result.viewer_error, RuntimeError)
        assert "already running" in str(result.viewer_error)
        assert (
            result.state.episode_id,
            result.state.step_id,
            result.state.payload,
        ) == (0, 1, 1)
        assert events.count("reset") == 1
        assert session.step(2, 0.1).state.payload == 3
