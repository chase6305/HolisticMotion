"""Dependency-free composition of caller-supplied simulation adapters.

Payloads remain native objects. Adapters own device synchronization and buffer
lifetime; this module never converts them to NumPy or starts a viewer service.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from numbers import Integral
from threading import Lock
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class StateFrame:
    episode_id: int
    step_id: int
    sim_time: float
    payload: Any


@dataclass(frozen=True)
class ObservationFrame:
    state: StateFrame
    payload: Any


@dataclass(frozen=True)
class StepResult:
    state: StateFrame
    observation: Optional[ObservationFrame]
    viewer_error: Optional[Exception]


class PhysicsBackend(Protocol):
    def reset(self) -> Any:
        """Reset all environments and return native state."""
        ...

    def step(self, action: Any, dt: float) -> Any:
        """Advance dt and return native state; action semantics are adapter-owned."""
        ...

    def close(self) -> None: ...


class ObservationBackend(Protocol):
    def capture(self, state: StateFrame) -> Any:
        """Synchronize geometry and capture this state; retain native buffers."""
        ...

    def close(self) -> None: ...


class VisualizationBackend(Protocol):
    def publish(self, state: StateFrame) -> None:
        """Publish without blocking on UI consumption, or raise on failure."""
        ...

    def close(self) -> None: ...


class NullViewer:
    def publish(self, state: StateFrame) -> None:
        pass

    def close(self) -> None:
        pass


def _cadence(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _close_resources(resources):
    errors = []
    seen = set()
    for resource in reversed(resources):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        try:
            resource.close()
        except BaseException as error:  # noqa: BLE001 - finish cleanup before propagating interrupts
            errors.append(error)
    return tuple(errors)


def _validate_adapter(kind, adapter):
    methods = {
        "physics": ("reset", "step", "close"),
        "observation": ("capture", "close"),
        "viewer": ("publish", "close"),
    }
    if any(not callable(getattr(adapter, name, None)) for name in methods[kind]):
        raise TypeError(f"{kind} adapter must implement {', '.join(methods[kind])}")


def _exclusive_operation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        # Fail immediately: waiting here would deadlock a reentrant adapter
        # callback and could queue a stale reset/action behind an active step.
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("a simulation session operation is already running")
        try:
            return method(self, *args, **kwargs)
        finally:
            self._operation_lock.release()

    return guarded


class SimulationSession:
    """Synchronous orchestration with independent camera and viewer cadences.

    Call reset before stepping. Physics/capture failures require another reset;
    viewer failures are recorded and disable further publication. This class
    owns its adapters and closes all of them, including a failed viewer.
    reset/step/close reject concurrent or reentrant operations without waiting.
    """

    def __init__(
        self,
        physics: PhysicsBackend,
        *,
        observation: Optional[ObservationBackend] = None,
        viewer: Optional[VisualizationBackend] = None,
        capture_every_n_steps: int = 1,
        publish_every_n_steps: int = 1,
    ):
        self._capture_every = _cadence(capture_every_n_steps, "capture_every_n_steps")
        self._publish_every = _cadence(publish_every_n_steps, "publish_every_n_steps")
        _validate_adapter("physics", physics)
        if observation is not None:
            _validate_adapter("observation", observation)
        if viewer is not None:
            _validate_adapter("viewer", viewer)
        self._physics = physics
        self._observation = observation
        self._viewer = NullViewer() if viewer is None else viewer
        self._resources = (physics, observation, self._viewer)
        self._operation_lock = Lock()
        self._closed = False
        self._ready = False
        self._episode = -1
        self._state = None
        self.viewer_error = None
        self.close_errors = ()

    def _require_open(self):
        if self._closed:
            raise RuntimeError("simulation session is closed")

    @_exclusive_operation
    def reset(self) -> StateFrame:
        self._require_open()
        self._ready = False
        # Consume the episode ID even on a failed reset: an adapter may already
        # have changed state before raising, so never recycle its frame IDs.
        self._episode += 1
        payload = self._physics.reset()
        self._state = StateFrame(self._episode, 0, 0.0, payload)
        self._ready = True
        return self._state

    @_exclusive_operation
    def step(self, action: Any, dt: float) -> StepResult:
        self._require_open()
        if not self._ready:
            raise RuntimeError(
                "reset is required before stepping after creation or failure"
            )
        if isinstance(dt, bool):
            raise TypeError("dt must be a real duration, not boolean")
        dt = float(dt)
        next_time = self._state.sim_time + dt
        if not math.isfinite(dt) or dt <= 0.0 or not math.isfinite(next_time):
            raise ValueError(
                "dt and accumulated simulation time must be finite and positive"
            )
        if next_time <= self._state.sim_time:
            raise ValueError("dt is too small to advance the simulation timestamp")
        self._ready = False
        payload = self._physics.step(action, dt)
        state = StateFrame(self._episode, self._state.step_id + 1, next_time, payload)
        self._state = state
        observation = None
        if self._observation is not None and state.step_id % self._capture_every == 0:
            observation = ObservationFrame(state, self._observation.capture(state))
        self._ready = True
        if self.viewer_error is None and state.step_id % self._publish_every == 0:
            try:
                self._viewer.publish(state)
            except Exception as error:  # noqa: BLE001 - isolate and report optional viewer failures
                self.viewer_error = error
        return StepResult(state, observation, self.viewer_error)

    @_exclusive_operation
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready = False
        self.close_errors = _close_resources(self._resources)
        for error in self.close_errors:
            if not isinstance(error, Exception):
                # Honor cancellation after every owned adapter got its close
                # attempt; do not turn Ctrl-C/SystemExit into a RuntimeError.
                raise error
        if self.close_errors:
            raise RuntimeError(
                "one or more simulation adapters failed to close"
            ) from self.close_errors[0]

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise
        return False


@dataclass(frozen=True)
class _Factory:
    create: Callable[..., Any]
    scene_format: Optional[str]


class BackendRegistry:
    """Explicit factories with scene-format compatibility checked before loading.

    Register lazy factory functions to isolate optional imports. None means a
    scene-independent consumer; physics must declare a format. Bridging belongs
    inside a separately registered adapter, not an implicit fallback.
    """

    def __init__(self):
        self._factories = {"physics": {}, "observation": {}, "viewer": {}}
        self.register("viewer", "null", lambda physics: NullViewer(), scene_format=None)

    def register(
        self,
        kind: str,
        name: str,
        factory: Callable[..., Any],
        *,
        scene_format: Optional[str],
    ):
        if kind not in self._factories:
            raise ValueError(f"unknown backend kind: {kind}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("backend name must be non-empty")
        if name in self._factories[kind]:
            raise ValueError(f"backend already registered: {kind}/{name}")
        if not callable(factory):
            raise TypeError("backend factory must be callable")
        if (kind == "physics" and scene_format is None) or (
            scene_format is not None
            and (not isinstance(scene_format, str) or not scene_format.strip())
        ):
            raise ValueError(
                "physics requires a non-empty scene format; consumers may use None"
            )
        self._factories[kind][name] = _Factory(factory, scene_format)

    def create(
        self,
        *,
        physics: str,
        observation: Optional[str] = None,
        viewer: str = "null",
        options: Optional[Mapping[str, Mapping[str, Any]]] = None,
        capture_every_n_steps: int = 1,
        publish_every_n_steps: int = 1,
    ) -> SimulationSession:
        capture_every_n_steps = _cadence(capture_every_n_steps, "capture_every_n_steps")
        publish_every_n_steps = _cadence(publish_every_n_steps, "publish_every_n_steps")
        selected = {"physics": physics, "observation": observation, "viewer": viewer}
        specs = {}
        for kind, name in selected.items():
            if kind == "observation" and name is None:
                continue
            if not isinstance(name, str) or name not in self._factories[kind]:
                raise ValueError(f"unknown {kind} backend: {name}")
            specs[kind] = self._factories[kind][name]
        for kind, spec in specs.items():
            if kind != "physics" and spec.scene_format not in (
                None,
                specs["physics"].scene_format,
            ):
                raise ValueError(
                    f"incompatible {kind} scene format: {spec.scene_format}; physics provides {specs['physics'].scene_format}; register an explicit bridge adapter"
                )
        options = {} if options is None else dict(options)
        if options.keys() - specs.keys():
            raise ValueError("options supplied for an unknown or disabled backend kind")
        arguments = {kind: dict(options.get(kind, {})) for kind in specs}
        if any("physics" in values for values in arguments.values()):
            raise ValueError(
                "the physics factory argument is reserved for dependency injection"
            )
        resources = []
        try:
            engine = specs["physics"].create(**arguments["physics"])
            resources.append(engine)
            _validate_adapter("physics", engine)
            consumers = {}
            for kind in ("observation", "viewer"):
                if kind in specs:
                    consumers[kind] = specs[kind].create(
                        physics=engine, **arguments[kind]
                    )
                    resources.append(consumers[kind])
                    _validate_adapter(kind, consumers[kind])
            return SimulationSession(
                engine,
                observation=consumers.get("observation"),
                viewer=consumers["viewer"],
                capture_every_n_steps=capture_every_n_steps,
                publish_every_n_steps=publish_every_n_steps,
            )
        except BaseException:
            _close_resources(resources)
            raise
