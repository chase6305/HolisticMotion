# Simulation, camera, and visualization backends

<div class="language-switcher">English · <a href="../zh_CN/simulation_backends.html">简体中文</a></div>

## Scope and implementation status

`holistic_motion.kit.simulation` provides three backend protocols,
`BackendRegistry`, and `SimulationSession` for caller-supplied adapters.
The only built-in viewer is the dependency-free `NullViewer`. No physics engine,
camera renderer, or platform viewer is selected or installed by this module.

HolisticMotion supplies motion calculations. The application owns engine choice,
actuator semantics, assets, and the training loop. External integrations implement
these interfaces with their own optional dependencies.

## Independent interfaces

| Interface | Current contract | Implementation |
| --- | --- | --- |
| `PhysicsBackend` | Reset all environments, step typed actions, return native state, close | Caller-provided |
| `ObservationBackend` | Capture a state and return a native payload, close | Caller-provided |
| `VisualizationBackend` | Publish state at an independent cadence, close | Built-in NullViewer or caller-provided |

## Available composition API

Register factories explicitly. Physics factories receive their configured keyword
arguments; observation/viewer factories additionally receive `physics=engine`.
Put optional imports inside these factories. The registry checks names, declared
scene formats, and cadences before calling any factory. Consumer scene format
`None` means scene-independent; incompatible formats need an explicitly registered
bridge adapter. Options belong to `physics`, `observation`, or `viewer` mappings.

```python
from holistic_motion.kit.simulation import BackendRegistry

def compose(make_physics, make_camera):
    registry = BackendRegistry()
    registry.register("physics", "engine", make_physics, scene_format="engine-v1")
    registry.register("observation", "camera", make_camera, scene_format="engine-v1")
    return registry.create(
        physics="engine", observation="camera", viewer="null",
        capture_every_n_steps=4, publish_every_n_steps=20,
    )
```

Use the returned session as a context manager, call `reset()`, then
`step(action, dt)`. `PhysicsBackend` implements `reset`, `step`, and `close`;
`ObservationBackend` implements `capture(StateFrame)` and `close`;
`VisualizationBackend` implements `publish(StateFrame)` and `close`.
Already constructed adapters can instead be passed to `SimulationSession` directly.

Every step returns `StepResult.state`, an optional `ObservationFrame`, and a
`viewer_error`. State includes a session-wide episode ID, step ID, simulation time,
and native payload. Camera payloads retain identity and reference the same state
envelope; no array copy or conversion occurs. When capture is not due, observation
is `None`, never the previous frame. Payload contents can still be mutable and
borrowed; adapters must specify lifetime and device synchronization. The wrapper
does not validate camera metadata or guarantee GPU completion.

Reset currently affects all environments and restarts the step/capture cadence.
Partial resets, per-environment episode IDs, and asynchronous execution are future
work. A physics or camera exception propagates and requires reset before another
step; the session does not retry an action that may already have advanced physics.
A viewer exception is reported and permanently disables publication for that
session, including after reset; physics/capture continue. `publish` must itself
be nonblocking; bounded Web queues are the adapter's responsibility.

Sessions own adapter cleanup in reverse order. Factory failure closes previously
created adapters while preserving the construction exception. Explicit `close()`
attempts all adapters, reports failures through `close_errors`, and is idempotent.
Context cleanup preserves an already-active application exception. Factories
must release any partially created resources before raising themselves.

Cleanup also handles `KeyboardInterrupt` and `SystemExit`: all remaining owned
adapters receive a close attempt before the first such interruption is re-raised
by explicit `close()`. `close_errors` retains ordinary errors and interruptions
in cleanup order. An active construction or context-body exception takes
precedence over cleanup failures, including cleanup interruptions. This cannot
force an adapter's blocking `close()` to return or handle process termination.

`reset`, `step`, and `close` share a per-session nonblocking operation guard.
Concurrent calls and calls reentered from adapter callbacks raise `RuntimeError`
with `already running` before touching adapters or frame counters. No operation
is queued automatically. Wait for the active call to finish before closing;
`close` does not cancel an active step. The guard is released on success,
exceptions, and interrupts. Interrupted physics/capture still require reset.
An uncaught reentry error from a viewer follows the normal viewer-failure policy.

This protects session transitions, not arbitrary adapter access or borrowed
payloads. Adapters may require one owner thread or explicit GPU synchronization;
sequential calls from different threads are not automatically safe. Send UI
requests to the application loop instead of calling session methods recursively.

Run `python examples/python/simulation/backend_composition.py` for a synthetic
composition example with observations at steps 2, 4, and 6 and no viewer. It does
not simulate robot dynamics or render camera images.

## Compatibility and data boundaries

The registry compares declared scene formats before creating adapters. Independent
interfaces do not imply that every engine and camera combination is compatible.
A consumer must accept the engine's native representation or use an explicit
bridge adapter. No implicit fallback engine or renderer is provided.

Applications map joints by name and DOF layout, with explicit units and coordinate
frames. Configuration and velocity dimensions may differ (`nq != nv`). Position,
velocity, and torque actions must retain their declared meaning across adapters.

Image adapters should specify environment/camera IDs, timestamps, intrinsics and
extrinsics, shape, dtype, color space, depth semantics, validity masks, device,
ownership, and completion events. Preserve environment and camera batch axes;
display tiling belongs in the viewer adapter. The generic layer carries opaque
payloads and does not validate these image-specific fields.

## Independent capture and display

Use `observation=None` to disable capture and `viewer="null"` to disable display.
A null viewer does not disable an active observation adapter. Physics, capture,
and display cadences remain separate. Training termination follows the application
budget or cancellation request, not window state.

Web adapters should use bounded display queues and keep display-frame dropping
separate from training-observation delivery. Device buffers must not be reused
until asynchronous consumers finish; an adapter's synchronization policy remains
its responsibility.

## Validation scope

Synthetic adapters verify factory selection, compatibility rejection, independent
cadences, reset and failure recovery, cleanup, and operation guards. Run
`pytest tests/python/test_simulation_backends.py` to check this contract.
The synthetic example does not establish real-engine integration, camera accuracy,
GPU synchronization, or training throughput; those require the application's
chosen adapters and workloads.
