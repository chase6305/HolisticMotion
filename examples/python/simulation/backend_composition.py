"""Exercise backend composition with synthetic data; no physics engine or rendering."""

from holistic_motion.kit.simulation import BackendRegistry


class CounterPhysics:
    def reset(self):
        self.value = 0
        return self.value

    def step(self, action, dt):
        self.value += action
        return self.value

    def close(self):
        pass


class SyntheticObservation:
    def __init__(self, physics):
        self.physics = physics

    def capture(self, state):
        # Real camera adapters return native image buffers plus camera metadata.
        return {"synthetic_value": self.physics.value}

    def close(self):
        pass


def main():
    registry = BackendRegistry()
    registry.register("physics", "counter", CounterPhysics, scene_format="counter-v1")
    registry.register(
        "observation", "synthetic", SyntheticObservation, scene_format="counter-v1"
    )
    with registry.create(
        physics="counter",
        observation="synthetic",
        viewer="null",
        capture_every_n_steps=2,
    ) as session:
        session.reset()
        for _ in range(6):
            result = session.step(action=1, dt=0.01)
            if result.observation is not None:
                print(
                    f"step={result.state.step_id} observation={result.observation.payload}"
                )


if __name__ == "__main__":
    main()
