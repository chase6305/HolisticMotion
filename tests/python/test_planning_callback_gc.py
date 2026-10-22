"""Native planning callbacks participate in Python cycle collection."""

import gc
import platform
import weakref

import holistic_motion as hm
import numpy as np
import pytest

requires_cpython_gc = pytest.mark.skipif(
    platform.python_implementation() != "CPython",
    reason="native callback cycle traversal is supported by CPython's collector",
)


@requires_cpython_gc
@pytest.mark.parametrize("base", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize("subclass", [False, True])
def test_dynamic_dictionary_has_one_gc_ownership_edge(base, subclass):
    class Derived(base):
        pass

    owner = (Derived if subclass else base)([-1.0], [1.0])
    owner.cycle = owner
    dictionary = owner.__dict__
    assert sum(value is dictionary for value in gc.get_referents(owner)) == 1


@requires_cpython_gc
@pytest.mark.parametrize("base", [hm.SamplingPlanner, hm.PathOptimizer])
def test_resurrected_local_subclass_is_collectible_again(base):
    survivors = []

    def create():
        class Derived(base):
            def __del__(self):
                survivors.append(self)

        owner = Derived([-1.0], [1.0])
        owner.cycle = owner

        def callback(state):
            return owner.cycle is owner

        owner.callback = callback
        owner.set_state_validator(callback)

    create()
    gc.collect()
    assert len(survivors) == 1
    owner = survivors.pop()
    references = weakref.ref(owner), weakref.ref(owner.callback)
    del owner
    gc.collect()
    assert all(reference() is None for reference in references)


@requires_cpython_gc
@pytest.mark.parametrize("kind", ["planner", "validator", "cost", "gradient"])
@pytest.mark.parametrize("subclass", [False, True])
@pytest.mark.parametrize("python_cycle", [False, True])
def test_callbacks_capturing_their_owner_are_collected(kind, subclass, python_cycle):
    base = hm.SamplingPlanner if kind == "planner" else hm.PathOptimizer

    class Derived(base):
        pass

    cls = Derived if subclass else base

    def create():
        owner = cls([-1.0], [1.0])
        # Include a Python-side cycle too, exercising the existing dictionary
        # traversal as well as the native callback edge.
        if python_cycle:
            owner.cycle = owner

        def callback(state):
            assert owner.joint_space.dof == 1
            if kind in ("planner", "validator"):
                return True
            if kind == "cost":
                return float(state @ state)
            return 2.0 * state

        if kind == "planner":
            owner.set_state_validator(callback)
        else:
            getattr(
                owner, "set_state_" + ("cost_gradient" if kind == "gradient" else kind)
            )(callback)
        return weakref.ref(owner), weakref.ref(callback)

    references = [create() for _ in range(16)]
    gc.collect()
    assert all(owner() is None and callback() is None for owner, callback in references)


@requires_cpython_gc
@pytest.mark.parametrize("cls", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize("named_space", [False, True])
def test_constructor_callback_cycles_are_collected(cls, named_space):
    def create():
        def callback(state):
            return owner.joint_space.dof == 1

        owner = (
            cls(hm.JointSpace([-1.0], [1.0]), callback)
            if named_space
            else cls([-1.0], [1.0], callback)
        )
        return weakref.ref(owner), weakref.ref(callback)

    owner, callback = create()
    gc.collect()
    assert owner() is None and callback() is None


@requires_cpython_gc
@pytest.mark.parametrize("collect_during_init", [False, True])
def test_multiple_native_bases_trace_all_callback_edges(collect_during_init):
    class Both(hm.SamplingPlanner, hm.PathOptimizer):
        pass

    class CollectingArray:
        def __init__(self, value):
            self.value = value

        def __array__(self, dtype=None, copy=None):
            gc.collect()
            return np.asarray([self.value], dtype=dtype)

    def create():
        owner = Both.__new__(Both)
        limits = (
            (CollectingArray(-1.0), CollectingArray(1.0))
            if collect_during_init
            else ([-1.0], [1.0])
        )
        hm.SamplingPlanner.__init__(owner, *limits)
        hm.PathOptimizer.__init__(owner, *limits)

        def callback(state):
            return owner.joint_space.dof == 1

        hm.SamplingPlanner.set_state_validator(owner, callback)
        hm.PathOptimizer.set_state_cost(owner, callback)
        return weakref.ref(owner), weakref.ref(callback)

    owner, callback = create()
    gc.collect()
    assert owner() is None and callback() is None


@requires_cpython_gc
@pytest.mark.parametrize("bound_method", [False, True])
@pytest.mark.parametrize("kind", ["planner", "validator", "cost", "gradient"])
def test_callable_instances_and_bound_methods_are_collected(kind, bound_method):
    class Callback:
        def __init__(self, owner):
            self.owner = owner

        def __call__(self, state):
            assert self.owner.joint_space.dof == 1
            if kind in ("planner", "validator"):
                return True
            if kind == "cost":
                return float(state @ state)
            return 2.0 * state

    def create():
        owner = (
            hm.SamplingPlanner([-1.0], [1.0])
            if kind == "planner"
            else hm.PathOptimizer([-1.0], [1.0])
        )
        callback = Callback(owner)
        function = callback.__call__ if bound_method else callback
        if kind in ("planner", "validator"):
            owner.set_state_validator(function)
        else:
            getattr(owner, "set_state_" + ("cost_gradient" if kind == "gradient" else kind))(
                function
            )
        return weakref.ref(owner), weakref.ref(callback)

    owner, callback = create()
    gc.collect()
    assert owner() is None and callback() is None


@pytest.mark.parametrize("cls", [hm.SamplingPlanner, hm.PathOptimizer])
def test_gc_handles_uninitialized_and_rejected_planning_objects(cls):
    uninitialized = cls.__new__(cls)
    reference = weakref.ref(uninitialized)
    del uninitialized
    with pytest.raises(ValueError):
        cls([], [])
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("release_gil", [False, True])
def test_active_request_retains_replaced_callback_during_gc(release_gil):
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    options = hm.PathOptimizationOptions()
    options.max_iterations = 4
    options.state_cost_weight = 1.0
    calls = []

    def callback(state):
        optimizer.set_state_cost(None)
        gc.collect()
        calls.append(float(state[0]))
        return float(state @ state)

    reference = weakref.ref(callback)
    optimizer.set_state_cost(callback)
    del callback
    result = optimizer.optimize(
        [[-0.8], [0.1], [0.8]], options, release_gil=release_gil
    )
    assert result.success and len(calls) > 1
    gc.collect()
    gc.collect()
    assert reference() is None
    np.testing.assert_array_equal(np.asarray(result.path)[[0, -1]], [[-0.8], [0.8]])


@pytest.mark.parametrize("kind", ["planner", "validator", "cost", "gradient"])
def test_explicit_callback_clear_releases_owner_cycle(kind):
    owner = (
        hm.SamplingPlanner([-1.0], [1.0])
        if kind == "planner"
        else hm.PathOptimizer([-1.0], [1.0])
    )

    def callback(state):
        return owner.joint_space.dof == 1

    setter = (
        owner.set_state_validator
        if kind in ("planner", "validator")
        else getattr(owner, "set_state_" + ("cost_gradient" if kind == "gradient" else kind))
    )
    setter(callback)
    reference = weakref.ref(callback)
    del callback
    setter(None)
    gc.collect()
    gc.collect()
    assert reference() is None
