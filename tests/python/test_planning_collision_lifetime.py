"""Native collision callbacks own resources throughout requests and collection."""

import gc
import platform
import weakref

import holistic_motion as hm
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "CollisionModel"), reason="native collision support is disabled"
)
requires_cpython_gc = pytest.mark.skipif(
    platform.python_implementation() != "CPython",
    reason="native reference cycle traversal requires CPython's collector",
)
FACTORIES = [
    ("mesh", "sampling_full"),
    ("mesh", "sampling_indices"),
    ("mesh", "sampling_named"),
    ("sphere", "sampling_full"),
    ("sphere", "sampling_named"),
    ("mesh", "optimizer_named"),
    ("sphere", "optimizer_full"),
    ("sphere", "optimizer_named"),
]


@pytest.fixture(params=FACTORIES, ids=lambda item: "_".join(item))
def resource_factory(tmp_path, request):
    family, kind = request.param
    base = hm.CollisionModel if family == "mesh" else hm.SphereCollisionModel

    class Model(base):
        pass

    path = tmp_path / "model.urdf"
    geometry = '<collision><geometry><sphere radius="0.2"/></geometry></collision>'
    path.write_text(
        f'<robot name="resources"><link name="base">{geometry}</link>'
        f'<link name="slider">{geometry}</link>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="slider"/><axis xyz="1 0 0"/>'
        '<limit lower="0" upper="2" effort="1" velocity="1"/></joint></robot>'
    )

    def create():
        model = (
            Model(str(path), exclude_adjacent=False)
            if family == "mesh"
            else Model(
                str(path),
                [
                    hm.CollisionSphere("base", "base", [0, 0, 0], 0.2),
                    hm.CollisionSphere("slider", "slider", [0, 0, 0], 0.2),
                ],
            )
        )
        assert model.pair_count == 1
        if kind == "sampling_indices":
            owner = hm.SamplingPlanner.from_collision_model(
                [0.0], [2.0], model, [0], [1.0]
            )
        elif kind == "sampling_full":
            factory = (
                hm.SamplingPlanner.from_collision_model
                if family == "mesh"
                else hm.SamplingPlanner.from_sphere_collision_model
            )
            owner = factory([0.0], [2.0], model)
        elif kind == "sampling_named":
            factory = (
                hm.SamplingPlanner.from_collision_joints
                if family == "mesh"
                else hm.SamplingPlanner.from_sphere_collision_joints
            )
            owner = factory(model, ["slide"], [1.0])
        elif kind == "optimizer_full":
            owner = hm.PathOptimizer.from_sphere_collision_model(
                [0.0], [2.0], model, clearance=0.6
            )
        else:
            factory = (
                hm.PathOptimizer.from_collision_joints
                if family == "mesh"
                else hm.PathOptimizer.from_sphere_collision_joints
            )
            owner = factory(model, ["slide"], [1.0], clearance=0.6)
        return model, owner

    return create


def _query(owner, release):
    if isinstance(owner, hm.SamplingPlanner):
        return owner.plan([0.8], [1.2], release_gil=release)
    return owner.optimize([[0.8], [1.0], [1.2]], release_gil=release)


@pytest.mark.parametrize("release", [False, True])
def test_native_factory_retains_model_after_caller_releases_it(
    resource_factory, release
):
    model, owner = resource_factory()
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is not None
    assert _query(owner, release).success


@requires_cpython_gc
def test_collision_factory_back_reference_cycles_are_collected(resource_factory):
    references = []
    for _ in range(8):
        model, owner = resource_factory()
        model.owner = owner
        references.append((weakref.ref(model), weakref.ref(owner)))
        del model, owner
    gc.collect()
    assert all(model() is None and owner() is None for model, owner in references)


@requires_cpython_gc
def test_replacing_native_callbacks_releases_their_resource(resource_factory):
    model, owner = resource_factory()
    reference = weakref.ref(model)
    del model
    owner.set_state_validator(None)
    if isinstance(owner, hm.PathOptimizer):
        owner.set_state_cost(None)
        owner.set_state_cost_gradient(None)
    gc.collect()
    assert reference() is None
    assert _query(owner, True).success


@requires_cpython_gc
@pytest.mark.parametrize("release", [False, True])
def test_active_request_keeps_model_during_reentrant_callback_replacement(
    resource_factory, release
):
    model, owner = resource_factory()
    if not isinstance(owner, hm.PathOptimizer):
        pytest.skip("requires optimizer objective callbacks")
    reference = weakref.ref(model)
    del model
    calls = []

    def cost(state):
        owner.set_state_validator(None)
        owner.set_state_cost(None)
        owner.set_state_cost_gradient(None)
        gc.collect()
        assert reference() is not None
        calls.append(True)
        return 0.0

    owner.set_state_cost(cost)
    options = hm.PathOptimizationOptions()
    options.state_cost_weight = 1.0
    options.max_iterations = 2
    assert owner.optimize([[0.8], [1.0], [1.2]], options, release_gil=release).success
    assert calls
    gc.collect()
    assert reference() is None


@requires_cpython_gc
@pytest.mark.parametrize("release", [False, True])
def test_model_finalizer_can_query_its_owner_before_cycle_teardown(
    resource_factory, release
):
    model, owner = resource_factory()
    calls = []

    def finalize(self):
        calls.append(_query(self.owner, release).success)

    type(model).__del__ = finalize
    model.owner = owner
    model_ref, owner_ref = weakref.ref(model), weakref.ref(owner)
    del model, owner
    gc.collect()
    assert calls == [True]
    assert model_ref() is None and owner_ref() is None


@pytest.mark.parametrize(
    "family,kind",
    [
        ("mesh", "sampling_named"),
        ("mesh", "sampling_indices"),
        ("mesh", "optimizer"),
        ("sphere", "sampling_named"),
        ("sphere", "optimizer"),
    ],
)
@pytest.mark.parametrize("release", [False, True])
def test_factory_prepares_inactive_manifold_context_before_caller_edits(
    tmp_path, family, kind, release
):
    import numpy as np

    path = tmp_path / "floating.urdf"
    geometry = '<collision><geometry><sphere radius="0.1"/></geometry></collision>'
    path.write_text(
        f'<robot name="context"><link name="base">{geometry}</link>'
        '<link name="body"/>'
        f'<link name="tool">{geometry}</link>'
        '<joint name="floating" type="floating"><parent link="base"/>'
        '<child link="body"/></joint>'
        '<joint name="slide" type="prismatic"><parent link="body"/>'
        '<child link="tool"/><origin xyz="1 0 0"/><axis xyz="1 0 0"/>'
        '<limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>'
    )
    model = (
        hm.CollisionModel(str(path), exclude_adjacent=False)
        if family == "mesh"
        else hm.SphereCollisionModel(
            str(path),
            [
                hm.CollisionSphere("base", "base", [0, 0, 0], 0.1),
                hm.CollisionSphere("tool", "tool", [0, 0, 0], 0.1),
            ],
        )
    )
    context = model.neutral_configuration()
    assert context.size == 8 and model.nv == 7
    context[0] = 2.0
    names, indices = ["slide"], [7]
    if kind == "sampling_indices":
        owner = hm.SamplingPlanner.from_collision_model(
            [0.0], [1.0], model, indices, context
        )
    else:
        cls = hm.SamplingPlanner if kind.startswith("sampling") else hm.PathOptimizer
        factory = getattr(
            cls,
            "from_collision_joints"
            if family == "mesh"
            else "from_sphere_collision_joints",
        )
        options = {"clearance": 0.6} if kind == "optimizer" else {}
        owner = factory(model, names, context, **options)
    colliding = context.copy()
    colliding[0], colliding[7] = -1.3, 0.3
    assert model.in_collision(colliding)
    context.fill(np.nan)
    names.clear()
    indices[:] = [-1]
    if kind == "optimizer":
        options = hm.PathOptimizationOptions()
        options.state_cost_weight = 1.0
        result = owner.optimize([[0.2], [0.3], [0.5]], options, release_gil=release)
    else:
        result = owner.plan([0.2], [0.5], release_gil=release)
    assert result.success
    np.testing.assert_array_equal(result.path[0], [0.2])
    np.testing.assert_array_equal(result.path[-1], [0.5])


@requires_cpython_gc
def test_resource_cycle_survives_model_finalizer_resurrection(resource_factory):
    model, owner = resource_factory()
    rescued, finalizations = [], []

    def finalize(self):
        finalizations.append(True)
        rescued.append(self.owner)

    type(model).__del__ = finalize
    model.owner = owner
    del model, owner
    gc.collect()
    assert len(rescued) == 1
    owner = rescued[0]
    assert _query(owner, True).success
    resources = [
        value
        for value in gc.get_referents(owner)
        if isinstance(value, (hm.CollisionModel, hm.SphereCollisionModel))
    ]
    assert resources
    model_ref, owner_ref = weakref.ref(resources[0]), weakref.ref(owner)
    del resources, owner
    rescued.clear()
    gc.collect()
    assert model_ref() is None and owner_ref() is None
    assert finalizations == [True] and rescued == []
