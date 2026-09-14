"""Integration checks use the real optional dex VectorOptimizer and NLopt."""

import builtins
import gc
import subprocess
import sys
import weakref

import numpy as np
import pytest
from holistic_motion.kit.retargeting import DexHandRetargetingSolver


def hand_urdf(tmp_path, *, mimic=False, continuous=False, multiplier=-1.0, offset=0.4):
    path = tmp_path / "hand.urdf"
    kind = "continuous" if continuous else "prismatic"
    follower = (
        f"""<link name="follower"/>
      <joint name="a_follow" type="prismatic">
        <parent link="palm"/><child link="follower"/><axis xyz="1 0 0"/>
        <limit lower="0.1" upper="0.3" velocity="1" effort="1"/>
        <mimic joint="z_drive" multiplier="{multiplier}" offset="{offset}"/>
      </joint>"""
        if mimic
        else ""
    )
    path.write_text(f"""<robot name="hand">
      <link name="root"/><link name="palm"/><link name="tip"/>
      <joint name="wrist" type="prismatic">
        <parent link="root"/><child link="palm"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/>
      </joint>
      <joint name="z_drive" type="{kind}">
        <parent link="palm"/><child link="tip"/><axis xyz="1 0 0"/>
        <limit lower="0" upper="0.4" velocity="1" effort="1"/>
      </joint>{follower}</robot>""")
    return path


def make_solver(tmp_path, *, mimic=False, multiplier=-1.0, offset=0.4, **options):
    for name in ("nlopt", "pinocchio", "dex_retargeting"):
        pytest.importorskip(name, exc_type=ImportError)
    defaults = {"temporal_weight": 0.0, "vector_tolerance": 1e-5}
    defaults.update(options)
    return DexHandRetargetingSolver(
        hand_urdf(tmp_path, mimic=mimic, multiplier=multiplier, offset=offset),
        joint_names=["z_drive"],
        origin_links=["palm", "palm"] if mimic else ["palm"],
        task_links=["tip", "follower"] if mimic else ["tip"],
        keypoint_pairs=[[0, 1], [0, 2]] if mimic else [[0, 1]],
        **defaults,
    )


@pytest.mark.parametrize("mimic", [False, True])
@pytest.mark.parametrize("translation", [[0.0, 0.0, 0.0], [2.0, -3.0, 1.0]])
def test_real_vector_backend_preserves_named_fixed_and_mimic_joints(
    tmp_path, mimic, translation
):
    solver = make_solver(tmp_path, mimic=mimic, scaling=2.0)
    points = np.array([[0, 0, 0], [0.125, 0, 0], [0.075, 0, 0]]) + translation
    result = solver.solve(points, seed={"wrist": 0.7, "z_drive": 0.2})
    assert result.success
    assert result.optimizer_status > 0
    assert result.evaluations > 0
    assert result.residual <= solver.vector_tolerance
    assert result.joint_positions["wrist"] == 0.7
    assert result.joint_positions["z_drive"] == pytest.approx(0.25, abs=1e-5)
    if mimic:
        assert result.joint_positions["a_follow"] == pytest.approx(0.15, abs=1e-5)
    with pytest.raises(ValueError):
        result.configuration[0] = 0.0
    with pytest.raises(TypeError):
        result.joint_positions["wrist"] = 0.0
    np.testing.assert_array_equal(solver._last_q, result.configuration)
    solver.reset(result.joint_positions)
    again = solver.solve(points)
    assert again.success


@pytest.mark.parametrize("mimic", [False, True])
@pytest.mark.parametrize("context", ["enable_grad", "no_grad", "inference_mode"])
def test_corrected_temporal_objective_matches_its_gradient(tmp_path, mimic, context):
    solver = make_solver(tmp_path, mimic=mimic, temporal_weight=0.3)
    import torch

    reference = solver._configuration({"wrist": 0.2, "z_drive": 0.2})
    vectors = np.array([[0.27, 0.03, 0.0], [0.13, -0.02, 0.0]])[: 2 if mimic else 1]
    x = np.array([0.24])
    gradient = np.empty(1)
    with getattr(torch, context)():
        objective = solver._objective(vectors, reference)
        value = objective(x, gradient)
        numerical = (
            objective(x + 1e-6, np.empty(0)) - objective(x - 1e-6, np.empty(0))
        ) / 2e-6
    np.testing.assert_allclose(gradient, [numerical], atol=1e-8, rtol=1e-6)
    upstream = solver._optimizer.get_objective_function(
        vectors, reference[solver._optimizer.idx_pin2fixed], reference[solver._target]
    )
    assert value - upstream(x, np.empty(0)) == pytest.approx(0.3 * 0.04**2)


@pytest.mark.parametrize("mimic", [False, True])
@pytest.mark.parametrize("context", ["no_grad", "inference_mode"])
def test_solves_and_holds_inside_torch_inference_context(tmp_path, mimic, context):
    solver = make_solver(tmp_path, mimic=mimic)
    import torch

    points = [[0, 0, 0], [0.25, 0, 0], [0.15, 0, 0]]
    seed = {"wrist": 0.2, "z_drive": 0.2}
    baseline = solver.solve(points, seed=seed)
    assert baseline.success
    before = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
    with getattr(torch, context)():
        expected = (False, context == "inference_mode")
        result = solver.solve(points, seed=seed)
        assert result.success
        assert result.evaluations > 0
        np.testing.assert_array_equal(result.configuration, baseline.configuration)
        assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == expected
        held = solver.solve(points)
        assert held.success
        assert held.evaluations == held.optimizer_status == 0
        np.testing.assert_array_equal(held.configuration, result.configuration)
        assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == expected
    assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == before


@pytest.mark.parametrize("mimic", [False, True])
@pytest.mark.parametrize("context", ["enable_grad", "no_grad", "inference_mode"])
def test_hand_created_during_inference_can_solve(tmp_path, mimic, context):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    with torch.inference_mode():
        solver = make_solver(tmp_path, mimic=mimic)
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
    with getattr(torch, context)():
        result = solver.solve([[0, 0, 0], [0.25, 0, 0], [0.15, 0, 0]])
        assert result.success
        assert result.evaluations > 0
        assert result.joint_positions["z_drive"] == pytest.approx(0.25, abs=1e-5)
        assert torch.is_inference_mode_enabled() == (context == "inference_mode")
        assert torch.is_grad_enabled() == (context == "enable_grad")


@pytest.mark.parametrize("mimic", [False, True])
@pytest.mark.parametrize(
    "setting,value",
    [
        ("scaling", 2.0),
        ("temporal_weight", 0.3),
        ("huber_delta", 0.5),
        ("objective_tolerance", 1e-4),
    ],
)
def test_updated_hand_settings_match_fresh_solver(tmp_path, mimic, setting, value):
    solver = make_solver(tmp_path, mimic=mimic)
    points = [[0, 0, 0], [0.25, 0, 0], [0.15, 0, 0]]
    solver.solve(points)
    if setting == "scaling":
        points = [[0, 0, 0], [0.125, 0, 0], [0.075, 0, 0]]
    setattr(solver, setting, value)
    fresh = make_solver(tmp_path, mimic=mimic, **{setting: value})
    seed = {"wrist": 0.2, "z_drive": 0.2}
    actual = solver.solve(points, seed=seed)
    expected = fresh.solve(points, seed=seed)
    assert actual.success == expected.success
    assert actual.termination_reason == expected.termination_reason
    np.testing.assert_array_equal(actual.configuration, expected.configuration)
    assert actual.objective == pytest.approx(expected.objective)
    assert actual.residual == pytest.approx(expected.residual)
    assert solver._optimizer.opt.get_ftol_abs() == fresh._optimizer.opt.get_ftol_abs()

    vectors = np.array(points[1 : 3 if mimic else 2])
    reference = solver._configuration(seed)
    objective = solver._objective(vectors, reference)
    gradient = np.empty(1)
    x = np.array([0.24])
    value = objective(x, gradient)
    fresh_gradient = np.empty(1)
    fresh_value = fresh._objective(vectors, reference)(x, fresh_gradient)
    assert value == pytest.approx(fresh_value, rel=1e-12, abs=1e-14)
    np.testing.assert_allclose(gradient, fresh_gradient, atol=1e-12, rtol=1e-12)
    numerical = (
        objective(x + 1e-6, np.empty(0)) - objective(x - 1e-6, np.empty(0))
    ) / 2e-6
    np.testing.assert_allclose(gradient, [numerical], atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("scaling", float("nan")),
        ("temporal_weight", -1.0),
        ("huber_delta", 0.0),
        ("vector_tolerance", float("inf")),
        ("objective_tolerance", float("nan")),
    ],
)
def test_invalid_updated_settings_preserve_native_settings_and_history(
    tmp_path, setting, value
):
    solver = make_solver(tmp_path)
    before = solver._last_q.copy()
    native = solver._optimizer
    settings = (
        native.scaling,
        native.norm_delta,
        native.huber_loss.beta,
        native.opt.get_ftol_abs(),
    )
    solver.scaling = 2.0  # A valid pending update must not be partially applied.
    setattr(solver, setting, value)
    with pytest.raises(ValueError, match=setting):
        solver.solve([[0, 0, 0], [0.125, 0, 0]])
    np.testing.assert_array_equal(solver._last_q, before)
    assert (
        native.scaling,
        native.norm_delta,
        native.huber_loss.beta,
        native.opt.get_ftol_abs(),
    ) == settings


@pytest.mark.parametrize("context", ["no_grad", "inference_mode"])
def test_objective_failure_restores_torch_context(tmp_path, monkeypatch, context):
    solver = make_solver(tmp_path)
    import torch

    before = solver._last_q.copy()
    interruption = RuntimeError("hand objective failed")

    def fail(x, gradient):
        assert gradient.size > 0
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        raise interruption

    monkeypatch.setattr(solver._optimizer, "get_objective_function", lambda *args: fail)
    outer = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
    with getattr(torch, context)():
        with pytest.raises(RuntimeError, match="hand objective failed") as caught:
            solver.solve([[0, 0, 0], [0.25, 0, 0]])
        assert caught.value is interruption
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled() == (context == "inference_mode")
    assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == outer
    np.testing.assert_array_equal(solver._last_q, before)


def test_mimic_limits_restrict_the_source_without_upstream_padding(tmp_path):
    solver = make_solver(tmp_path, mimic=True, vector_tolerance=0.21)
    np.testing.assert_allclose(solver._optimizer.opt.get_lower_bounds(), [0.1])
    np.testing.assert_allclose(solver._optimizer.opt.get_upper_bounds(), [0.3])
    result = solver.solve([[0, 0, 0], [0.5, 0, 0], [-0.1, 0, 0]])
    assert result.success
    assert result.evaluations > 0
    assert result.joint_positions["z_drive"] <= 0.3 + 1e-15
    assert result.joint_positions["a_follow"] >= 0.1


@pytest.mark.parametrize("multiplier,offset", [(1.0, 0.0), (-1.0, 0.4), (0.0, 0.2)])
@pytest.mark.parametrize("target", [-1.0, 1.0])
def test_mimic_affine_limits_hold_at_both_extremes(
    tmp_path, multiplier, offset, target
):
    solver = make_solver(
        tmp_path, mimic=True, multiplier=multiplier, offset=offset, vector_tolerance=3.0
    )
    bound = solver._target_limits[0, 0 if target < 0 else 1]
    solver.vector_tolerance = (abs(target - 0.2) + abs(target - bound)) / 2.0
    result = solver.solve(
        [[0, 0, 0], [target, 0, 0], [multiplier * target + offset, 0, 0]],
        seed={"wrist": 0.0, "z_drive": 0.2},
    )
    assert result.success
    assert result.evaluations > 0
    assert result.joint_positions["z_drive"] == pytest.approx(bound)
    assert solver._valid(result.configuration)
    assert result.joint_positions["a_follow"] == (
        multiplier * result.joint_positions["z_drive"] + offset
    )


@pytest.mark.parametrize(
    "multiplier,offset", [(0.0, 0.5), (1.0, 1.0), (float("nan"), 0.0)]
)
def test_infeasible_mimic_model_is_rejected(tmp_path, multiplier, offset):
    with pytest.raises(ValueError, match="mimic"):
        make_solver(tmp_path, mimic=True, multiplier=multiplier, offset=offset)


@pytest.mark.parametrize(
    "multiplier,offset,source",
    [(5.0, 0.3, 0.34), (-5.0, -0.3, 0.34), (7.0, 0.1, 0.07), (-7.0, -0.1, 0.07)],
)
def test_locked_mimic_keeps_representable_source(tmp_path, multiplier, offset, source):
    pytest.importorskip("dex_retargeting", exc_type=ImportError)
    follower = multiplier * source + offset
    path = hand_urdf(tmp_path, mimic=True, multiplier=multiplier, offset=offset)
    path.write_text(
        path.read_text().replace(
            'lower="0.1" upper="0.3"', f'lower="{follower}" upper="{follower}"'
        )
    )
    solver = DexHandRetargetingSolver(
        path,
        joint_names=["z_drive"],
        origin_links=["palm", "palm"],
        task_links=["tip", "follower"],
        keypoint_pairs=[[0, 1], [0, 2]],
    )
    low, high = solver._target_limits[0]
    assert low <= source <= high
    for boundary in (low, high):
        solver.reset({"wrist": 0.0, "z_drive": boundary})
        assert solver._valid(solver._last_q)
        assert solver._last_q[solver.joint_names.index("a_follow")] == follower
    assert multiplier * np.nextafter(low, -np.inf) + offset != follower
    assert multiplier * np.nextafter(high, np.inf) + offset != follower
    result = solver.solve(
        [[0, 0, 0], [source, 0, 0], [follower, 0, 0]],
        seed={"wrist": 0.2, "z_drive": source},
    )
    assert result.success
    assert result.evaluations == 0
    assert result.joint_positions["z_drive"] == source
    assert result.joint_positions["a_follow"] == follower


@pytest.mark.parametrize("multiplier", [1e-310, -1e-310])
def test_tiny_mimic_multiplier_keeps_full_source_interval(tmp_path, multiplier):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        solver = make_solver(tmp_path, mimic=True, multiplier=multiplier, offset=0.2)
        np.testing.assert_array_equal(solver._target_limits, [[0.0, 0.4]])
        result = solver.solve([[0, 0, 0], [0.25, 0, 0], [0.2, 0, 0]])
    assert result.success
    assert result.joint_positions["z_drive"] == pytest.approx(0.25, abs=1e-5)
    assert result.joint_positions["a_follow"] == 0.2


@pytest.mark.parametrize(
    "multiplier,offset,center",
    [(5.0, 0.3, 0.34), (-5.0, -0.3, -0.34), (3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)],
)
@pytest.mark.parametrize("locked", [False, True])
def test_mimic_interval_matches_exhaustive_neighbor_enumeration(
    multiplier, offset, center, locked
):
    from holistic_motion.kit.retargeting.dex_hand import _mimic_source_limits

    below, above = [], []
    left = right = center
    for _ in range(64):
        left = np.nextafter(left, -np.inf)
        right = np.nextafter(right, np.inf)
        below.append(left)
        above.append(right)
    sources = np.array([*reversed(below), center, *above])
    followers = multiplier * sources + offset
    selected = followers[64:65] if locked else followers[32:97]
    low, high = selected.min(), selected.max()
    valid = sources[(followers >= low) & (followers <= high)]
    actual = _mimic_source_limits(
        (sources[0], sources[-1]), (low, high), multiplier, offset
    )
    np.testing.assert_array_equal(actual, [valid[0], valid[-1]])


@pytest.mark.parametrize("multiplier", [3.0, -3.0])
def test_mimic_rejects_locked_value_between_representable_outputs(multiplier):
    from holistic_motion.kit.retargeting.dex_hand import _mimic_source_limits

    tiny = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="no feasible intersection"):
        _mimic_source_limits((-1.0, 1.0), (tiny, tiny), multiplier, 0.0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_mimic_large_finite_coefficients_keep_exact_bounds(sign):
    from holistic_motion.kit.retargeting.dex_hand import _mimic_source_limits

    maximum = np.finfo(float).max
    multiplier = sign * float(maximum)
    low, high = _mimic_source_limits((-maximum, maximum), (-0.1, 0.3), multiplier, 0.0)
    assert low <= 0.0 <= high
    for boundary in (low, high):
        assert -0.1 <= multiplier * boundary <= 0.3
    for outside in (np.nextafter(low, -np.inf), np.nextafter(high, np.inf)):
        assert not -0.1 <= multiplier * outside <= 0.3


@pytest.mark.parametrize("angle", [-0.8, 0.3, 1.0])
def test_revolute_finger_uses_real_rotational_jacobian(tmp_path, angle):
    pytest.importorskip("dex_retargeting", exc_type=ImportError)
    path = tmp_path / "finger.urdf"
    path.write_text("""<robot name="finger">
      <link name="palm"/><link name="finger"/><link name="tip"/>
      <joint name="flex" type="revolute">
        <parent link="palm"/><child link="finger"/><axis xyz="0 0 1"/>
        <limit lower="-1.5" upper="1.5" velocity="1" effort="1"/>
      </joint>
      <joint name="tip_fixed" type="fixed">
        <parent link="finger"/><child link="tip"/><origin xyz="0.1 0 0"/>
      </joint></robot>""")
    solver = DexHandRetargetingSolver(
        path,
        joint_names=["flex"],
        origin_links=["palm"],
        task_links=["tip"],
        keypoint_pairs=[[0, 1]],
        temporal_weight=0.0,
        vector_tolerance=1e-5,
    )
    result = solver.solve([[0, 0, 0], [0.1 * np.cos(angle), 0.1 * np.sin(angle), 0]])
    assert result.success
    assert result.joint_positions["flex"] == pytest.approx(angle, abs=1e-4)


@pytest.mark.parametrize(
    "seed",
    [
        {"z_drive": 0.2},
        {"wrist": 0.0, "z_drive": 0.2, "unknown": 0.0},
        {"wrist": 0.0, "z_drive": float("nan")},
        {"wrist": 0.0, "z_drive": 0.5},
        {"wrist": 0.0, "z_drive": 0.2, "a_follow": 0.3},
    ],
)
def test_invalid_seed_is_rejected_without_changing_history(tmp_path, seed):
    solver = make_solver(tmp_path, mimic=True)
    before = solver._last_q.copy()
    with pytest.raises(ValueError):
        solver.solve([[0, 0, 0], [0.2, 0, 0], [0.2, 0, 0]], seed=seed)
    np.testing.assert_array_equal(solver._last_q, before)


@pytest.mark.parametrize(
    "points", [[[0, 0]], [[0, 0, 0]], [[0, 0, 0], [float("nan"), 0, 0]]]
)
def test_invalid_keypoints_are_rejected(tmp_path, points):
    solver = make_solver(tmp_path)
    with pytest.raises(ValueError):
        solver.solve(points)


def test_budget_exhaustion_is_not_reported_as_success(tmp_path):
    solver = make_solver(tmp_path, max_evaluations=1)
    before = solver._last_q.copy()
    result = solver.solve([[0, 0, 0], [0.3, 0, 0]])
    assert not result.success
    assert result.termination_reason == "maximum_evaluations"
    assert result.optimizer_status == solver._nlopt.MAXEVAL_REACHED
    np.testing.assert_array_equal(result.configuration, before)
    np.testing.assert_array_equal(solver._last_q, before)


def test_unreachable_vector_does_not_update_warm_start(tmp_path):
    solver = make_solver(tmp_path)
    solver.reset({"wrist": 0.2, "z_drive": 0.1})
    before = solver._last_q.copy()
    result = solver.solve([[0, 0, 0], [0.3, 1, 0]], seed={"wrist": 0.4, "z_drive": 0.2})
    assert not result.success
    assert result.termination_reason == "task_tolerance"
    assert result.joint_positions == {"wrist": 0.4, "z_drive": 0.2}
    np.testing.assert_array_equal(solver._last_q, before)


def test_native_optimizer_failure_has_explicit_diagnostics(tmp_path, monkeypatch):
    solver = make_solver(tmp_path)
    before = solver._last_q.copy()

    def fail(*args):
        raise RuntimeError("native solver failed")

    monkeypatch.setattr(solver._optimizer.opt, "optimize", fail)
    result = solver.solve([[0, 0, 0], [0.2, 0, 0]])
    assert not result.success
    assert result.termination_reason == "optimizer_failed"
    assert "native solver failed" in result.message
    np.testing.assert_array_equal(solver._last_q, before)


@pytest.mark.parametrize("outcome", ["success", "budget", "unreachable"])
@pytest.mark.parametrize("retain_native", [False, True])
def test_hand_resources_are_collectable_after_optimization(
    tmp_path, outcome, retain_native
):
    def run():
        solver = make_solver(
            tmp_path, max_evaluations=1 if outcome == "budget" else 100
        )
        result = solver.solve(
            [[0, 0, 0], [0.25, 1.0 if outcome == "unreachable" else 0.0, 0]]
        )
        assert result.success == (outcome == "success")
        assert result.evaluations > 0
        references = [
            weakref.ref(value) for value in (solver, solver._optimizer, solver._robot)
        ]
        native = solver._optimizer.opt if retain_native else None
        return references, native, result

    references, native, result = run()
    gc.collect()
    assert all(reference() is None for reference in references)
    if native is not None:
        assert native.last_optimize_result() == result.optimizer_status
        assert native.get_numevals() == result.evaluations


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, KeyboardInterrupt])
def test_hand_resources_are_collectable_after_callback_error(
    tmp_path, monkeypatch, error_type
):
    def run():
        solver = make_solver(tmp_path)
        before = solver._last_q.copy()

        def fail(*args):
            raise error_type("callback aborted")

        with monkeypatch.context() as patch:
            patch.setattr(
                solver._optimizer, "get_objective_function", lambda *args: fail
            )
            with pytest.raises(error_type, match="callback aborted"):
                solver.solve([[0, 0, 0], [0.25, 0, 0]])
        np.testing.assert_array_equal(solver._last_q, before)
        return weakref.ref(solver), weakref.ref(solver._robot), solver._optimizer.opt

    solver_ref, robot_ref, native = run()
    gc.collect()
    assert solver_ref() is None
    assert robot_ref() is None
    assert native.last_optimize_result() < 0


def test_each_frame_releases_objective_and_later_frames_still_solve(
    tmp_path, monkeypatch
):
    solver = make_solver(tmp_path)
    original = solver._objective
    references = []

    def tracked(*args):
        objective = original(*args)
        references.append(weakref.ref(objective))
        return objective

    monkeypatch.setattr(solver, "_objective", tracked)
    for target, expected_evaluations in [(0.25, True), (0.25, False), (0.1, True)]:
        result = solver.solve([[0, 0, 0], [target, 0, 0]])
        assert result.success
        assert (result.evaluations > 0) == expected_evaluations
        gc.collect()
        assert all(reference() is None for reference in references)


def test_callback_exception_is_not_swallowed(tmp_path, monkeypatch):
    solver = make_solver(tmp_path)
    before = solver._last_q.copy()

    def fail(*args):
        raise RuntimeError("objective failed")

    monkeypatch.setattr(solver._optimizer, "get_objective_function", lambda *args: fail)
    with pytest.raises(RuntimeError, match="objective failed"):
        solver.solve([[0, 0, 0], [0.2, 0, 0]])
    np.testing.assert_array_equal(solver._last_q, before)


@pytest.mark.parametrize("values", [[float("nan")], [1.0], [0.2, 0.3]])
def test_invalid_native_output_is_not_published(tmp_path, monkeypatch, values):
    solver = make_solver(tmp_path)
    before = solver._last_q.copy()
    monkeypatch.setattr(solver._optimizer.opt, "optimize", lambda x: np.array(values))
    if len(values) != 1:
        with pytest.raises(ValueError, match="shape"):
            solver.solve([[0, 0, 0], [0.2, 0, 0]])
    else:
        result = solver.solve([[0, 0, 0], [0.2, 0, 0]])
        assert not result.success
        assert result.termination_reason == "invalid_result"
    np.testing.assert_array_equal(solver._last_q, before)


def test_whole_body_continuous_model_is_rejected(tmp_path):
    pytest.importorskip("dex_retargeting", exc_type=ImportError)
    with pytest.raises(ValueError, match="scalar-joint"):
        DexHandRetargetingSolver(
            hand_urdf(tmp_path, continuous=True),
            joint_names=["z_drive"],
            origin_links=["palm"],
            task_links=["tip"],
            keypoint_pairs=[[0, 1]],
        )


def test_import_keeps_hand_dependencies_optional():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from holistic_motion.kit.retargeting import DexHandRetargetingSolver; import sys; assert all(name not in sys.modules for name in ('torch', 'nlopt', 'dex_retargeting'))",
        ],
        check=True,
    )


def test_missing_optional_dependency_has_install_hint(tmp_path, monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "nlopt":
            raise ImportError("missing nlopt")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="hand-retargeting"):
        DexHandRetargetingSolver(
            hand_urdf(tmp_path),
            joint_names=["z_drive"],
            origin_links=["palm"],
            task_links=["tip"],
            keypoint_pairs=[[0, 1]],
        )


@pytest.mark.parametrize(
    "pairs", [[[0.0, 1.0]], [[True, False]], [[-1, 1]], [[0, 0]], [[0, 1, 2]]]
)
def test_invalid_keypoint_pairs_are_rejected_before_loading_backend(tmp_path, pairs):
    with pytest.raises(ValueError):
        DexHandRetargetingSolver(
            hand_urdf(tmp_path),
            joint_names=["z_drive"],
            origin_links=["palm"],
            task_links=["tip"],
            keypoint_pairs=pairs,
        )


@pytest.mark.parametrize("name", ["missing", "z_drive"])
def test_joint_or_unknown_frame_is_rejected_before_native_lookup(tmp_path, name):
    with pytest.raises(ValueError, match="URDF links"):
        DexHandRetargetingSolver(
            hand_urdf(tmp_path),
            joint_names=["z_drive"],
            origin_links=["palm"],
            task_links=[name],
            keypoint_pairs=[[0, 1]],
        )


def test_coupled_finger_converges_with_tight_vector_tolerance(tmp_path):
    pytest.importorskip("dex_retargeting", exc_type=ImportError)
    path = tmp_path / "serial_finger.urdf"
    pieces = ['<robot name="finger"><link name="base"/>']
    for i, length in enumerate((0.0, 0.04, 0.03)):
        parent = "base" if i == 0 else f"l{i - 1}"
        pieces.append(f'''<link name="l{i}"/><joint name="j{i}" type="revolute">
          <parent link="{parent}"/><child link="l{i}"/>
          <origin xyz="{length} 0 0"/><axis xyz="0 0 1"/>
          <limit lower="0" upper="1.5" effort="1" velocity="1"/></joint>''')
    pieces.append("""<link name="tip"/><joint name="fixed" type="fixed">
      <parent link="l2"/><child link="tip"/><origin xyz="0.02 0 0"/>
      </joint></robot>""")
    path.write_text("".join(pieces))
    angles = np.cumsum([0.4, 0.5, 0.3])
    endpoint = [
        np.dot([0.04, 0.03, 0.02], np.cos(angles)),
        np.dot([0.04, 0.03, 0.02], np.sin(angles)),
        0.0,
    ]
    solver = DexHandRetargetingSolver(
        path,
        joint_names=["j2", "j0", "j1"],
        origin_links=["base"],
        task_links=["tip"],
        keypoint_pairs=[[0, 1]],
        temporal_weight=0.0,
        vector_tolerance=1e-6,
    )
    result = solver.solve([[0, 0, 0], endpoint])
    assert result.success
    assert result.residual < 1e-6
    assert solver._valid(result.configuration)


@pytest.mark.parametrize("mimic", [False, True])
@pytest.mark.parametrize("max_evaluations", [1, 100])
@pytest.mark.parametrize("explicit_seed", [False, True])
def test_satisfied_frame_holds_without_native_optimization(
    tmp_path, monkeypatch, mimic, max_evaluations, explicit_seed
):
    solver = make_solver(tmp_path, mimic=mimic, max_evaluations=max_evaluations)
    solver.reset({"wrist": 0.4, "z_drive": 0.2})
    seed = {"wrist": 0.7, "z_drive": 0.25} if explicit_seed else None
    reference = solver._configuration(seed)
    drive = reference[solver.joint_names.index("z_drive")]
    # A small nonzero tracking error should hold within the requested tolerance.
    points = [
        [0, 0, 0],
        [drive + 0.25 * solver.vector_tolerance, 0, 0],
        [0.4 - drive, 0, 0],
    ]

    def forbidden(*args):
        pytest.fail("a satisfied hand frame must not call NLopt")

    monkeypatch.setattr(solver._optimizer.opt, "optimize", forbidden)
    result = solver.solve(points, seed=seed)
    assert result.success
    assert result.termination_reason == "converged"
    assert result.optimizer_status == 0
    assert result.evaluations == 0
    assert 0.0 < result.residual <= solver.vector_tolerance
    assert result.objective > 0.0
    np.testing.assert_array_equal(result.configuration, reference)
    np.testing.assert_array_equal(solver._last_q, reference)


@pytest.mark.parametrize("lock", ["source", "source_with_mimic", "mimic"])
@pytest.mark.parametrize("satisfied", [False, True])
def test_locked_hand_reports_feasibility_without_native_optimization(
    tmp_path, monkeypatch, lock, satisfied
):
    pytest.importorskip("dex_retargeting", exc_type=ImportError)
    path = hand_urdf(tmp_path, mimic=lock != "source")
    original_bounds = (
        'lower="0.1" upper="0.3"' if lock == "mimic" else 'lower="0" upper="0.4"'
    )
    path.write_text(
        path.read_text().replace(original_bounds, 'lower="0.2" upper="0.2"')
    )
    solver = DexHandRetargetingSolver(
        path,
        joint_names=["z_drive"],
        origin_links=["palm"],
        task_links=["tip"],
        keypoint_pairs=[[0, 1]],
        max_evaluations=1,
    )
    before = solver._last_q.copy()
    seed = {"wrist": 0.7, "z_drive": 0.2}
    reference = solver._configuration(seed)

    def forbidden(*args):
        pytest.fail("a locked hand must not call NLopt")

    monkeypatch.setattr(solver._optimizer.opt, "optimize", forbidden)
    result = solver.solve([[0, 0, 0], [0.2 if satisfied else 0.3, 0, 0]], seed=seed)
    assert result.success == satisfied
    assert result.termination_reason == (
        "converged" if satisfied else "no_feasible_motion"
    )
    assert result.optimizer_status == 0
    assert result.evaluations == 0
    assert (result.residual <= solver.vector_tolerance) == satisfied
    np.testing.assert_array_equal(result.configuration, reference)
    np.testing.assert_array_equal(solver._last_q, reference if satisfied else before)


@pytest.mark.parametrize("previous_failure", [False, True])
def test_hold_does_not_reuse_native_diagnostics_or_compute_jacobians(
    tmp_path, monkeypatch, previous_failure
):
    solver = make_solver(tmp_path)
    previous = solver.solve([[0, 0, 0], [0.25, 1.0 if previous_failure else 0.0, 0]])
    assert previous.success != previous_failure
    assert previous.evaluations > 0
    assert previous.optimizer_status != 0
    drive = solver._last_q[solver.joint_names.index("z_drive")]

    def forbidden(*args):
        pytest.fail("hold must neither optimize nor compute a Jacobian")

    with monkeypatch.context() as patch:
        patch.setattr(solver._optimizer.opt, "optimize", forbidden)
        patch.setattr(solver._robot, "compute_single_link_local_jacobian", forbidden)
        result = solver.solve([[0, 0, 0], [drive, 0, 0]])
    assert result.success
    assert result.evaluations == result.optimizer_status == 0
    # A later moving frame resumes actual optimization using the held seed.
    resumed = solver.solve([[0, 0, 0], [0.35, 0, 0]])
    assert resumed.success
    assert resumed.evaluations > 0


def test_hold_objective_exception_preserves_previous_history(tmp_path, monkeypatch):
    solver = make_solver(tmp_path)
    before = solver._last_q.copy()

    def fail(*args):
        raise RuntimeError("hold objective failed")

    monkeypatch.setattr(solver._optimizer, "get_objective_function", lambda *args: fail)
    with pytest.raises(RuntimeError, match="hold objective failed"):
        solver.solve([[0, 0, 0], [0.2, 0, 0]], seed={"wrist": 0.4, "z_drive": 0.2})
    np.testing.assert_array_equal(solver._last_q, before)
