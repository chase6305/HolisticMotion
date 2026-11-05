"""cuRobo adapter contracts, using real CPU tensors without installing cuRobo."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from holistic_motion.kit.curobo_planner import CuroboPlanner, CuroboTrajectory


@pytest.fixture
def backend(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    instances = []
    configs = []

    class Scene:
        def __init__(self, data=None):
            self.data = data

        @classmethod
        def create(cls, data):
            return cls(data)

    class JointState:
        @staticmethod
        def from_position(position, joint_names):
            return SimpleNamespace(position=position, joint_names=joint_names)

    class Pose:
        @staticmethod
        def from_matrix(matrix):
            return matrix

    class GoalToolPose:
        @staticmethod
        def from_poses(poses):
            return poses

    class MotionPlanner:
        joint_names = ("shoulder", "slide")
        tool_frames = ("tool",)
        default_joint_state = SimpleNamespace(position=torch.zeros(2))

        def __init__(self, config):
            self.calls = []
            self.destroyed = False
            self.scene = config["scene_model"]
            self.resets = 0
            self.failure = None
            self.state = None
            # Limits have a different order from both input and backend order.
            limits = SimpleNamespace(
                joint_names=["slide", "shoulder"],
                position=torch.tensor([[-2.0, -1.0], [2.0, 1.0]]),
            )
            self.kinematics = SimpleNamespace(
                get_joint_limits=lambda: limits,
                kinematics_config=SimpleNamespace(link_name_to_idx_map={"payload": 0}),
            )
            self.inertial_updates = []
            instances.append(self)

        def plan_cspace(self, goal, start, **kwargs):
            self.calls.append((goal, start, kwargs))
            if self.failure == "none":
                return None
            if self.failure == "failed":
                return SimpleNamespace(
                    success=torch.tensor([[False]]),
                    status="INFEASIBLE",
                    debug_info={
                        "validation": {
                            "collision_valid": torch.tensor([False]),
                            "path_valid": torch.tensor([True]),
                        }
                    },
                )
            # The public trim helper preserves singleton batch and seed axes
            # and includes locked joints in a different output order.
            positions = torch.stack((start.position[0], goal.position[0]))
            self.state = SimpleNamespace(
                joint_names=["slide", "locked", "shoulder"],
                position=torch.stack(
                    (positions[:, 1], torch.ones(2), positions[:, 0]), dim=-1
                )[None, None],
                velocity=torch.tensor([[[[2.0, 0.0, 3.0], [4.0, 0.0, 5.0]]]]),
                acceleration=torch.zeros(1, 1, 2, 3),
                jerk=torch.ones(1, 1, 2, 3),
                dt=torch.tensor([[0.125]]),
            )
            if callable(self.failure):
                self.failure(self.state)
            return SimpleNamespace(
                success=torch.tensor([[True]]), get_interpolated_plan=lambda: self.state
            )

        def update_world(self, scene):
            self.scene = scene

        def compute_kinematics(self, state):
            matrix = torch.eye(4)[None]
            matrix[0, :2, 3] = state.position[0]
            pose = SimpleNamespace(get_matrix=lambda: matrix)
            return SimpleNamespace(
                tool_poses=SimpleNamespace(get_link_pose=lambda _name: pose)
            )

        def plan_pose(self, poses, start, **kwargs):
            target = poses["tool"]
            goal = JointState.from_position(target[None, :2, 3], self.joint_names)
            return self.plan_cspace(goal, start, **kwargs)

        def reset_seed(self):
            self.resets += 1

        def update_link_inertial(self, *args):
            self.inertial_updates.append(args)

        def warmup(self, **kwargs):
            self.warmup_options = kwargs

        def destroy(self):
            self.destroyed = True

    class MotionPlannerCfg:
        @staticmethod
        def create(**kwargs):
            configs.append(kwargs)
            return kwargs

    for name, members in {
        "curobo": {},
        "curobo.motion_planner": {
            "MotionPlanner": MotionPlanner,
            "MotionPlannerCfg": MotionPlannerCfg,
        },
        "curobo.scene": {"Scene": Scene},
        "curobo.trajectory_timeopt": {"TrajectoryTimeOpt": object},
        "curobo.types": {
            "DeviceCfg": SimpleNamespace,
            "JointState": JointState,
            "Pose": Pose,
            "GoalToolPose": GoalToolPose,
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    path = tmp_path / "robot.yml"
    path.write_text("robot_cfg: {}\n", encoding="utf-8")
    return SimpleNamespace(path=path, instances=instances, configs=configs, torch=torch)


def test_curobo_module_and_kit_import_without_optional_dependencies():
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "python")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['curobo'] = None
from holistic_motion.kit import curobo_planner
assert curobo_planner.CuroboPlanner is not None
assert 'warp' not in sys.modules
assert 'pinocchio' not in sys.modules
assert '_holistic_motion' not in sys.modules
""",
        ],
        env=environment,
        check=True,
    )


def test_name_permutations_trimmed_timing_and_owned_outputs(backend):
    planner = CuroboPlanner(backend.path, joint_names=["slide", "shoulder"])
    np.testing.assert_array_equal(planner.lower_limits, [-2.0, -1.0])
    start, goal = np.array([0.2, 0.3]), np.array([0.4, 0.5])
    result = planner.plan(start, goal, max_attempts=3)
    assert result.success, result.message
    native = backend.instances[0]
    actual_goal, actual_start, options = native.calls[0]
    np.testing.assert_allclose(actual_start.position, [[0.3, 0.2]])
    np.testing.assert_allclose(actual_goal.position, [[0.5, 0.4]])
    assert actual_goal.joint_names == ["shoulder", "slide"]
    assert options["max_attempts"] == 3
    trajectory = result.trajectory
    assert trajectory.joint_names == ("slide", "shoulder")
    np.testing.assert_allclose(trajectory.position, [start, goal])
    np.testing.assert_array_equal(trajectory.velocity, [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_array_equal(trajectory.time, [0.0, 0.125])
    assert trajectory.duration == 0.125
    native.state.position.fill_(99)
    start[:] = 99
    assert planner.plan([0.0, 0.0], [0.1, 0.1]).success
    planner.close()
    np.testing.assert_allclose(trajectory.position, [[0.2, 0.3], [0.4, 0.5]])
    assert not trajectory.position.flags.writeable
    assert not trajectory.time.flags.writeable
    assert backend.configs[0]["enable_timeopt"] is True
    assert backend.configs[0]["self_collision_check"] is True
    assert backend.configs[0]["graph_planner_algorithm"] == "rrt"
    assert backend.configs[0]["robot"]["load_dynamics"] is True


@pytest.mark.parametrize(
    "failure, message", [("none", "no solution"), ("failed", "INFEASIBLE")]
)
def test_unsuccessful_results_never_expose_a_trajectory(backend, failure, message):
    with CuroboPlanner(backend.path) as planner:
        backend.instances[0].failure = failure
        result = planner.plan([0.0, 0.0], [0.2, 0.3])
        assert not result.success and result.trajectory is None
        assert message in result.message
        if failure == "failed":
            assert "failed checks: collision" in result.message
            assert "path/pose" not in result.message


@pytest.mark.parametrize(
    "value, error",
    [
        ([complex(0, 0), 0.0], TypeError),
        ([float("nan"), 0.0], ValueError),
        ([1.1, 0.0], ValueError),
        ([0.0], ValueError),
        ([[0.0, 0.0]], ValueError),
    ],
)
def test_invalid_coordinates_do_not_reach_the_backend(backend, value, error):
    with CuroboPlanner(backend.path) as planner:
        with pytest.raises(error):
            planner.plan(value, [0.0, 0.0])
        assert backend.instances[0].calls == []


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda state: state.position.fill_(float("nan")), "finite"),
        (lambda state: state.velocity.fill_(float("inf")), "finite"),
        (lambda state: state.position.fill_(0.9), "endpoints"),
        (lambda state: setattr(state, "dt", None), "timed"),
        (lambda state: state.dt.fill_(0.0), "stationary"),
        (
            lambda state: setattr(state, "joint_names", ["x", "locked", "shoulder"]),
            "missing",
        ),
        (
            lambda state: setattr(state, "position", state.position.repeat(2, 1, 1, 1)),
            "shape",
        ),
    ],
)
def test_backend_output_is_checked_before_success(backend, edit, message):
    with CuroboPlanner(backend.path) as planner:
        backend.instances[0].failure = edit
        result = planner.plan([0.0, 0.0], [0.2, 0.3])
        assert not result.success and result.trajectory is None
        assert message in result.message


def test_scene_snapshot_replacement_resets_search_and_close_is_idempotent(backend):
    scene = {"cuboid": {"wall": {"dims": [1.0, 2.0, 3.0]}}}
    planner = CuroboPlanner(backend.path, scene=scene)
    scene["cuboid"]["wall"]["dims"][0] = 100
    native = backend.instances[0]
    assert native.scene.data["cuboid"]["wall"]["dims"][0] == 1.0
    planner.update_world({})
    assert native.scene.data == {} and native.resets == 1
    planner.warmup(3)
    assert native.warmup_options == {"enable_graph": True, "num_warmup_iterations": 3}
    planner.close()
    planner.close()
    assert native.destroyed
    with pytest.raises(RuntimeError, match="closed"):
        planner.plan([0.0, 0.0], [0.2, 0.3])


@pytest.mark.parametrize(
    "names", [["shoulder"], ["shoulder", "unknown"], ["slide", "slide"]]
)
def test_joint_set_must_match_and_failed_construction_releases_backend(backend, names):
    with pytest.raises(ValueError):
        CuroboPlanner(backend.path, joint_names=names)
    assert all(instance.destroyed for instance in backend.instances)


def test_trajectory_constructor_owns_and_validates_samples():
    source = np.array([[0.0], [1.0]])
    trajectory = CuroboTrajectory(("slide",), [0.0, 1.0], source, source, source)
    source[:] = 100
    np.testing.assert_array_equal(trajectory.position, [[0.0], [1.0]])
    with pytest.raises(ValueError, match="increase strictly"):
        CuroboTrajectory(("slide",), [0.0, 0.0], source, source, source)


def test_pose_planning_preserves_frames_order_and_validates_final_fk(backend):
    with CuroboPlanner(backend.path, joint_names=["slide", "shoulder"]) as planner:
        target = planner.forward_kinematics([0.4, 0.3])
        np.testing.assert_allclose(target["tool"][:3, 3], [0.3, 0.4, 0.0])
        result = planner.plan_pose([0.0, 0.0], target)
        assert result.success, result.message
        np.testing.assert_allclose(result.trajectory.position[-1], [0.4, 0.3])
        target["tool"][:3, :3] = np.diag([-1.0, -1.0, 1.0])
        rejected = planner.plan_pose([0.0, 0.0], target)
        assert not rejected.success and rejected.trajectory is None
        assert "tool target" in rejected.message


def test_invalid_pose_and_unknown_tools_do_not_reach_planning(backend):
    with CuroboPlanner(backend.path) as planner:
        with pytest.raises(ValueError, match="every configured tool"):
            planner.plan_pose([0.0, 0.0], {"wrong": np.eye(4)})
        target = np.eye(4)
        target[0, 0] = 2.0
        with pytest.raises(ValueError, match="rotation"):
            planner.plan_pose([0.0, 0.0], {"tool": target})
        assert backend.instances[0].calls == []


def test_failed_world_update_disables_planning_until_replaced(backend, monkeypatch):
    with CuroboPlanner(backend.path) as planner:
        native = backend.instances[0]
        update = native.update_world

        def partial_update(_scene):
            native.scene = None
            raise RuntimeError("scene capacity exceeded")

        monkeypatch.setattr(native, "update_world", partial_update)
        with pytest.raises(RuntimeError, match="capacity"):
            planner.update_world({})
        assert native.resets == 1
        with pytest.raises(RuntimeError, match="failed world update"):
            planner.plan([0.0, 0.0], [0.1, 0.1])
        monkeypatch.setattr(native, "update_world", update)
        planner.update_world({})
        assert planner.plan([0.0, 0.0], [0.1, 0.1]).success


def test_scene_file_is_parsed_for_both_construction_and_updates(backend, tmp_path):
    scene = tmp_path / "scene.yml"
    scene.write_text("cuboid: {}\n", encoding="utf-8")
    with CuroboPlanner(backend.path, scene=scene) as planner:
        assert backend.instances[0].scene.data == {"cuboid": {}}
        planner.update_world(scene)
        assert backend.instances[0].scene.data == {"cuboid": {}}


def test_inertial_updates_keep_units_order_and_reset_search(backend):
    with CuroboPlanner(backend.path) as planner:
        planner.update_link_inertial(
            "payload",
            mass=2.0,
            com=[0.1, 0.2, 0.3],
            inertia=[1.0, 2.0, 2.0, 0.1, 0.0, 0.0],
        )
        native = backend.instances[0]
        link, mass, com, inertia = native.inertial_updates[0]
        assert link == "payload" and mass == 2.0
        np.testing.assert_allclose(com, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(inertia, [1.0, 2.0, 2.0, 0.1, 0.0, 0.0])
        assert native.resets == 1


@pytest.mark.parametrize(
    "mass, inertia",
    [
        (-1.0, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        (1.0, [-1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        (1.0, [1.0, 1.0, 3.0, 0.0, 0.0, 0.0]),
        (1.0, [1.0, 1.0, 1.0, float("nan"), 0.0, 0.0]),
    ],
)
def test_invalid_inertial_updates_do_not_touch_the_backend(backend, mass, inertia):
    with CuroboPlanner(backend.path) as planner:
        with pytest.raises(ValueError):
            planner.update_link_inertial(
                "payload", mass=mass, com=[0.0, 0.0, 0.0], inertia=inertia
            )
        assert backend.instances[0].inertial_updates == []
        assert planner.plan([0.0, 0.0], [0.1, 0.1]).success


def test_explicit_rrt_attempt_policy_is_forwarded_and_validated(backend):
    with CuroboPlanner(backend.path) as planner:
        assert planner.plan([0.0, 0.0], [0.1, 0.1], enable_graph_attempt=0).success
        assert backend.instances[0].calls[-1][2]["enable_graph_attempt"] == 0
        for value, exception in ((True, TypeError), (-1, ValueError), (0.5, TypeError)):
            with pytest.raises(exception):
                planner.plan([0.0, 0.0], [0.1, 0.1], enable_graph_attempt=value)


def test_exact_stationary_zero_timestep_is_an_owned_single_instant(backend):
    with CuroboPlanner(backend.path) as planner:

        def stationary(state):
            state.dt.zero_()
            state.velocity.zero_()
            state.acceleration.zero_()
            state.jerk.zero_()

        backend.instances[0].failure = stationary
        result = planner.plan([0.0, 0.0], [0.0, 0.0])
        assert result.success, result.message
        assert result.trajectory.duration == 0.0
        assert result.trajectory.time.size == 1
        np.testing.assert_array_equal(result.trajectory.position, [[0.0, 0.0]])


def test_cpu_device_and_missing_v2_api_fail_before_backend_construction(
    backend, monkeypatch
):
    with pytest.raises(RuntimeError, match="CUDA device"):
        CuroboPlanner(backend.path, device="cpu")
    monkeypatch.setitem(sys.modules, "curobo.trajectory_timeopt", None)
    with pytest.raises(ImportError, match="cuRobo v2"):
        CuroboPlanner(backend.path)
    assert backend.instances == []


def test_world_update_waits_for_active_solve_and_preserves_its_output(
    backend, monkeypatch
):
    with CuroboPlanner(backend.path) as planner:
        native = backend.instances[0]
        entered, release, update_entered = Event(), Event(), Event()
        original_plan, original_update = native.plan_cspace, native.update_world

        def blocked_plan(*args, **kwargs):
            entered.set()
            assert release.wait(5.0)
            return original_plan(*args, **kwargs)

        def update(scene):
            update_entered.set()
            return original_update(scene)

        monkeypatch.setattr(native, "plan_cspace", blocked_plan)
        monkeypatch.setattr(native, "update_world", update)
        with ThreadPoolExecutor(2) as workers:
            query = workers.submit(planner.plan, [0.0, 0.0], [0.1, 0.1])
            try:
                assert entered.wait(5.0)
                edit = workers.submit(planner.update_world, {})
                assert not update_entered.wait(0.05)
            finally:
                release.set()
            result = query.result(timeout=5.0)
            edit.result(timeout=5.0)
        assert result.success and update_entered.is_set()
        np.testing.assert_allclose(result.trajectory.position, [[0.0, 0.0], [0.1, 0.1]])


def test_partial_inertial_update_failure_requires_reconstruction(backend, monkeypatch):
    with CuroboPlanner(backend.path) as planner:

        def failed_update(*_args):
            raise RuntimeError("partial dynamics update")

        monkeypatch.setattr(backend.instances[0], "update_link_inertial", failed_update)
        with pytest.raises(RuntimeError, match="partial dynamics"):
            planner.update_link_inertial(
                "payload",
                mass=1.0,
                com=[0.0, 0.0, 0.0],
                inertia=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            )
        with pytest.raises(RuntimeError, match="recreating"):
            planner.plan([0.0, 0.0], [0.1, 0.1])
