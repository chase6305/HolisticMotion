"""cuRobo adapter contracts, using real CPU tensors without installing cuRobo."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import MappingProxyType, ModuleType, SimpleNamespace

import numpy as np
import pytest
from holistic_motion.kit.curobo_planner import CuroboPlanner, CuroboTrajectory


@pytest.fixture
def backend(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: None)
    instances = []
    configs = []

    class VoxelGrid(SimpleNamespace):
        def get_grid_shape(self):
            return [round(dim / self.voxel_size) for dim in self.dims], None, None

    class Scene:
        def __init__(self, data=None, **kwargs):
            for kind in ("cuboid", "sphere", "capsule", "cylinder", "mesh", "voxel"):
                obstacle_type = VoxelGrid if kind == "voxel" else SimpleNamespace
                values = [
                    obstacle_type(name=name, **fields)
                    for name, fields in (data or {}).get(kind, {}).items()
                ]
                setattr(self, kind, kwargs.get(kind, values))
            self.objects = [
                value
                for kind in ("cuboid", "sphere", "capsule", "cylinder", "mesh", "voxel")
                for value in getattr(self, kind)
            ]

        def __iter__(self):
            return iter(self.objects)

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
            original = torch.tensor([[0.0, 0.0, 0.0, -100.0]] * 3)
            self.spheres = original.clone()
            self.sphere_updates = []

            def replace_spheres(link, value):
                self.spheres.copy_(value)
                self.sphere_updates.append(link)

            params = SimpleNamespace(
                link_name_to_idx_map={"payload": 0, "bare_link": 1},
                get_number_of_spheres=lambda link: 3 if link == "payload" else 0,
                update_link_spheres=replace_spheres,
                reset_link_spheres=lambda link: replace_spheres(link, original),
            )
            self.attachment_manager = SimpleNamespace(kinematics_params=params)
            self.obstacle_poses = {}
            self.scene_collision_checker = SimpleNamespace(
                data=SimpleNamespace(
                    meshes=None,
                    get_valid_data=lambda: [
                        type(kind.title() + "Data", (), {"max_n": count})()
                        for kind in ("cuboid", "capsule", "cylinder", "mesh")
                        if (count := (config["collision_cache"] or {}).get(kind, 0)) > 0
                    ],
                ),
                check_obstacle_exists=lambda name: any(
                    obstacle.name == name for obstacle in self.scene
                ),
                update_obstacle_pose=lambda name, pose: self.obstacle_poses.update(
                    {name: pose.clone()}
                ),
            )
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
            class Config(dict):
                pass

            config = Config(kwargs)
            config.ik_solver_config = SimpleNamespace(use_lm_seed=True)
            configs.append(config)
            return config

    for name, members in {
        "curobo": {},
        "curobo.motion_planner": {
            "MotionPlanner": MotionPlanner,
            "MotionPlannerCfg": MotionPlannerCfg,
        },
        "curobo.scene": {"Scene": Scene, "Capsule": SimpleNamespace},
        "curobo.trajectory_timeopt": {"TrajectoryTimeOpt": object},
        "curobo.types": {
            "DeviceCfg": lambda device, dtype: SimpleNamespace(
                device=torch.device("cpu"), dtype=dtype
            ),
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
    assert backend.configs[0].ik_solver_config.use_lm_seed is False


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
    scene = {
        "cuboid": {"wall": {"dims": [1.0, 2.0, 3.0], "pose": [0, 0, 0, 1, 0, 0, 0]}}
    }
    planner = CuroboPlanner(backend.path, scene=scene)
    scene["cuboid"]["wall"]["dims"][0] = 100
    native = backend.instances[0]
    assert native.scene.cuboid[0].dims[0] == 1.0
    planner.update_world({})
    assert list(native.scene) == [] and native.resets == 1
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


def test_fk_accepts_result_bound_roundoff_while_requests_keep_strict_limits(backend):
    with CuroboPlanner(backend.path) as planner:

        def rounded_endpoint(state):
            state.position[0, 0, -1, 2] += 4e-7

        backend.instances[0].failure = rounded_endpoint
        result = planner.plan([0.0, 0.0], [1.0, 0.0])
        assert result.success, result.message
        endpoint = result.trajectory.position[-1]
        assert endpoint[0] > planner.upper_limits[0]
        actual = planner.forward_kinematics(endpoint)
        np.testing.assert_allclose(actual["tool"][:2, 3], endpoint)
        with pytest.raises(ValueError, match="outside"):
            planner.plan([0.0, 0.0], endpoint)


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
    scene.write_text(
        "cuboid:\n  box:\n    dims: [1, 1, 1]\n    pose: [0, 0, 0, 1, 0, 0, 0]\n",
        encoding="utf-8",
    )
    with CuroboPlanner(backend.path, scene=scene) as planner:
        assert backend.instances[0].scene.cuboid[0].dims == [1, 1, 1]
        scene.write_text(
            "cuboid:\n  box:\n    dims: [2, 2, 2]\n    pose: [0, 0, 0, 1, 0, 0, 0]\n",
            encoding="utf-8",
        )
        planner.update_world(scene)
        assert backend.instances[0].scene.cuboid[0].dims == [2, 2, 2]


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


def test_collision_sphere_replacement_pads_restores_and_keeps_old_results(backend):
    with CuroboPlanner(backend.path) as planner:
        previous = planner.plan([0.0, 0.0], [0.1, 0.1]).trajectory
        snapshot = previous.position.copy()
        native = backend.instances[0]
        spheres = np.array([[0.1, 0.2, 0.3, 0.05], [-0.1, 0.0, 0.2, 0.02]])
        planner.update_link_collision_spheres("payload", spheres)
        spheres.fill(99.0)
        np.testing.assert_allclose(
            native.spheres,
            [[0.1, 0.2, 0.3, 0.05], [-0.1, 0.0, 0.2, 0.02], [0, 0, 0, -100]],
        )
        planner.update_link_collision_spheres("payload", [[0.0, 0.0, 0.4, 0.1]])
        np.testing.assert_array_equal(native.spheres[1:, 3], [-100.0, -100.0])
        planner.update_link_collision_spheres("payload", None)
        np.testing.assert_array_equal(native.spheres, [[0.0, 0.0, 0.0, -100.0]] * 3)
        assert native.resets == 3
        np.testing.assert_array_equal(previous.position, snapshot)


@pytest.mark.parametrize(
    "spheres",
    [
        [],
        [[0.0, 0.0, 0.0]],
        [[0, 0, 0, 0]],
        [[0, 0, 0, -1]],
        [[0, 0, np.nan, 0.1]],
        [[0, 0, 0, np.inf]],
        [[1e40, 0, 0, 0.1]],
        [[0, 0, 0, 1e-100]],
        [[0, 0, 0, 1j]],
        [[0, 0, 0, 0.1]] * 4,
    ],
)
def test_invalid_collision_spheres_leave_robot_unchanged(backend, spheres):
    with CuroboPlanner(backend.path) as planner:
        with pytest.raises((ValueError, TypeError)):
            planner.update_link_collision_spheres("payload", spheres)
        native = backend.instances[0]
        assert native.sphere_updates == [] and native.resets == 0
        assert planner.plan([0.0, 0.0], [0.1, 0.1]).success


@pytest.mark.parametrize("link", ["unknown", "bare_link", "", None])
def test_collision_sphere_updates_require_an_existing_allocated_link(backend, link):
    with CuroboPlanner(backend.path) as planner:
        with pytest.raises(ValueError):
            planner.update_link_collision_spheres(link, [[0, 0, 0, 0.1]])
        assert backend.instances[0].sphere_updates == []


def test_partial_collision_sphere_update_blocks_reuse(backend, monkeypatch):
    with CuroboPlanner(backend.path) as planner:
        params = backend.instances[0].attachment_manager.kinematics_params

        def failed_update(*_args):
            raise RuntimeError("partial geometry update")

        monkeypatch.setattr(params, "update_link_spheres", failed_update)
        with pytest.raises(RuntimeError, match="partial geometry"):
            planner.update_link_collision_spheres("payload", [[0, 0, 0, 0.1]])
        planner.update_world({})
        with pytest.raises(RuntimeError, match="failed collision sphere update"):
            planner.plan([0.0, 0.0], [0.1, 0.1])
        with pytest.raises(RuntimeError, match="recreating"):
            planner.warmup()


@pytest.mark.parametrize(
    "scene",
    [
        {"cubiod": {}},
        {"objects": []},
        {"cuboid": {"": {"dims": [1, 1, 1]}}},
        {"cuboid": {"same": {}}, "sphere": {"same": {}}},
        {"cuboid": {"box": {"dims": [-1, 1, 1]}}},
        {"cuboid": {"box": {"dims": [0, 1, 1]}}},
        {"cuboid": {"box": {"dims": [1e-50, 1, 1]}}},
        {"cuboid": {"box": {"dims": [1, 1]}}},
        {"cuboid": {"box": {"dims": [float("inf"), 1, 1]}}},
        {"sphere": {"ball": {"radius": -0.1}}},
        {"sphere": {"ball": {"radius": float("nan")}}},
        {"sphere": {"ball": {"radius": 16777217.0}}},
        {"cuboid": {"box": {"pose": [0, 0, 0, 0, 0, 0, 0]}}},
        {"cuboid": {"box": {"pose": [0, 0, 0, 2, 0, 0, 0]}}},
        {"cuboid": {"box": {"pose": [float("nan"), 0, 0, 1, 0, 0, 0]}}},
        {"capsule": {"pill": {"base": [0, 0], "radius": 0.1}}},
        {"mesh": {"mesh": {"scale": [1, 0, 1]}}},
        {"mesh": {"mesh": {"scale": [1, 1e-50, 1]}}},
    ],
)
def test_malformed_scene_is_rejected_before_replacing_the_world(backend, scene):
    for obstacles in scene.values():
        if isinstance(obstacles, dict):
            for fields in obstacles.values():
                fields.setdefault("pose", [0, 0, 0, 1, 0, 0, 0])
    with CuroboPlanner(backend.path) as planner:
        native = backend.instances[0]
        previous = native.scene
        with pytest.raises((TypeError, ValueError)):
            planner.update_world(scene)
        assert native.scene is previous and native.resets == 0
        assert planner.plan([0.0, 0.0], [0.1, 0.1]).success


@pytest.mark.parametrize(
    "values, dtype",
    [
        (None, None),
        ([0.0] * 8, None),
        ([0] * 8, "int64"),
        ([], "float32"),
        ([float("nan")] * 8, "float32"),
        ([float("inf")] * 8, "float16"),
        ([70000.0] * 8, "float64"),
        ([65536.0] * 8, "bfloat16"),
        ([-65536.0] * 8, "bfloat16"),
    ],
)
def test_invalid_voxel_distances_preserve_the_existing_world(backend, values, dtype):
    features = values
    if dtype is not None:
        features = backend.torch.tensor(values, dtype=getattr(backend.torch, dtype))
    scene = {
        "voxel": {
            "grid": {
                "dims": [1, 1, 1],
                "voxel_size": 0.5,
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "feature_tensor": features,
            }
        }
    }
    with CuroboPlanner(backend.path) as planner:
        native = backend.instances[0]
        previous = native.scene
        with pytest.raises((TypeError, ValueError), match="ESDF"):
            planner.update_world(scene)
        assert native.scene is previous and native.resets == 0
        assert planner.plan([0, 0], [0.1, 0.1]).success


def test_voxel_distances_are_owned_during_creation_and_replacement(backend):
    features = backend.torch.ones(8, dtype=backend.torch.float16)
    scene = {
        "voxel": {
            "grid": {
                "dims": [1, 1, 1],
                "voxel_size": 0.5,
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "feature_tensor": features,
            }
        }
    }
    with CuroboPlanner(backend.path, scene=scene) as planner:
        native = backend.instances[0]
        features.fill_(-1)
        assert backend.torch.all(native.scene.voxel[0].feature_tensor == 1)
        planner.update_world(scene)
        features.fill_(2)
        assert backend.torch.all(native.scene.voxel[0].feature_tensor == -1)


@pytest.mark.parametrize("public_scene", [False, True])
@pytest.mark.parametrize(
    "representation", ["float32", "float64", "nonleaf", "strided", "inference"]
)
def test_voxel_features_use_owned_contiguous_detached_storage(
    backend, public_scene, representation
):
    torch = backend.torch
    features = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) / 10
    if representation == "float64":
        features = features.double()
    elif representation == "nonleaf":
        features = features.requires_grad_(True) * 2
    elif representation == "strided":
        features = features.half().transpose(0, 1)
    elif representation == "inference":
        with torch.inference_mode():
            features = features.clone()
    expected = features.detach().half().contiguous().clone()
    scene = {
        "voxel": {
            "grid": {
                "dims": [1, 1, 1],
                "voxel_size": 0.5,
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "feature_tensor": features,
            }
        }
    }
    if public_scene:
        from curobo.scene import Scene

        scene = Scene.create(scene)
    with CuroboPlanner(backend.path, scene=scene) as planner:
        native = backend.instances[0]
        for replace in (False, True):
            if replace:
                with torch.inference_mode():
                    planner.update_world(scene)
            owned = native.scene.voxel[0].feature_tensor
            assert owned.dtype == torch.float16
            assert owned.device == native.default_joint_state.position.device
            assert owned.is_contiguous() and not owned.requires_grad
            assert not owned.is_inference()
            assert owned.data_ptr() != features.data_ptr()
            torch.testing.assert_close(owned, expected)
        with torch.inference_mode():
            features.fill_(-1)
        torch.testing.assert_close(owned, expected)


def test_public_scene_objects_receive_the_same_geometry_checks(backend):
    from curobo.scene import Scene

    scene = Scene.create(
        {"sphere": {"ball": {"radius": -1.0, "pose": [0, 0, 0, 1, 0, 0, 0]}}}
    )
    with pytest.raises(ValueError, match="radius"):
        CuroboPlanner(backend.path, scene=scene)
    assert backend.instances == []


def test_voxel_conversion_does_not_change_other_scene_tensor_fields(backend):
    torch = backend.torch
    values = torch.tensor(
        [[0.123456, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32
    )
    scene = MappingProxyType(
        {
            "voxel": {
                "grid": {
                    "dims": [1, 1, 1.5],
                    "voxel_size": 0.5,
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "feature_tensor": values,
                }
            },
            "mesh": {
                "mesh": {
                    "vertices": values,
                    "faces": [[0, 1, 2]],
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                }
            },
        }
    )
    with CuroboPlanner(backend.path, scene=scene):
        owned = backend.instances[0].scene
        assert owned.voxel[0].feature_tensor.dtype == torch.float16
        assert owned.mesh[0].vertices.dtype == torch.float32
        assert owned.mesh[0].vertices.data_ptr() != values.data_ptr()
        torch.testing.assert_close(owned.mesh[0].vertices, values, rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 7, 9])
@pytest.mark.parametrize("construction", [False, True])
def test_voxel_feature_count_matches_grid_before_backend_use(
    backend, count, construction
):
    scene = {
        "voxel": {
            "grid": {
                "dims": [1, 1, 1],
                "voxel_size": 0.5,
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "feature_tensor": backend.torch.ones(count),
            }
        }
    }
    if construction:
        with pytest.raises(ValueError, match="feature.*count|grid.*shape"):
            CuroboPlanner(backend.path, scene=scene)
        assert backend.instances == []
    else:
        with CuroboPlanner(backend.path) as planner:
            previous = backend.instances[0].scene
            with pytest.raises(ValueError, match="feature.*count|grid.*shape"):
                planner.update_world(scene)
            assert backend.instances[0].scene is previous
            assert planner.plan([0, 0], [0.1, 0.1]).success


def test_voxel_axes_must_contain_cells_before_the_world_changes(backend):
    scene = {
        "voxel": {
            "grid": {
                "dims": [0.1, 1, 1],
                "voxel_size": 0.5,
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "feature_tensor": backend.torch.ones(1),
            }
        }
    }
    with CuroboPlanner(backend.path) as planner:
        previous = backend.instances[0].scene
        with pytest.raises(ValueError, match="one cell per axis"):
            planner.update_world(scene)
        assert backend.instances[0].scene is previous
        assert planner.plan([0, 0], [0.1, 0.1]).success


def test_scene_spheres_use_exact_capsules_during_creation_and_replacement(backend):
    pose = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    scene = {"sphere": {"ball": {"pose": pose, "radius": 0.1}}}
    with CuroboPlanner(backend.path, scene=scene) as planner:
        native = backend.instances[0]
        assert native.scene.sphere == []
        capsule = native.scene.capsule[0]
        assert capsule.name == "ball" and capsule.radius == 0.1
        assert capsule.pose == pose
        scene["sphere"]["ball"]["radius"] = 0.2
        assert capsule.radius == 0.1
        planner.update_world(scene)
        assert native.scene.sphere == []
        assert native.scene.capsule[0].radius == 0.2
        assert native.resets == 1


def test_scene_objects_cannot_hide_obstacles_from_typed_storage(backend):
    from curobo.scene import Scene

    scene = Scene()
    scene.objects.append(SimpleNamespace(name="hidden", radius=0.1))
    with pytest.raises(ValueError, match="typed obstacle lists"):
        CuroboPlanner(backend.path, scene=scene)
    assert backend.instances == []


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_backend_owns_its_autograd_context_and_restores_the_caller(
    backend, monkeypatch, mode
):
    torch = backend.torch
    from curobo.motion_planner import MotionPlannerCfg

    observed = []

    def record_context(call):
        def recorded(*args, **kwargs):
            observed.append(
                (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            )
            return call(*args, **kwargs)

        return recorded

    monkeypatch.setattr(
        MotionPlannerCfg, "create", record_context(MotionPlannerCfg.create)
    )
    with getattr(torch, mode)():
        before = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        with CuroboPlanner(
            backend.path,
            scene={
                "cuboid": {"box": {"dims": [0.1] * 3, "pose": [0, 0, 0, 1, 0, 0, 0]}}
            },
        ) as planner:
            native = backend.instances[0]
            for name in (
                "plan_cspace",
                "update_world",
                "compute_kinematics",
                "update_link_inertial",
                "warmup",
            ):
                monkeypatch.setattr(native, name, record_context(getattr(native, name)))
            params = native.attachment_manager.kinematics_params
            checker = native.scene_collision_checker
            monkeypatch.setattr(
                checker,
                "update_obstacle_pose",
                record_context(checker.update_obstacle_pose),
            )
            monkeypatch.setattr(
                params,
                "update_link_spheres",
                record_context(params.update_link_spheres),
            )
            planner.warmup(1)
            assert planner.plan([0.0, 0.0], [0.1, 0.1]).success
            targets = planner.forward_kinematics([0.1, 0.1])
            assert planner.plan_pose([0.0, 0.0], targets).success
            planner.update_obstacle_pose("box", np.eye(4))
            planner.update_world({})
            planner.update_link_inertial(
                "payload", mass=0, com=[0] * 3, inertia=[0] * 6
            )
            planner.update_link_collision_spheres("payload", [[0, 0, 0, 0.1]])
            assert (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            ) == before
            with pytest.raises(ValueError):
                planner.plan([99.0, 0.0], [0.1, 0.1])
            assert (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            ) == before
    assert len(observed) >= 8
    assert all(value == (True, False) for value in observed)


def test_world_replacement_discards_obsolete_mesh_geometry(backend, monkeypatch):
    with CuroboPlanner(backend.path) as planner:
        native = backend.instances[0]
        events = []
        cache = {"shape": "old geometry", "removed": "unused geometry"}
        meshes = SimpleNamespace(get_cached_mesh_names=lambda: list(cache))

        def clear(*, clear_warp_cache):
            assert clear_warp_cache
            events.append("clear")
            cache.clear()

        meshes.clear = clear
        native.scene_collision_checker.data.meshes = meshes
        monkeypatch.setattr(
            backend.torch.cuda, "synchronize", lambda _device: events.append("sync")
        )
        original_update = native.update_world

        def update(scene):
            events.append("load")
            assert cache == {}
            original_update(scene)

        monkeypatch.setattr(native, "update_world", update)
        planner.update_world({})
        assert events == ["sync", "clear", "load", "sync"]
        assert native.resets == 1


def test_obstacle_pose_updates_preserve_geometry_and_validate_names(backend):
    scene = {
        "cuboid": {"box": {"dims": [0.1, 0.1, 0.1], "pose": [0, 0, 0, 1, 0, 0, 0]}}
    }
    with CuroboPlanner(backend.path, scene=scene) as planner:
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        planner.update_obstacle_pose("box", transform)
        transform.fill(99.0)
        native = backend.instances[0]
        np.testing.assert_array_equal(
            native.obstacle_poses["box"][:3, 3], [1.0, 2.0, 3.0]
        )
        assert native.scene.cuboid[0].dims == scene["cuboid"]["box"]["dims"]
        assert native.resets == 1
        with pytest.raises(ValueError, match="unknown cuRobo obstacle"):
            planner.update_obstacle_pose("absent", np.eye(4))
        with pytest.raises(ValueError, match="rotation"):
            planner.update_obstacle_pose(
                "box", np.zeros((4, 4)) + np.diag([0, 0, 0, 1])
            )
        assert native.resets == 1


def test_partial_obstacle_pose_update_blocks_planning_until_world_recovery(
    backend, monkeypatch
):
    with CuroboPlanner(
        backend.path,
        scene={"sphere": {"ball": {"radius": 0.1, "pose": [0, 0, 0, 1, 0, 0, 0]}}},
    ) as planner:
        native = backend.instances[0]

        def fail(*_args):
            raise RuntimeError("partial pose update")

        monkeypatch.setattr(
            native.scene_collision_checker, "update_obstacle_pose", fail
        )
        with pytest.raises(RuntimeError, match="partial pose"):
            planner.update_obstacle_pose("ball", np.eye(4))
        assert native.resets == 1
        with pytest.raises(RuntimeError, match="failed world update"):
            planner.plan([0.0, 0.0], [0.1, 0.1])
        planner.update_world({})
        assert planner.plan([0.0, 0.0], [0.1, 0.1]).success


def test_incompatible_live_graph_layouts_fail_before_capture_and_release_on_close(
    backend,
):
    first = CuroboPlanner(backend.path, collision_cache={"cuboid": 4})
    try:
        with pytest.raises(RuntimeError, match="share a collision cache layout"):
            CuroboPlanner(backend.path, collision_cache={"capsule": 2})
        assert backend.instances[-1].destroyed
        with CuroboPlanner(backend.path, collision_cache={"cuboid": 4}) as matching:
            assert matching.plan([0, 0], [0.1, 0.1]).success
        assert first.plan([0, 0], [0.1, 0.1]).success
    finally:
        first.close()
    with CuroboPlanner(backend.path, collision_cache={"capsule": 2}) as changed:
        assert changed.plan([0, 0], [0.1, 0.1]).success


def test_distinct_layouts_work_without_graphs_but_cannot_invalidate_a_live_capture(
    backend,
):
    with CuroboPlanner(
        backend.path, collision_cache={"cuboid": 4}, use_cuda_graph=False
    ):
        with CuroboPlanner(
            backend.path, collision_cache={"capsule": 2}, use_cuda_graph=False
        ):
            pass
        with pytest.raises(RuntimeError, match="share a collision cache layout"):
            CuroboPlanner(backend.path, collision_cache={"capsule": 2})


def test_layout_registry_does_not_keep_abandoned_planners_alive(backend):
    import gc
    from weakref import ref

    planner = CuroboPlanner(backend.path, collision_cache={"cuboid": 4})
    reference = ref(planner)
    del planner
    gc.collect()
    assert reference() is None
    with CuroboPlanner(backend.path, collision_cache={"capsule": 2}):
        pass


def test_different_planner_threads_do_not_allocate_during_a_backend_capture(
    backend, monkeypatch
):
    entered, release, requested, updated = Event(), Event(), Event(), Event()
    with CuroboPlanner(backend.path) as first, CuroboPlanner(backend.path) as second:
        first_native, second_native = backend.instances
        solve = first_native.plan_cspace
        update = second_native.update_link_inertial

        def blocked_solve(*args, **kwargs):
            entered.set()
            assert release.wait(5.0)
            return solve(*args, **kwargs)

        def record_update(*args):
            updated.set()
            return update(*args)

        def submit_update():
            requested.set()
            second.update_link_inertial("payload", mass=0, com=[0] * 3, inertia=[0] * 6)

        monkeypatch.setattr(first_native, "plan_cspace", blocked_solve)
        monkeypatch.setattr(second_native, "update_link_inertial", record_update)
        with ThreadPoolExecutor(2) as workers:
            query = workers.submit(first.plan, [0, 0], [0.1, 0.1])
            try:
                assert entered.wait(5.0)
                edit = workers.submit(submit_update)
                assert requested.wait(5.0)
                assert not updated.wait(0.05)
            finally:
                release.set()
            assert query.result(timeout=5.0).success
            edit.result(timeout=5.0)
        assert updated.is_set()


@pytest.mark.parametrize(
    "kind", ["cuboid", "sphere", "capsule", "cylinder", "mesh", "voxel"]
)
def test_missing_obstacle_pose_is_rejected_without_invalidating_the_current_world(
    backend, kind
):
    with CuroboPlanner(backend.path) as planner:
        native = backend.instances[0]
        previous = native.scene
        fields = {}
        if kind == "voxel":
            fields = {
                "dims": [1, 1, 1],
                "voxel_size": 0.5,
                "feature_tensor": backend.torch.ones(8),
            }
        with pytest.raises(ValueError, match="explicit.*pose"):
            planner.update_world({kind: {"unplaced": fields}})
        assert native.scene is previous and native.resets == 0
        assert planner.plan([0, 0], [0.1, 0.1]).success
