"""Black-box checks of an explicitly supplied installed C++ SDK.

Set HOLISTICMOTION_TEST_SDK_PREFIX and HOLISTICMOTION_TEST_TOOLCHAIN to run these
consumer checks. Source-only Python toolkits do not require a native SDK.
"""

import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def run(command, *, success=True):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == success, output
    return output


@pytest.fixture(scope="module")
def sdk(tmp_path_factory):
    prefix = os.environ.get("HOLISTICMOTION_TEST_SDK_PREFIX")
    if prefix is None:
        pytest.skip("set HOLISTICMOTION_TEST_SDK_PREFIX to test an installed SDK")
    cmake = shutil.which("cmake")
    assert cmake is not None, "CMake is required for installed-SDK validation"
    configs = list(Path(prefix).resolve().rglob("HolisticMotionConfig.cmake"))
    assert len(configs) == 1, f"expected one SDK configuration under {prefix}"
    arguments = [
        cmake,
        "--no-warn-unused-cli",
        f"-DHolisticMotion_DIR={configs[0].parent}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    toolchain = os.environ.get("HOLISTICMOTION_TEST_TOOLCHAIN")
    if toolchain is not None:
        assert Path(toolchain).is_file(), toolchain
        arguments.append(f"-DCMAKE_TOOLCHAIN_FILE={Path(toolchain).resolve()}")
    source = tmp_path_factory.mktemp("sdk_capabilities")
    (source / "CMakeLists.txt").write_text(r"""cmake_minimum_required(VERSION 3.22)
project(SDKCapabilities LANGUAGES CXX)
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core)
if(HolisticMotion_collision_FOUND)
  set(collision true)
else()
  set(collision false)
endif()
if(TARGET CUDA::cudart)
  set(cuda true)
else()
  set(cuda false)
endif()
file(WRITE "${CMAKE_BINARY_DIR}/capabilities.json"
  "{\"collision\":${collision},\"cuda\":${cuda}}")
""")
    build = source / "build"
    run(
        arguments
        + [
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_DISABLE_FIND_PACKAGE_pinocchio=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_coal=ON",
        ]
    )
    capability = json.loads((build / "capabilities.json").read_text())
    return arguments, capability


@pytest.mark.parametrize("component", ["core", "collision"])
def test_installed_component_builds_with_only_its_dependencies(
    sdk, tmp_path, component
):
    arguments, capability = sdk
    repository = Path(__file__).resolve().parents[2]
    build = tmp_path / "consumer"
    flags = [f"-DHOLISTICMOTION_CONSUMER_COMPONENTS={component}"]
    if component == "core":
        flags += [
            "-DCMAKE_DISABLE_FIND_PACKAGE_pinocchio=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_coal=ON",
        ]
        targets = ["coordinate_consumer", "trajectory_consumer", "fep_consumer"]
    else:
        flags += [
            "-DHOLISTICMOTION_CONSUMER_TEST_COLLISION=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_fmt=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_spdlog=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON",
        ]
        targets = ["collision_consumer"]
    expected = component == "core" or capability["collision"]
    output = run(
        arguments + ["-S", str(repository / "test_package"), "-B", str(build)] + flags,
        success=expected,
    )
    if not expected:
        assert "built without the collision component" in output
        return
    run([arguments[0], "--build", str(build), "--parallel", "2", "--target", *targets])
    for target in targets:
        executable = build / (target + (".exe" if os.name == "nt" else ""))
        run([str(executable)])


def test_installed_core_preserves_inputs_from_reused_output_containers(sdk, tmp_path):
    arguments, _ = sdk
    sources = Path(__file__).resolve().parents[2] / "tests" / "cpp" / "kinematics"
    (tmp_path / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.22)
project(InstalledInputOwnership LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core)
add_executable(ik_alias "${alias_sources}/ik_seed_alias_smoke.cpp")
add_executable(null_space_alias "${alias_sources}/srs_null_space_smoke.cpp")
target_link_libraries(ik_alias PRIVATE HolisticMotion::holistic_motion)
target_link_libraries(null_space_alias PRIVATE HolisticMotion::holistic_motion)
""")
    build = tmp_path / "build"
    run(
        arguments
        + [
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-Dalias_sources={sources}",
            "-DCMAKE_DISABLE_FIND_PACKAGE_pinocchio=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_coal=ON",
        ]
    )
    run([arguments[0], "--build", str(build), "--parallel", "2"])
    for name in ("ik_alias", "null_space_alias"):
        run([str(build / (name + (".exe" if os.name == "nt" else "")))])


def test_unknown_required_component_has_a_clear_diagnostic(sdk, tmp_path):
    arguments, _ = sdk
    (tmp_path / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.22)
project(UnknownComponent LANGUAGES CXX)
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS missing_component)
""")
    output = run(
        arguments + ["-S", str(tmp_path), "-B", str(tmp_path / "build")], success=False
    )
    assert "Unknown HolisticMotion component: missing_component" in output


def test_external_custom_solver_keeps_conservative_python_calling_contract(
    sdk, tmp_path
):
    arguments, _ = sdk
    prefix = Path(os.environ["HOLISTICMOTION_TEST_SDK_PREFIX"]).resolve()
    if not any(
        (prefix / "holistic_motion" / ("_holistic_motion" + suffix)).is_file()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        pytest.skip("requires an SDK with a matching native Python extension")
    (tmp_path / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.22)
project(CustomPythonSolver LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 QUIET CONFIG)
if(NOT pybind11_FOUND)
  file(WRITE "${CMAKE_BINARY_DIR}/available" "0")
  return()
endif()
file(WRITE "${CMAKE_BINARY_DIR}/available" "1")
pybind11_add_module(_custom_solver_probe probe.cpp)
target_link_libraries(_custom_solver_probe PRIVATE HolisticMotion::holistic_motion)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/probe_path.txt"
  CONTENT "$<TARGET_FILE:_custom_solver_probe>")
""")
    (tmp_path / "probe.cpp").write_text(r"""#include <pybind11/pybind11.h>
#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"
using namespace holistic_motion::robotics;
static std::unique_ptr<planning::SamplingPlanner> borrowed_planner;
class CustomSolver : public KinematicsBase {
public:
    CustomSolver() { dof_ = 1; }
    bool GetFK(const Eigen::VectorXd &q, SE3d &pose) const override {
        pose = SE3d();
        return q.size() == 1;
    }
    bool GetIK(const SE3d &, IkRtn &result, Eigen::VectorXd &seed,
               std::vector<double> &distances) const override {
        result.ik_joints = {seed};
        distances = {0.0};
        return PyGILState_Check() != 0;
    }
    bool GetNearstIK(const SE3d &, IkRtn &result, Eigen::VectorXd &seed,
                     double &distance) const override {
        result.ik_joints = {seed};
        distance = 0.0;
        return PyGILState_Check() != 0;
    }
};
class CustomNumericalSolver : public NumericalKinematics {
public:
    CustomNumericalSolver() : NumericalKinematics(Nodes()) {}
    bool GetAllFK(const Eigen::VectorXd &joints,
                   std::vector<SE3d> &poses) const override {
        return PyGILState_Check() != 0 && KinematicsBase::GetAllFK(joints, poses);
    }
private:
    static std::vector<JointNode> Nodes() {
        return {JointNode(SE3d(), Eigen::Vector3d::UnitZ(), JointType::REVOLUTE,
                          -2.0, 2.0),
                JointNode(SE3d(), Eigen::Vector3d::Zero(), JointType::FIXED,
                          0.0, 0.0)};
    }
};
PYBIND11_MODULE(_custom_solver_probe, module) {
    module.def("make_solver", []() -> std::shared_ptr<KinematicsBase> {
        return std::make_shared<CustomSolver>();
    });
    module.def("make_numerical_solver", []() -> std::shared_ptr<KinematicsBase> {
        return std::make_shared<CustomNumericalSolver>();
    });
    module.def("make_planner", [](bool require_gil) {
        return planning::SamplingPlanner(Eigen::VectorXd::Constant(1, -1.0),
                                          Eigen::VectorXd::Constant(1, 1.0),
            [require_gil](const Eigen::VectorXd &) {
                return static_cast<bool>(PyGILState_Check()) == require_gil;
            });
    });
    module.def("make_optimizer", [](bool require_gil) {
        return planning::PathOptimizer(Eigen::VectorXd::Constant(1, -1.0),
                                        Eigen::VectorXd::Constant(1, 1.0),
            [require_gil](const Eigen::VectorXd &) {
                return static_cast<bool>(PyGILState_Check()) == require_gil;
            });
    });
    module.def("borrow_planner", []() {
        borrowed_planner = std::make_unique<planning::SamplingPlanner>(
            Eigen::VectorXd::Constant(1, -1.0), Eigen::VectorXd::Constant(1, 1.0));
        return borrowed_planner.get();
    }, pybind11::return_value_policy::reference);
    module.def("clear_borrowed_callback", []() {
        borrowed_planner->SetStateValidator({});
    });
}
""")
    build = tmp_path / "build"
    run(
        arguments
        + [
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
    )
    if (build / "available").read_text() == "0":
        pytest.skip("pybind11 development headers are unavailable")
    run([arguments[0], "--build", str(build), "--parallel", "2"])
    plugin = (build / "probe_path.txt").read_text()
    run(
        [
            sys.executable,
            "-c",
            """
import importlib.util
import os
import sys
os.environ['HOLISTICMOTION_PURE_PYTHON'] = '0'
sys.path.insert(0, sys.argv[1])
import holistic_motion as hm
import numpy as np
import gc
import weakref
spec = importlib.util.spec_from_file_location('_custom_solver_probe', sys.argv[2])
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)
for solver, family, algorithm in (
    (plugin.make_solver(), hm.KinematicsFamily.CUSTOM, hm.IKAlgorithm.CUSTOM),
    (plugin.make_numerical_solver(), hm.KinematicsFamily.NUMERICAL,
     hm.IKAlgorithm.NUMERICAL),
):
    assert isinstance(solver, hm.KinematicsBase)
    assert solver.capabilities.family == family
    for selection in (hm.IKSelection.NEAREST, hm.IKSelection.ALL):
        report = solver.solve_ik(np.eye(4), [0.0], selection)
        assert report.success, 'the legacy custom solver lost its interpreter contract'
        assert report.algorithm == algorithm
    assert not solver.capabilities.supports_concurrent_queries
for release in (False, True):
    planner = plugin.make_planner(not release)
    assert planner.plan([-.8], [.8], release_gil=release).success
    optimizer = plugin.make_optimizer(not release)
    assert optimizer.optimize([[-.8], [.1], [.8]], release_gil=release).success
# Existing native callbacks may assume the interpreter lock. The default
# calling mode remains compatible unless the caller opts into release.
assert plugin.make_planner(True).plan([-.8], [.8]).success
assert plugin.make_optimizer(True).optimize([[-.8], [.1], [.8]]).success
def borrowed_cycle():
    planner = plugin.borrow_planner()
    def valid(state):
        return planner.joint_space.dof == 1
    planner.set_state_validator(valid)
    return weakref.ref(planner), weakref.ref(valid)
planner_ref, callback_ref = borrowed_cycle()
gc.collect()
# A non-owning Python view must not clear the external C++ owner's callback.
assert planner_ref() is not None and callback_ref() is not None
plugin.clear_borrowed_callback()
gc.collect()
assert planner_ref() is None and callback_ref() is None
""",
            str(prefix),
            plugin,
        ]
    )


def test_collision_sanitizers_require_a_matching_dependency_cohort(sdk, tmp_path):
    arguments, _ = sdk
    repository = Path(__file__).resolve().parents[2]
    output = run(
        arguments
        + [
            "-S",
            str(repository),
            "-B",
            str(tmp_path / "build"),
            "-DHOLISTICMOTION_BUILD_PYTHON=OFF",
            "-DHOLISTICMOTION_BUILD_TESTS=OFF",
            "-DHOLISTICMOTION_ENABLE_CUDA=OFF",
            "-DHOLISTICMOTION_ENABLE_COLLISION=ON",
            "-DHOLISTICMOTION_ENABLE_SANITIZERS=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_pinocchio=ON",
        ],
        success=False,
    )
    assert "HOLISTICMOTION_ENABLE_COLLISION=OFF" in output
    assert "Pinocchio/Eigen" in output


def test_unavailable_optional_collision_preserves_core(sdk, tmp_path):
    arguments, _ = sdk
    (tmp_path / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.22)
project(OptionalCollision LANGUAGES CXX)
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core OPTIONAL_COMPONENTS collision)
if(NOT TARGET HolisticMotion::holistic_motion OR HolisticMotion_collision_FOUND)
  message(FATAL_ERROR "optional collision discovery changed core availability")
endif()
if(TARGET HolisticMotion::collision)
  message(FATAL_ERROR "unavailable collision target was imported")
endif()
""")
    run(
        arguments
        + [
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            "-DCMAKE_DISABLE_FIND_PACKAGE_pinocchio=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_coal=ON",
        ]
    )


@pytest.mark.parametrize("dependency", ["fmt", "spdlog", "CUDAToolkit"])
def test_unavailable_optional_core_preserves_collision(sdk, tmp_path, dependency):
    arguments, capability = sdk
    if not capability["collision"]:
        pytest.skip("requires an SDK with the collision component")
    if dependency == "CUDAToolkit" and not capability["cuda"]:
        pytest.skip("requires a CUDA-enabled core")
    (tmp_path / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.22)
project(OptionalCore LANGUAGES CXX)
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS collision OPTIONAL_COMPONENTS core)
if(NOT TARGET HolisticMotion::collision OR HolisticMotion_core_FOUND)
  message(FATAL_ERROR "optional core discovery changed collision availability")
endif()
if(TARGET HolisticMotion::holistic_motion)
  message(FATAL_ERROR "unavailable core target was imported")
endif()
set(CMAKE_DISABLE_FIND_PACKAGE_${missing_dependency} FALSE)
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core)
if(NOT HolisticMotion_core_FOUND OR NOT TARGET HolisticMotion::holistic_motion)
  message(FATAL_ERROR "subsequent core discovery retained an unavailable state")
endif()
""")
    run(
        arguments
        + [
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            f"-Dmissing_dependency={dependency}",
            f"-DCMAKE_DISABLE_FIND_PACKAGE_{dependency}=ON",
        ]
    )
