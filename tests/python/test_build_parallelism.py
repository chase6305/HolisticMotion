"""Native build defaults respect visible budgets and explicit user choices."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "scripts" / "build_parallelism.py").is_file():
    pytest.skip("requires repository development scripts", allow_module_level=True)
spec = importlib.util.spec_from_file_location(
    "hm_build_parallelism", ROOT / "scripts" / "build_parallelism.py"
)
parallelism = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallelism)
GIB = 1024**3


@pytest.mark.parametrize(
    "cpus,memory,expected",
    [
        (32, 10 * GIB, 5),
        (4, 64 * GIB, 4),
        (8, 0, 1),
        (8, GIB, 1),
        (8, None, 8),
        (0, None, 1),
    ],
)
def test_resource_budget_never_returns_zero_or_exceeds_known_limits(
    cpus, memory, expected
):
    assert parallelism.choose_jobs(cpus, memory) == expected


@pytest.fixture
def resource_files(tmp_path, monkeypatch):
    monkeypatch.setattr(parallelism.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(
        parallelism.os, "sched_getaffinity", lambda _: set(range(16)), raising=False
    )
    proc, groups = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    groups.mkdir()
    (proc / "meminfo").write_text(f"MemAvailable: {64 * GIB // 1024} kB\n")
    (proc / "self" / "cgroup").write_text("0::/user/build\n")

    def write(group, name, value):
        path = groups / group / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value))

    return proc, groups, write


def test_parent_v2_cpu_quota_and_child_memory_usage_both_apply(resource_files):
    proc, groups, write = resource_files
    write("user", "cpu.max", "250000 100000")
    write("user/build", "cpu.max", "450000 100000")
    write("user", "memory.max", 16 * GIB)
    write("user", "memory.current", 2 * GIB)
    write("user/build", "memory.max", 8 * GIB)
    write("user/build", "memory.current", 4 * GIB)
    assert parallelism.recommended_jobs(proc, groups) == 2


def test_tighter_parent_memory_headroom_cannot_be_overridden_by_child(resource_files):
    proc, groups, write = resource_files
    write("user", "memory.max", 8 * GIB)
    write("user", "memory.current", 7 * GIB)
    write("user/build", "memory.max", 16 * GIB)
    write("user/build", "memory.current", GIB)
    assert parallelism.recommended_jobs(proc, groups) == 1


def test_namespace_root_limits_apply_when_membership_is_root(resource_files):
    proc, groups, write = resource_files
    (proc / "self" / "cgroup").write_text("0::/\n")
    write("", "cpu.max", "150000 100000")
    write("", "memory.max", 8 * GIB)
    write("", "memory.current", GIB)
    assert parallelism.recommended_jobs(proc, groups) == 2


def test_legacy_controller_mounts_and_missing_usage_fall_back_safely(resource_files):
    proc, groups, write = resource_files
    (proc / "self" / "cgroup").write_text("2:cpu,cpuacct:/build\n3:memory:/build\n")
    write("cpu,cpuacct/build", "cpu.cfs_quota_us", 250000)
    write("cpu,cpuacct/build", "cpu.cfs_period_us", 100000)
    write("memory/build", "memory.limit_in_bytes", 4 * GIB)
    assert parallelism.recommended_jobs(proc, groups) == 2
    write("memory/build", "memory.usage_in_bytes", 3 * GIB)
    assert parallelism.recommended_jobs(proc, groups) == 1


def test_host_available_memory_is_also_a_bound(resource_files):
    proc, groups, write = resource_files
    (proc / "meminfo").write_text(f"MemAvailable: {3 * GIB // 1024} kB\n")
    write("user/build", "memory.max", 32 * GIB)
    write("user/build", "memory.current", GIB)
    assert parallelism.recommended_jobs(proc, groups) == 1


def test_missing_or_malformed_files_keep_cpu_affinity_fallback(resource_files):
    proc, groups, write = resource_files
    (proc / "meminfo").write_text("MemAvailable: unknown kB\n")
    write("user/build", "cpu.max", "max 100000")
    write("user/build", "memory.max", "max")
    assert parallelism.recommended_jobs(proc, groups) == 16
    write("user/build", "cpu.max", "invalid")
    write("user/build", "memory.max", "-1")
    assert parallelism.recommended_jobs(proc, groups) == 16


def test_affinity_remains_usable_when_cpu_count_is_unknown(resource_files, monkeypatch):
    proc, groups, _ = resource_files
    monkeypatch.setattr(parallelism.os, "cpu_count", lambda: None)
    assert parallelism.recommended_jobs(proc, groups) == 16


def _mount(root, point, filesystem="cgroup2", controllers=""):
    def escape(value):
        return (
            str(value)
            .replace("\\", "\\134")
            .replace(" ", "\\040")
            .replace("\t", "\\011")
        )

    return (
        f"42 25 0:42 {escape(root)} {escape(point)} rw,nosuid shared:7 "
        f"- {filesystem} cgroup rw{',' + controllers if controllers else ''}\n"
    )


@pytest.mark.parametrize("root", ["/", "/user"])
@pytest.mark.parametrize(
    "mount_name", ["mounted group", "mounted\tgroup\\name", "资源组"]
)
def test_v2_limits_use_actual_mount_and_subtree_root(resource_files, root, mount_name):
    proc, groups, write = resource_files
    (proc / "self" / "mountinfo").write_text(_mount(root, groups / mount_name))
    relative = "user/build" if root == "/" else "build"
    write(f"{mount_name}/{relative}", "cpu.max", "350000 100000")
    write(mount_name, "memory.max", 4 * GIB)
    write(mount_name, "memory.current", GIB)
    assert parallelism.recommended_jobs(proc, groups) == 1


def test_v1_controllers_use_separate_actual_mounts(resource_files):
    proc, groups, write = resource_files
    (proc / "self" / "cgroup").write_text(
        "2:cpu,cpuacct:/user/build\n3:memory:/user/build\n"
    )
    (proc / "self" / "mountinfo").write_text(
        _mount("/user", groups / "custom cpu", "cgroup", "cpuacct,cpu")
        + _mount("/", groups / "custom memory", "cgroup", "memory")
    )
    write("custom cpu", "cpu.cfs_quota_us", 150000)
    write("custom cpu", "cpu.cfs_period_us", 100000)
    write("custom memory/user/build", "memory.limit_in_bytes", 16 * GIB)
    write("custom memory/user/build", "memory.usage_in_bytes", 4 * GIB)
    assert parallelism.recommended_jobs(proc, groups) == 2


def test_unrelated_bind_mount_does_not_limit_the_process(resource_files):
    proc, groups, write = resource_files
    (proc / "self" / "mountinfo").write_text(
        _mount("/other", groups / "unrelated") + _mount("/user", groups / "actual")
    )
    write("unrelated", "memory.max", 0)
    write("actual/build", "cpu.max", "350000 100000")
    assert parallelism.recommended_jobs(proc, groups) == 4


def test_namespace_root_can_use_a_mounted_subtree(resource_files):
    proc, groups, write = resource_files
    (proc / "self" / "cgroup").write_text("0::/\n")
    (proc / "self" / "mountinfo").write_text(
        _mount("/user/build", groups / "namespace")
    )
    write("namespace", "cpu.max", "150000 100000")
    assert parallelism.recommended_jobs(proc, groups) == 2


def test_malformed_mount_information_keeps_conventional_fallback(resource_files):
    proc, groups, write = resource_files
    (proc / "self" / "mountinfo").write_text(
        "invalid\n42 25 0:42 / /missing rw - cgroup2\n"
        + _mount("/", groups / "not-cgroup", "tmpfs")
    )
    write("user/build", "cpu.max", "250000 100000")
    assert parallelism.recommended_jobs(proc, groups) == 3


@pytest.mark.parametrize("name", ["资源\u3000组", "line\u2028group", "carriage\rgroup"])
def test_unicode_whitespace_in_mount_paths_is_not_a_field_separator(
    resource_files, name
):
    proc, groups, write = resource_files
    (proc / "self" / "mountinfo").write_text(_mount("/user", groups / name))
    write(f"{name}/build", "memory.max", 4 * GIB)
    write(f"{name}/build", "memory.current", GIB)
    assert parallelism.recommended_jobs(proc, groups) == 1


@pytest.mark.parametrize(
    "membership", ["user/build\u3000", "user/row\u2028name", "user/carriage\rname"]
)
def test_unicode_whitespace_in_membership_names_is_preserved(
    resource_files, membership
):
    proc, groups, write = resource_files
    (proc / "self" / "cgroup").write_text(f"0::/{membership}\n")
    (proc / "self" / "mountinfo").write_text(_mount("/", groups / "actual"))
    write(f"actual/{membership}", "cpu.max", "150000 100000")
    assert parallelism.recommended_jobs(proc, groups) == 2


@pytest.fixture
def fake_build(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("requires Bash")
    repository = tmp_path / "repository with spaces"
    (repository / "scripts").mkdir(parents=True)
    (repository / "python" / "holistic_motion").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts" / "build.sh", repository / "scripts" / "build.sh")
    shutil.copyfile(
        ROOT / "python" / "holistic_motion" / "_python_abi.py",
        repository / "python" / "holistic_motion" / "_python_abi.py",
    )
    (repository / "scripts" / "build_parallelism.py").write_text(
        "import os,pathlib\npathlib.Path(os.environ['HM_JOBS_MARKER']).touch()\nprint(3)\n"
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    conan = commands / "conan"
    conan.write_text(
        "#!/usr/bin/env python3\nimport pathlib,sys\n"
        "if sys.argv[1]=='install':\n"
        " root=next(a.split('=',1)[1] for a in sys.argv if a.startswith('--output-folder='))\n"
        " p=pathlib.Path(root)/'build/Release/generators/conan_toolchain.cmake'\n"
        " p.parent.mkdir(parents=True,exist_ok=True)\n p.write_text('# fixture\\n')\n"
    )
    cmake = commands / "cmake"
    cmake.write_text(
        "#!/usr/bin/env python3\nimport json,os,pathlib,sys\n"
        "if '--build' in sys.argv:\n"
        " pathlib.Path(os.environ['HM_BUILD_ARGUMENTS']).write_text(json.dumps(sys.argv[1:]))\n"
        " sys.exit(42)\n"
    )
    conan.chmod(0o755)
    cmake.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("JOBS", None)
    environment.update(
        PATH=str(commands) + os.pathsep + environment["PATH"],
        PYTHON_EXECUTABLE=sys.executable,
        BUILD_DIR=str(tmp_path / "build with spaces"),
        BUILD_TYPE="Release",
        CONAN_PROFILE_HOST="fixture",
        CONAN_PROFILE_BUILD="fixture",
        HM_JOBS_MARKER=str(tmp_path / "automatic-called"),
        HM_BUILD_ARGUMENTS=str(tmp_path / "build-arguments.json"),
    )
    return repository, environment


@pytest.mark.parametrize("explicit", [None, "7"])
def test_build_script_keeps_explicit_jobs_and_only_probes_for_auto(
    fake_build, explicit
):
    repository, environment = fake_build
    if explicit is not None:
        environment["JOBS"] = explicit
    result = subprocess.run(
        ["bash", str(repository / "scripts" / "build.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 42, result.stdout + result.stderr
    arguments = json.loads(Path(environment["HM_BUILD_ARGUMENTS"]).read_text())
    assert arguments[arguments.index("--parallel") + 1] == (explicit or "3")
    assert Path(environment["HM_JOBS_MARKER"]).exists() == (explicit is None)


@pytest.mark.parametrize("jobs", ["0", "-1", "many"])
def test_invalid_explicit_jobs_fail_before_dependency_work(fake_build, jobs):
    repository, environment = fake_build
    environment["JOBS"] = jobs
    result = subprocess.run(
        ["bash", str(repository / "scripts" / "build.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert "JOBS must be a positive integer" in result.stderr
    assert not Path(environment["BUILD_DIR"]).exists()
