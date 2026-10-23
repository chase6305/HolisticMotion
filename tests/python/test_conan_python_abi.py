"""Opt-in Conan cache identity checks; use the already resolved local graph."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("HOLISTICMOTION_TEST_CONAN_PYTHON_ABI") != "1",
    reason="set HOLISTICMOTION_TEST_CONAN_PYTHON_ABI=1 after resolving Conan dependencies",
)


def graph(version, *, with_python=True, executable=None, abi="auto"):
    conan = shutil.which("conan")
    assert conan is not None, "Conan is required for package identity validation"
    repository = Path(__file__).resolve().parents[2]
    command = [
        conan,
        "graph",
        "info",
        str(repository),
        "--no-remote",
        "--format=json",
        "-o",
        f"&:with_python={with_python}",
        "-o",
        "&:with_collision=False",
        "-o",
        f"&:python_version={version}",
        "-o",
        f"&:python_abi={abi}",
    ]
    if executable is not None:
        command += [
            "-c",
            "tools.cmake.cmaketoolchain:extra_variables="
            + json.dumps({"Python3_EXECUTABLE": str(executable)}),
        ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["graph"]["nodes"]["0"]


def test_native_python_minors_have_different_package_identities():
    first, second = graph("3.10"), graph("3.12")
    assert first["package_id"] != second["package_id"]
    assert first["info"]["options"]["python_version"] == "3.10"
    assert second["info"]["options"]["python_version"] == "3.12"


def test_cpp_only_package_identity_does_not_depend_on_python():
    first = graph("3.10", with_python=False, abi="cpython")
    second = graph("3.12", with_python=False, abi="cpython-t")
    assert first["package_id"] == second["package_id"]
    assert "python_version" not in first["info"]["options"]
    assert "python_abi" not in first["info"]["options"]


def test_auto_uses_the_selected_interpreter_minor():
    automatic = graph("auto", executable=sys.executable)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    explicit = graph(version, executable=sys.executable)
    assert automatic["info"]["options"]["python_version"] == version
    assert automatic["package_id"] == explicit["package_id"]


@pytest.mark.parametrize(
    "abi", ["cpython-t", "cpython-d", "cpython-dt", "pypy-pypy313-pp73"]
)
def test_interpreter_variants_of_the_same_minor_have_distinct_binary_ids(abi):
    standard = graph("3.13", abi="cpython")
    variant = graph("3.13", abi=abi)
    assert standard["package_id"] != variant["package_id"]
    assert variant["info"]["options"]["python_abi"] == abi


def test_auto_records_the_selected_interpreter_abi():
    probe = (
        Path(__file__).resolve().parents[2] / "python/holistic_motion/_python_abi.py"
    )
    identity = json.loads(
        subprocess.check_output([sys.executable, "-I", "-S", str(probe)], text=True)
    )
    automatic = graph("auto", executable=sys.executable)
    explicit = graph(identity["version"], abi=identity["abi"])
    assert automatic["info"]["options"]["python_abi"] == identity["abi"]
    assert automatic["package_id"] == explicit["package_id"]


@pytest.mark.parametrize("abi", ["invalid tag", "../cpython", ""])
def test_invalid_python_abi_has_a_configuration_diagnostic(abi):
    assert "python_abi must be" in graph("3.13", abi=abi)["info_invalid"]


def test_cmake_rejects_an_interpreter_with_the_wrong_abi(tmp_path):
    toolchain = os.environ.get("HOLISTICMOTION_TEST_TOOLCHAIN")
    if toolchain is None:
        pytest.skip("set HOLISTICMOTION_TEST_TOOLCHAIN for the CMake ABI check")
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(repository),
            "-B",
            str(tmp_path / "build"),
            f"-DCMAKE_TOOLCHAIN_FILE={Path(toolchain).resolve()}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DHOLISTICMOTION_BUILD_TESTS=OFF",
            "-DHOLISTICMOTION_ENABLE_COLLISION=OFF",
            "-DHOLISTICMOTION_ENABLE_CUDA=OFF",
            "-DHOLISTICMOTION_BUILD_PYTHON=ON",
            "-DHOLISTICMOTION_PYTHON_ABI=mismatched-abi",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Python ABI" in result.stdout + result.stderr
    assert (
        "does not match the Conan package identity mismatched-abi"
        in " ".join((result.stdout + result.stderr).split())
    )


@pytest.mark.parametrize("version", ["3.8", "3.10.1", "invalid"])
def test_invalid_python_minor_has_a_configuration_diagnostic(version):
    assert "python_version must be" in graph(version)["info_invalid"]


def test_cmake_rejects_an_interpreter_with_the_wrong_minor(tmp_path):
    toolchain = os.environ.get("HOLISTICMOTION_TEST_TOOLCHAIN")
    if toolchain is None:
        pytest.skip("set HOLISTICMOTION_TEST_TOOLCHAIN for the CMake ABI check")
    repository = Path(__file__).resolve().parents[2]
    version = f"{sys.version_info.major}.{sys.version_info.minor + 1}"
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(repository),
            "-B",
            str(tmp_path / "build"),
            f"-DCMAKE_TOOLCHAIN_FILE={Path(toolchain).resolve()}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DHOLISTICMOTION_BUILD_TESTS=OFF",
            "-DHOLISTICMOTION_ENABLE_COLLISION=OFF",
            "-DHOLISTICMOTION_ENABLE_CUDA=OFF",
            "-DHOLISTICMOTION_BUILD_PYTHON=ON",
            f"-DHOLISTICMOTION_PYTHON_VERSION={version}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Python3" in result.stdout + result.stderr
    assert f'exact version "{version}"' in result.stdout + result.stderr
