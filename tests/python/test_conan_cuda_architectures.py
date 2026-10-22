"""Opt-in CUDA binary identity checks; no compiler or device is required."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("HOLISTICMOTION_TEST_CONAN_CUDA_ARCHITECTURES") != "1",
    reason="set HOLISTICMOTION_TEST_CONAN_CUDA_ARCHITECTURES=1 after resolving Conan",
)


def graph(
    architecture="auto",
    *,
    cuda=True,
    extra=None,
    success=True,
    version="12.8",
    variables=None,
    compilers=None,
):
    conan = shutil.which("conan")
    assert conan is not None, "Conan is required for package identity validation"
    command = [
        conan,
        "graph",
        "info",
        str(Path(__file__).resolve().parents[2]),
        "--no-remote",
        "--format=json",
        "-o",
        "&:with_python=False",
        "-o",
        "&:with_collision=False",
        "-o",
        f"&:with_cuda={cuda}",
        "-o",
        f"&:cuda_architectures={architecture}",
        "-o",
        f"&:cuda_version={version}",
    ]
    variables = dict(variables or {})
    if extra is not None:
        variables["HOLISTICMOTION_CUDA_ARCHITECTURES"] = extra
    if variables:
        command += [
            "-c",
            "tools.cmake.cmaketoolchain:extra_variables=" + json.dumps(variables),
        ]
    if compilers is not None:
        command += ["-c", "tools.build:compiler_executables=" + json.dumps(compilers)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    if not success:
        return result.stderr
    return json.loads(result.stdout)["graph"]["nodes"]["0"]


def test_different_gpu_architectures_have_different_package_identities():
    first, second = graph("75"), graph("120")
    assert first["package_id"] != second["package_id"]
    assert first["info"]["options"]["cuda_architectures"] == "75"
    assert second["info"]["options"]["cuda_architectures"] == "120"


@pytest.mark.parametrize("extra", ["75;120", {"value": "75;120", "type": "STRING"}])
def test_existing_cmake_configuration_keys_the_same_binary_as_explicit_options(extra):
    automatic, explicit = graph(extra=extra), graph("120;75;75")
    assert automatic["package_id"] == explicit["package_id"]
    assert automatic["info"]["options"]["cuda_architectures"] == "75;120"


def test_non_cuda_package_identity_does_not_depend_on_architectures():
    first, second = graph("75", cuda=False), graph("120", cuda=False)
    assert first["package_id"] == second["package_id"]
    assert "cuda_architectures" not in first["info"]["options"]


def test_real_and_virtual_code_are_distinct_package_capabilities():
    assert graph("120-real")["package_id"] != graph("120-virtual")["package_id"]


@pytest.mark.parametrize("value", ["native", "all", "all-major", "75;;120", "75-bad"])
def test_unresolved_or_invalid_architectures_have_a_clear_diagnostic(value):
    assert "cuda_architectures must be an explicit" in graph(value, success=False)


def test_conflicting_option_and_cmake_configuration_are_rejected():
    assert "cuda_architectures conflicts" in graph("75", extra="120", success=False)


def test_cmake_architectures_require_a_cmake_list_string():
    assert "cuda_architectures must be an explicit" in graph(
        extra=["75", "120"], success=False
    )


def test_different_toolkit_versions_have_different_binary_identities():
    first, second = graph(version="12.8"), graph(version="13.0")
    assert first["package_id"] != second["package_id"]
    assert first["info"]["options"]["cuda_version"] == "12.8"
    assert second["info"]["options"]["cuda_version"] == "13.0"


def test_cpu_binary_identity_does_not_depend_on_cuda_toolkit():
    first = graph(cuda=False, version="12.8")
    second = graph(cuda=False, version="13.0")
    assert first["package_id"] == second["package_id"]
    assert "cuda_version" not in first["info"]["options"]


@pytest.mark.parametrize("version", ["12", "12.8.1", "12.08", "invalid"])
def test_invalid_toolkit_versions_have_configuration_diagnostics(version):
    assert "cuda_version must be" in graph(version=version)["info_invalid"]


@pytest.mark.parametrize("setting", ["extra", "compiler"])
def test_auto_toolkit_identity_uses_selected_nvcc(setting):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("requires nvcc to validate automatic toolkit detection")
    output = subprocess.check_output([nvcc, "--version"], text=True)
    import re

    version = re.search(r"release\s+(\d+\.\d+)", output)[1]
    configuration = (
        {"variables": {"CMAKE_CUDA_COMPILER": {"value": nvcc, "type": "FILEPATH"}}}
        if setting == "extra"
        else {"compilers": {"cuda": nvcc}}
    )
    automatic = graph(version="auto", **configuration)
    assert automatic["package_id"] == graph(version=version)["package_id"]


def test_auto_toolkit_fails_clearly_for_an_unavailable_compiler():
    assert "Cannot determine CUDA toolkit version" in graph(
        version="auto",
        variables={"CMAKE_CUDA_COMPILER": "/missing/nvcc"},
        success=False,
    )
