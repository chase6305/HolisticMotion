import json
import os
import re
import runpy
import shutil
import subprocess
import sys

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout

required_conan_version = ">=2.28"


def _cuda_architectures(value):
    if isinstance(value, dict):
        value = value.get("value")
    parts = str(value).split(";")
    if not parts or any(
        re.fullmatch(r"[1-9]\d*(?:-(?:real|virtual))?", part.strip()) is None
        for part in parts
    ):
        raise ConanInvalidConfiguration(
            "cuda_architectures must be an explicit semicolon-separated list "
            "such as '75;120' or '120-real;120-virtual'; compiler/device-resolved "
            "keywords require a standalone CMake build"
        )
    parts = set(part.strip() for part in parts)
    return ";".join(sorted(parts, key=lambda part: (int(part.split("-")[0]), part)))


class HolisticMotionConan(ConanFile):
    name = "holistic-motion"
    version = "0.1.0"
    package_type = "library"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_python": [True, False],
        "with_tests": [True, False],
        "with_cuda": [True, False],
        "with_collision": [True, False],
        "python_version": ["ANY"],
        "python_abi": ["ANY"],
        "cuda_architectures": ["ANY"],
        "cuda_version": ["ANY"],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_python": True,
        "with_tests": False,
        "with_cuda": False,
        "with_collision": True,
        "python_version": "auto",
        "python_abi": "auto",
        "cuda_architectures": "auto",
        "cuda_version": "auto",
        "pinocchio/*:with_collision_support": True,
        "coal/*:with_octomap": False,
    }
    exports = "python/holistic_motion/_python_abi.py"
    exports_sources = (
        "CMakeLists.txt",
        "THIRD_PARTY_NOTICES.md",
        "cmake/*",
        "include/*",
        "src/*",
        "tests/*",
        "bindings/*",
        "benchmarks/*",
        "python/*",
        "!**/__pycache__/*",
        "!**/*.pyc",
    )

    def configure(self):
        if self.options.with_cuda:
            variables = self.conf.get(
                "tools.cmake.cmaketoolchain:extra_variables",
                default={},
                check_type=dict,
            )
            configured = variables.get("HOLISTICMOTION_CUDA_ARCHITECTURES")
            architecture = str(self.options.cuda_architectures)
            if architecture == "auto":
                architecture = _cuda_architectures(
                    configured if configured is not None else "75;80;86;89;120"
                )
            else:
                architecture = _cuda_architectures(architecture)
                if (
                    configured is not None
                    and _cuda_architectures(configured) != architecture
                ):
                    raise ConanInvalidConfiguration(
                        "cuda_architectures conflicts with "
                        "HOLISTICMOTION_CUDA_ARCHITECTURES in extra_variables"
                    )
            self._resolved_cuda_architectures = architecture
            self._resolved_cuda_version = str(self.options.cuda_version)
            if self._resolved_cuda_version == "auto":
                compilers = self.conf.get(
                    "tools.build:compiler_executables", default={}, check_type=dict
                )
                compiler = variables.get("CMAKE_CUDA_COMPILER")
                if isinstance(compiler, dict):
                    compiler = compiler.get("value")
                compiler = (
                    compiler or compilers.get("cuda") or os.environ.get("CUDACXX")
                )
                compiler = compiler or shutil.which("nvcc")
                if not compiler:
                    raise ConanInvalidConfiguration(
                        "cuda_version=auto requires nvcc; set cuda_version to the "
                        "package's major.minor toolkit version when consuming a binary"
                    )
                try:
                    output = subprocess.check_output(
                        [compiler, "--version"],
                        text=True,
                        stderr=subprocess.STDOUT,
                        timeout=10,
                    )
                    match = re.search(r"release\s+(\d+\.\d+)", output)
                    if match is None:
                        raise ValueError("nvcc release was not reported")
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    raise ConanInvalidConfiguration(
                        f"Cannot determine CUDA toolkit version from {compiler}; "
                        "select nvcc or set cuda_version explicitly"
                    ) from error
                self._resolved_cuda_version = match[1]
                self._resolved_cuda_compiler = os.path.abspath(
                    shutil.which(compiler) or compiler
                )
        else:
            self.options.rm_safe("cuda_architectures")
            self.options.rm_safe("cuda_version")
        if not self.options.with_python:
            self.options.rm_safe("python_version")
            self.options.rm_safe("python_abi")
            return
        self._resolved_python_version = str(self.options.python_version)
        self._resolved_python_abi = str(self.options.python_abi)
        variables = self.conf.get(
            "tools.cmake.cmaketoolchain:extra_variables", default={}, check_type=dict
        )
        executable = variables.get("Python3_EXECUTABLE")
        if isinstance(executable, dict):
            executable = executable.get("value")
        if executable is not None and (
            not isinstance(executable, str) or not executable
        ):
            raise ConanInvalidConfiguration(
                "Python3_EXECUTABLE must name an interpreter"
            )
        current_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        probe = os.path.join(
            self.recipe_folder, "python", "holistic_motion", "_python_abi.py"
        )
        self._conan_python_abi = runpy.run_path(probe)["python_abi"]()
        probe_abi = self._resolved_python_abi == "auto" and (
            executable is not None or self._resolved_python_version == current_minor
        )
        if self._resolved_python_version == "auto" or probe_abi:
            executable = executable or sys.executable
            try:
                output = subprocess.check_output(
                    [executable, "-I", "-S", probe],
                    text=True,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                )
                identity = json.loads(output)
                if self._resolved_python_version == "auto":
                    self._resolved_python_version = identity["version"]
                if self._resolved_python_abi == "auto":
                    self._resolved_python_abi = identity["abi"]
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
                raise ConanInvalidConfiguration(
                    f"Cannot determine Python version/ABI from {executable}; select an "
                    "available Python3_EXECUTABLE or set python_version and python_abi explicitly"
                ) from error
        if self._resolved_python_abi == "auto":
            # FindPython3 defaults to standard CPython when only a different
            # minor is requested. Explicit interpreter/ABI selection handles
            # other variants; CMake always verifies its actual interpreter.
            self._resolved_python_abi = "cpython"

    def package_id(self):
        if self.info.options.get_safe("with_cuda"):
            self.info.options.cuda_architectures = self._resolved_cuda_architectures
            self.info.options.cuda_version = self._resolved_cuda_version
        if self.info.options.get_safe("with_python"):
            self.info.options.python_version = self._resolved_python_version
            self.info.options.python_abi = self._resolved_python_abi

    def validate(self):
        if (
            self.options.with_cuda
            and re.fullmatch(r"[1-9]\d*\.(?:0|[1-9]\d*)", self._resolved_cuda_version)
            is None
        ):
            raise ConanInvalidConfiguration(
                "cuda_version must be a toolkit major.minor version, such as 12.8"
            )
        if self.options.with_python:
            if re.fullmatch(r"[a-z][a-z0-9_.-]*", self._resolved_python_abi) is None:
                raise ConanInvalidConfiguration(
                    "python_abi must be an interpreter ABI tag"
                )
            version = self._resolved_python_version
            match = re.fullmatch(r"3\.(\d+)", version)
            if match is None or int(match[1]) < 9:
                raise ConanInvalidConfiguration(
                    "python_version must be a supported major.minor version (3.9 or newer)"
                )

    def requirements(self):
        self.requires("eigen/3.4.0", transitive_headers=True)
        self.requires("fmt/10.2.1", transitive_headers=True)
        self.requires("spdlog/1.14.1", transitive_headers=True)
        self.requires("urdfdom/4.0.0", transitive_headers=True)
        if self.options.with_python:
            self.requires("pybind11/2.13.6")
        if self.options.with_collision:
            # Keep both collision dependencies explicit so Conan owns their
            # versions and makes them available to CMakeDeps and consumers.
            self.requires("pinocchio/3.8.0", transitive_headers=True)
            self.requires("coal/3.0.2", transitive_headers=True)
            # Coal 3.0.2 requires Boost 1.88 while Pinocchio accepts
            # Boost >=1.84,<1.90. Resolve both edges to Coal's ABI version.
            self.requires("boost/1.88.0", override=True)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        toolchain = CMakeToolchain(self)
        toolchain.variables["HOLISTICMOTION_CONAN_COMPILER"] = str(
            self.settings.compiler
        )
        toolchain.variables["HOLISTICMOTION_CONAN_COMPILER_VERSION"] = str(
            self.settings.compiler.version
        )
        toolchain.variables["HOLISTICMOTION_BUILD_PYTHON"] = self.options.with_python
        toolchain.variables["HOLISTICMOTION_BUILD_TESTS"] = self.options.with_tests
        toolchain.variables["HOLISTICMOTION_ENABLE_CUDA"] = self.options.with_cuda
        if self.options.with_cuda:
            toolchain.variables["HOLISTICMOTION_CUDA_ARCHITECTURES"] = (
                self._resolved_cuda_architectures
            )
            toolchain.variables["HOLISTICMOTION_CUDA_VERSION"] = (
                self._resolved_cuda_version
            )
            if hasattr(self, "_resolved_cuda_compiler"):
                toolchain.variables["CMAKE_CUDA_COMPILER"] = (
                    self._resolved_cuda_compiler
                )
        toolchain.variables["HOLISTICMOTION_ENABLE_COLLISION"] = (
            self.options.with_collision
        )
        if self.options.with_python:
            version = self._resolved_python_version
            toolchain.variables["HOLISTICMOTION_PYTHON_VERSION"] = version
            toolchain.variables["HOLISTICMOTION_PYTHON_ABI"] = self._resolved_python_abi
            variables = self.conf.get(
                "tools.cmake.cmaketoolchain:extra_variables",
                default={},
                check_type=dict,
            )
            # The default ABI follows the interpreter running Conan. An explicit
            # different minor lets FindPython3 locate that installed interpreter.
            if (
                "Python3_EXECUTABLE" not in variables
                and version == f"{sys.version_info.major}.{sys.version_info.minor}"
                and self._resolved_python_abi == self._conan_python_abi
            ):
                toolchain.variables["Python3_EXECUTABLE"] = sys.executable
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()

    def package_info(self):
        if self.options.with_python:
            self.runenv_info.prepend_path("PYTHONPATH", self.package_folder)
            # pybind11 is only needed to compile the extension; consumers load
            # the packaged module and do not need pybind11 on their link graph.
            self.cpp_info.ignored_requires.append("pybind11")
        core = self.cpp_info.components["core"]
        core.libs = ["holistic_motion"]
        core.requires = [
            "eigen::eigen",
            "fmt::fmt",
            "spdlog::spdlog",
            "urdfdom::urdfdom",
        ]
        self.cpp_info.set_property("cmake_file_name", "HolisticMotion")
        core.set_property("cmake_target_name", "HolisticMotion::holistic_motion")
        if self.options.with_collision:
            collision = self.cpp_info.components["collision"]
            collision.libs = ["holistic_motion_collision"]
            collision.requires = [
                "eigen::eigen",
                "pinocchio::pinocchio_parsers",
                "pinocchio::pinocchio_collision",
                "coal::coal",
                "urdfdom::urdfdom",
            ]
            collision.set_property("cmake_target_name", "HolisticMotion::collision")
        if self.options.with_cuda:
            # CMakeDeps reads build modules from package-level properties,
            # not component properties. The module gates CUDA on core selection.
            self.cpp_info.set_property(
                "cmake_build_modules",
                ["lib/cmake/HolisticMotion/HolisticMotionConanDeps.cmake"],
            )
