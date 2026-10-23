import ast
import os
import shlex
import subprocess
import sys

from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import load


class HolisticMotionTestPackage(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    test_type = "explicit"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        CMakeDeps(self).generate()
        toolchain = CMakeToolchain(self)
        toolchain.variables["HOLISTICMOTION_CONSUMER_TEST_COLLISION"] = (
            self.dependencies["holistic-motion"].options.with_collision
        )
        toolchain.variables["HOLISTICMOTION_CONSUMER_TEST_CUDA"] = self.dependencies[
            "holistic-motion"
        ].options.with_cuda
        toolchain.variables["HOLISTICMOTION_CONSUMER_TEST_DISCOVERY"] = True
        dependency = self.dependencies["holistic-motion"]
        test_python = bool(dependency.options.with_python) and can_run(self)
        toolchain.variables["HOLISTICMOTION_CONSUMER_TEST_PYTHON"] = test_python
        if test_python:
            metadata = ast.parse(
                load(
                    self,
                    os.path.join(
                        dependency.package_folder, "holistic_motion", "_build_config.py"
                    ),
                )
            )
            version = next(
                (
                    ast.literal_eval(node.value)
                    for node in metadata.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "PYTHON_VERSION"
                        for target in node.targets
                    )
                ),
                f"{sys.version_info.major}.{sys.version_info.minor}",
            )
            toolchain.variables["HOLISTICMOTION_CONSUMER_PYTHON_VERSION"] = ".".join(
                version.split(".")[:2]
            )
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if can_run(self):
            self.run(self.cpp.build.bindir + "/trajectory_consumer", env="conanrun")
            self.run(self.cpp.build.bindir + "/coordinate_consumer", env="conanrun")
            self.run(self.cpp.build.bindir + "/fep_consumer", env="conanrun")
            if self.dependencies["holistic-motion"].options.with_collision:
                self.run(self.cpp.build.bindir + "/collision_consumer", env="conanrun")
            if self.dependencies["holistic-motion"].options.with_python:
                dependency = self.dependencies["holistic-motion"]
                executable = load(
                    self, os.path.join(self.build_folder, "python_executable.txt")
                ).strip()
                arguments = [
                    executable,
                    os.path.join(self.source_folder, "src", "python_consumer.py"),
                    dependency.package_folder,
                    "1" if dependency.options.with_collision else "0",
                    "1" if dependency.options.with_cuda else "0",
                ]
                command = (
                    subprocess.list2cmdline(arguments)
                    if self.settings.os == "Windows"
                    else shlex.join(arguments)
                )
                self.run(command, env="conanrun")
