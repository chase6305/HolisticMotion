from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


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
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if can_run(self):
            self.run(self.cpp.build.bindir + "/trajectory_consumer", env="conanrun")
            if self.dependencies["holistic-motion"].options.with_collision:
                self.run(
                    self.cpp.build.bindir + "/collision_consumer", env="conanrun"
                )
