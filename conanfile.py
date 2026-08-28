from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout

required_conan_version = ">=2.28"


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
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_python": True,
        "with_tests": False,
        "with_cuda": False,
        "with_collision": True,
        "pinocchio/*:with_collision_support": True,
        "coal/*:with_octomap": False,
    }
    exports_sources = (
        "CMakeLists.txt",
        "THIRD_PARTY_NOTICES.md",
        "cmake/*",
        "include/*",
        "src/*",
        "tests/*",
        "bindings/*",
        "python/*",
        "!**/__pycache__/*",
        "!**/*.pyc",
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
        toolchain.variables["HOLISTICMOTION_BUILD_PYTHON"] = self.options.with_python
        toolchain.variables["HOLISTICMOTION_BUILD_TESTS"] = self.options.with_tests
        toolchain.variables["HOLISTICMOTION_ENABLE_CUDA"] = self.options.with_cuda
        toolchain.variables["HOLISTICMOTION_ENABLE_COLLISION"] = (
            self.options.with_collision
        )
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()

    def package_info(self):
        if self.options.with_python:
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
        core.set_property(
            "cmake_target_name", "HolisticMotion::holistic_motion"
        )
        if self.options.with_collision:
            collision = self.cpp_info.components["collision"]
            collision.libs = ["holistic_motion_collision"]
            collision.requires = [
                "eigen::eigen",
                "pinocchio::pinocchio_parsers",
                "pinocchio::pinocchio_collision",
                "coal::coal",
            ]
            collision.set_property(
                "cmake_target_name", "HolisticMotion::collision"
            )
        if self.options.with_cuda:
            core.set_property(
                "cmake_build_modules",
                ["lib/cmake/HolisticMotion/HolisticMotionConanDeps.cmake"],
            )
