cmake_minimum_required(VERSION 3.22)
project(HolisticMotion VERSION 0.1.0 LANGUAGES CXX)

option(HOLISTICMOTION_BUILD_PYTHON "Build the Python extension" ON)
option(HOLISTICMOTION_BUILD_TESTS "Build C++ tests" OFF)
option(HOLISTICMOTION_ENABLE_CUDA "Build optional CUDA batch backends" ON)
option(HOLISTICMOTION_ENABLE_COLLISION
       "Build the optional Pinocchio/Coal collision component" ON)
set(HOLISTICMOTION_CUDA_ARCHITECTURES "75;80;86;89;120" CACHE STRING
    "CUDA architectures for optional batch backends")
include(GNUInstallDirs)
include(cmake/modules/HolisticMotionSources.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED CONFIG)
find_package(fmt REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(urdfdom REQUIRED CONFIG)

add_library(holistic_motion ${HOLISTICMOTION_CORE_SOURCES})
if(HOLISTICMOTION_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(holistic_motion PRIVATE ${HOLISTICMOTION_CUDA_SOURCES})
    target_link_libraries(holistic_motion PUBLIC CUDA::cudart)
    target_compile_definitions(holistic_motion PUBLIC HOLISTICMOTION_HAS_CUDA=1)
    set_target_properties(holistic_motion PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${HOLISTICMOTION_CUDA_ARCHITECTURES}")
    target_compile_options(holistic_motion PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
add_library(HolisticMotion::holistic_motion ALIAS holistic_motion)

target_include_directories(holistic_motion
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(holistic_motion
    PUBLIC Eigen3::Eigen fmt::fmt spdlog::spdlog urdfdom::urdfdom)
target_compile_options(holistic_motion PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)

if(HOLISTICMOTION_ENABLE_COLLISION)
    find_package(pinocchio REQUIRED CONFIG)
    find_package(coal REQUIRED CONFIG)
    add_library(holistic_motion_collision SHARED
        ${HOLISTICMOTION_COLLISION_SOURCES})
    add_library(HolisticMotion::collision ALIAS holistic_motion_collision)
    target_include_directories(holistic_motion_collision PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(holistic_motion_collision
        PUBLIC Eigen3::Eigen
        PRIVATE pinocchio::pinocchio_parsers
                pinocchio::pinocchio_collision coal::coal)
    # Pinocchio 3.8 still includes Coal through the legacy hpp/fcl
    # compatibility path. Coal explicitly provides this definition for
    # consumers that cannot update the transitive include themselves.
    target_compile_definitions(holistic_motion_collision PRIVATE
        COAL_DISABLE_HPP_FCL_WARNINGS)
    target_compile_options(holistic_motion_collision PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
endif()

if(HOLISTICMOTION_BUILD_TESTS)
    enable_testing()
    add_executable(trajectory_smoke_test
        tests/cpp/trajectory/trajectory_smoke.cpp)
    target_link_libraries(trajectory_smoke_test PRIVATE holistic_motion)
    target_compile_options(trajectory_smoke_test PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
    add_test(NAME trajectory_smoke COMMAND trajectory_smoke_test)
    add_executable(analytic_kinematics_smoke_test
        tests/cpp/kinematics/analytic_kinematics_smoke.cpp)
    target_link_libraries(analytic_kinematics_smoke_test PRIVATE holistic_motion)
    target_compile_options(analytic_kinematics_smoke_test PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
    add_test(NAME analytic_kinematics_smoke COMMAND analytic_kinematics_smoke_test)
    if(HOLISTICMOTION_ENABLE_COLLISION)
        add_executable(collision_smoke_test
            tests/cpp/collision/collision_smoke.cpp)
        target_link_libraries(collision_smoke_test PRIVATE
            holistic_motion_collision)
        target_compile_options(collision_smoke_test PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
        add_test(NAME collision_smoke COMMAND collision_smoke_test)
    endif()
endif()

if(HOLISTICMOTION_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 REQUIRED CONFIG)
    pybind11_add_module(_holistic_motion
        ${HOLISTICMOTION_PYTHON_BINDING_SOURCES})
    target_link_libraries(_holistic_motion PRIVATE holistic_motion)
    if(HOLISTICMOTION_ENABLE_COLLISION)
        target_link_libraries(_holistic_motion PRIVATE holistic_motion_collision)
        target_compile_definitions(_holistic_motion PRIVATE
            HOLISTICMOTION_HAS_COLLISION=1)
    endif()
    target_compile_options(_holistic_motion PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
    if(APPLE)
        set_target_properties(_holistic_motion PROPERTIES
            INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
    elseif(UNIX)
        set_target_properties(_holistic_motion PROPERTIES
            INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    install(TARGETS _holistic_motion LIBRARY DESTINATION holistic_motion)
    install(DIRECTORY python/holistic_motion/
            DESTINATION holistic_motion
            FILES_MATCHING PATTERN "*.py")
endif()

include(CMakePackageConfigHelpers)
install(TARGETS holistic_motion EXPORT HolisticMotionTargets)
if(HOLISTICMOTION_ENABLE_COLLISION)
    install(TARGETS holistic_motion_collision EXPORT HolisticMotionTargets)
endif()
install(DIRECTORY include/holistic_motion/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holistic_motion
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(EXPORT HolisticMotionTargets NAMESPACE HolisticMotion::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HolisticMotion)
configure_package_config_file(
    cmake/HolisticMotionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HolisticMotionConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HolisticMotion)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HolisticMotionConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HolisticMotionConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HolisticMotionConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HolisticMotion)
