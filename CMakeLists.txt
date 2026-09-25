cmake_minimum_required(VERSION 3.22)
project(HolisticMotion VERSION 0.1.0 LANGUAGES CXX)
include(cmake/modules/HolisticMotionCompilerCheck.cmake)
holistic_motion_check_conan_compiler()

option(HOLISTICMOTION_BUILD_PYTHON "Build the Python extension" ON)
option(HOLISTICMOTION_BUILD_TESTS "Build C++ tests" OFF)
option(HOLISTICMOTION_BUILD_BENCHMARKS "Build standalone CPU benchmarks" OFF)
option(HOLISTICMOTION_ENABLE_CUDA "Build optional CUDA batch backends" OFF)
option(HOLISTICMOTION_ENABLE_COLLISION
       "Build the optional Pinocchio/Coal collision component" ON)
set(HOLISTICMOTION_CUDA_ARCHITECTURES "75;80;86;89;120" CACHE STRING
    "CUDA architectures for optional batch backends")
include(GNUInstallDirs)
include(cmake/modules/HolisticMotionSources.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(cmake/modules/HolisticMotionSanitizers.cmake)

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
    set_target_properties(holistic_motion_collision PROPERTIES
        EXPORT_NAME collision)
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
    include(cmake/modules/HolisticMotionTests.cmake)
    holistic_motion_register_tests()
endif()

if(HOLISTICMOTION_BUILD_BENCHMARKS)
    add_executable(sampling_planner_benchmark benchmarks/sampling_planner_benchmark.cpp)
    target_link_libraries(sampling_planner_benchmark PRIVATE holistic_motion)
    add_executable(srs_null_space_benchmark benchmarks/srs_null_space_benchmark.cpp)
    target_link_libraries(srs_null_space_benchmark PRIVATE holistic_motion)
    add_executable(fep_batch_benchmark benchmarks/fep_batch_benchmark.cpp)
    target_link_libraries(fep_batch_benchmark PRIVATE holistic_motion)
    add_executable(numerical_ik_benchmark benchmarks/numerical_ik_benchmark.cpp)
    target_link_libraries(numerical_ik_benchmark PRIVATE holistic_motion)
    add_executable(ik_limit_benchmark benchmarks/ik_limit_benchmark.cpp)
    target_link_libraries(ik_limit_benchmark PRIVATE holistic_motion)
    add_executable(robot_model_benchmark benchmarks/robot_model_benchmark.cpp)
    target_link_libraries(robot_model_benchmark PRIVATE holistic_motion)
    add_executable(path_optimizer_benchmark benchmarks/path_optimizer_benchmark.cpp)
    target_link_libraries(path_optimizer_benchmark PRIVATE holistic_motion)
    add_executable(trajectory_construction_benchmark benchmarks/trajectory_construction_benchmark.cpp)
    target_link_libraries(trajectory_construction_benchmark PRIVATE holistic_motion)
    add_executable(double_s_profile_audit benchmarks/double_s_profile_audit.cpp)
    target_link_libraries(double_s_profile_audit PRIVATE holistic_motion)
    add_executable(trajectory_query_benchmark benchmarks/trajectory_query_benchmark.cpp)
    target_link_libraries(trajectory_query_benchmark PRIVATE holistic_motion)
    add_executable(trajectory_report_benchmark benchmarks/trajectory_report_benchmark.cpp)
    target_link_libraries(trajectory_report_benchmark PRIVATE holistic_motion)
    add_executable(path_construction_benchmark benchmarks/path_construction_benchmark.cpp)
    target_link_libraries(path_construction_benchmark PRIVATE holistic_motion)
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
install(FILES cmake/HolisticMotionConanDeps.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HolisticMotion)
install(FILES THIRD_PARTY_NOTICES.md
        DESTINATION ${CMAKE_INSTALL_DATADIR}/HolisticMotion)
