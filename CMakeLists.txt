cmake_minimum_required(VERSION 3.22)
option(HOLISTICMOTION_PYTHON_TOOLKIT_ONLY
       "Install the pure Python toolkit without compiling native libraries" OFF)
if(HOLISTICMOTION_PYTHON_TOOLKIT_ONLY)
    project(HolisticMotion VERSION 0.1.0 LANGUAGES NONE)
    include(cmake/modules/HolisticMotionPythonToolkit.cmake)
    return()
endif()
project(HolisticMotion VERSION 0.1.0 LANGUAGES CXX)
include(cmake/modules/HolisticMotionCompilerCheck.cmake)
holistic_motion_check_conan_compiler()

option(HOLISTICMOTION_BUILD_PYTHON "Build the Python extension" ON)
option(HOLISTICMOTION_BUILD_TESTS "Build C++ tests" OFF)
option(HOLISTICMOTION_BUILD_BENCHMARKS "Build standalone benchmarks" OFF)
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
    set(HOLISTICMOTION_NATIVE_CUDA_VERSION
        "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
    if(DEFINED HOLISTICMOTION_CUDA_VERSION AND
       NOT HOLISTICMOTION_CUDA_VERSION STREQUAL HOLISTICMOTION_NATIVE_CUDA_VERSION)
        message(FATAL_ERROR
            "CUDA toolkit ${HOLISTICMOTION_NATIVE_CUDA_VERSION} does not match "
            "the Conan package identity ${HOLISTICMOTION_CUDA_VERSION}")
    endif()
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
                pinocchio::pinocchio_collision coal::coal urdfdom::urdfdom)
    # Pinocchio 3.8 still includes Coal through the legacy hpp/fcl
    # compatibility path. Coal explicitly provides this definition for
    # consumers that cannot update the transitive include themselves.
    target_compile_definitions(holistic_motion_collision PRIVATE
        COAL_DISABLE_HPP_FCL_WARNINGS)
    target_compile_options(holistic_motion_collision PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
endif()

include(cmake/modules/HolisticMotionBuildIdentity.cmake)
holistic_motion_use_build_identity(holistic_motion)
if(HOLISTICMOTION_ENABLE_COLLISION)
    holistic_motion_use_build_identity(holistic_motion_collision)
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
    add_executable(fep_selection_benchmark benchmarks/fep_selection_benchmark.cpp)
    target_include_directories(fep_selection_benchmark PRIVATE src/kinematics/fep)
    target_link_libraries(fep_selection_benchmark PRIVATE holistic_motion)
    add_executable(numerical_ik_benchmark benchmarks/numerical_ik_benchmark.cpp)
    target_link_libraries(numerical_ik_benchmark PRIVATE holistic_motion)
    add_executable(forward_kinematics_benchmark benchmarks/forward_kinematics_benchmark.cpp)
    target_link_libraries(forward_kinematics_benchmark PRIVATE holistic_motion)
    add_executable(joint_space_snapshot_benchmark benchmarks/joint_space_snapshot_benchmark.cpp)
    target_link_libraries(joint_space_snapshot_benchmark PRIVATE holistic_motion)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_definitions(joint_space_snapshot_benchmark PRIVATE HOLISTICMOTION_COUNT_MALLOC)
        target_link_options(joint_space_snapshot_benchmark PRIVATE -Wl,--wrap=malloc)
    endif()
    add_executable(nearest_ik_benchmark benchmarks/nearest_ik_benchmark.cpp)
    target_link_libraries(nearest_ik_benchmark PRIVATE holistic_motion)
    add_executable(ik_limit_benchmark benchmarks/ik_limit_benchmark.cpp)
    target_link_libraries(ik_limit_benchmark PRIVATE holistic_motion)
    add_executable(robot_model_benchmark benchmarks/robot_model_benchmark.cpp)
    target_link_libraries(robot_model_benchmark PRIVATE holistic_motion)
    add_executable(path_optimizer_benchmark benchmarks/path_optimizer_benchmark.cpp)
    target_link_libraries(path_optimizer_benchmark PRIVATE holistic_motion)
    add_executable(trajectory_construction_benchmark benchmarks/trajectory_construction_benchmark.cpp)
    target_link_libraries(trajectory_construction_benchmark PRIVATE holistic_motion)
    add_executable(trajectory_cap_benchmark benchmarks/trajectory_cap_benchmark.cpp)
    target_link_libraries(trajectory_cap_benchmark PRIVATE holistic_motion)
    add_executable(double_s_profile_audit benchmarks/double_s_profile_audit.cpp)
    target_link_libraries(double_s_profile_audit PRIVATE holistic_motion)
    add_executable(trajectory_query_benchmark benchmarks/trajectory_query_benchmark.cpp)
    target_link_libraries(trajectory_query_benchmark PRIVATE holistic_motion)
    add_executable(trajectory_report_benchmark benchmarks/trajectory_report_benchmark.cpp)
    target_link_libraries(trajectory_report_benchmark PRIVATE holistic_motion)
    add_executable(trajectory_sampling_benchmark benchmarks/trajectory_sampling_benchmark.cpp)
    target_link_libraries(trajectory_sampling_benchmark PRIVATE holistic_motion)
    add_executable(path_construction_benchmark benchmarks/path_construction_benchmark.cpp)
    target_link_libraries(path_construction_benchmark PRIVATE holistic_motion)
endif()

if(HOLISTICMOTION_BUILD_PYTHON)
    set(HOLISTICMOTION_PYTHON_VERSION "" CACHE STRING
        "Required Python major.minor version; empty selects Python 3.9 or newer")
    set(HOLISTICMOTION_PYTHON_TOOLKIT False)
    set(HOLISTICMOTION_PYTHON_COLLISION False)
    set(HOLISTICMOTION_PYTHON_CUDA False)
    if(HOLISTICMOTION_ENABLE_COLLISION)
        set(HOLISTICMOTION_PYTHON_COLLISION True)
    endif()
    if(HOLISTICMOTION_ENABLE_CUDA)
        set(HOLISTICMOTION_PYTHON_CUDA True)
    endif()
    if(HOLISTICMOTION_PYTHON_VERSION)
        find_package(Python3 ${HOLISTICMOTION_PYTHON_VERSION} EXACT REQUIRED
            COMPONENTS Interpreter Development.Module)
    else()
        find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
    endif()
    set(HOLISTICMOTION_PYTHON_NATIVE_VERSION "${Python3_VERSION}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -I -S
            "${PROJECT_SOURCE_DIR}/python/holistic_motion/_python_abi.py"
        RESULT_VARIABLE _python_abi_status OUTPUT_VARIABLE _python_identity
        ERROR_VARIABLE _python_abi_error OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _python_abi_status EQUAL 0)
        message(FATAL_ERROR "Cannot determine Python ABI: ${_python_abi_error}")
    endif()
    string(JSON HOLISTICMOTION_PYTHON_NATIVE_ABI GET "${_python_identity}" abi)
    if(DEFINED HOLISTICMOTION_PYTHON_ABI AND
       NOT HOLISTICMOTION_PYTHON_ABI STREQUAL HOLISTICMOTION_PYTHON_NATIVE_ABI)
        message(FATAL_ERROR
            "Python ABI ${HOLISTICMOTION_PYTHON_NATIVE_ABI} does not match "
            "the Conan package identity ${HOLISTICMOTION_PYTHON_ABI}")
    endif()
    set(HOLISTICMOTION_PYTHON_PINOCCHIO_VERSION "")
    if(HOLISTICMOTION_ENABLE_COLLISION)
        set(HOLISTICMOTION_PYTHON_PINOCCHIO_VERSION "${pinocchio_VERSION_STRING}")
    endif()
    configure_file(cmake/python_build_config.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/_holistic_motion_build_config.py @ONLY)
    find_package(pybind11 REQUIRED CONFIG)
    pybind11_add_module(_holistic_motion
        ${HOLISTICMOTION_PYTHON_BINDING_SOURCES})
    target_link_libraries(_holistic_motion PRIVATE holistic_motion)
    set(_holisticmotion_python_targets _holistic_motion)
    if(HOLISTICMOTION_ENABLE_COLLISION)
        # Keep Conan Pinocchio/Coal out of the core extension's dependency graph.
        # Existing Python collision names and factories load this target on use.
        target_compile_definitions(_holistic_motion PRIVATE
            HOLISTICMOTION_HAS_COLLISION=1)
        pybind11_add_module(_holistic_motion_collision
            ${HOLISTICMOTION_PYTHON_COLLISION_BINDING_SOURCES})
        target_link_libraries(_holistic_motion_collision PRIVATE
            holistic_motion holistic_motion_collision)
        target_compile_definitions(_holistic_motion_collision PRIVATE
            HOLISTICMOTION_HAS_COLLISION=1)
        list(APPEND _holisticmotion_python_targets _holistic_motion_collision)
    endif()
    foreach(_python_target IN LISTS _holisticmotion_python_targets)
        holistic_motion_use_build_identity(${_python_target})
        target_include_directories(${_python_target} PRIVATE "${PROJECT_SOURCE_DIR}/src")
        if(WIN32 AND HOLISTICMOTION_PYTHON_NATIVE_ABI MATCHES "^cpython-(d?t)(-|$)")
            # Select the layout reported by the interpreter's sysconfig probe.
            target_compile_definitions(${_python_target} PRIVATE Py_GIL_DISABLED=1)
        endif()
        target_compile_options(${_python_target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
        if(APPLE)
            set_target_properties(${_python_target} PROPERTIES
                INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
        elseif(UNIX)
            set_target_properties(${_python_target} PROPERTIES
                INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        endif()
        install(TARGETS ${_python_target} LIBRARY DESTINATION holistic_motion)
    endforeach()
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_holistic_motion_build_config.py
        DESTINATION holistic_motion RENAME _build_config.py)
    install(DIRECTORY python/holistic_motion/
            DESTINATION holistic_motion
            FILES_MATCHING PATTERN "*.py")
endif()

# Include the selected interpreter and pybind11 ABI after Python discovery.
holistic_motion_configure_build_identity()

include(CMakePackageConfigHelpers)
install(TARGETS holistic_motion EXPORT HolisticMotionCoreTargets)
if(HOLISTICMOTION_ENABLE_COLLISION)
    install(TARGETS holistic_motion_collision EXPORT HolisticMotionCollisionTargets)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_INSTALL_RPATH_USE_LINK_PATH)
    # scripts/build.sh creates a local install tied to its Conan cache. Its
    # per-library runtime paths avoid overriding Python dependency runtimes.
    install(CODE [[
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.holistic_motion_local_runtime"
             "Dependencies use this local install's runtime paths.\n")
    ]])
else()
    install(CODE [[
        file(REMOVE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.holistic_motion_local_runtime")
    ]])
endif()
install(DIRECTORY include/holistic_motion/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holistic_motion
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(EXPORT HolisticMotionCoreTargets NAMESPACE HolisticMotion::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HolisticMotion)
if(HOLISTICMOTION_ENABLE_COLLISION)
    install(EXPORT HolisticMotionCollisionTargets NAMESPACE HolisticMotion::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HolisticMotion)
endif()
configure_file(cmake/HolisticMotionTargets.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HolisticMotionTargets.cmake @ONLY)
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
    ${CMAKE_CURRENT_BINARY_DIR}/HolisticMotionTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HolisticMotion)
configure_file(cmake/HolisticMotionConanDeps.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HolisticMotionConanDeps.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/HolisticMotionConanDeps.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HolisticMotion)
install(FILES THIRD_PARTY_NOTICES.md
        DESTINATION ${CMAKE_INSTALL_DATADIR}/HolisticMotion)
